//! How the actors of the world are drawn.
//!
//! A sprite set is one way of drawing: the frames its actors are drawn from and
//! the animations that may change an actor's appearance. A set is described by
//! `assets/sprites/<set>.ron`, and its frames are the files of
//! `assets/textures/actor/<set>`, so that a set and its frames cannot drift
//! apart.
//!
//! The names of a set and of a frame are only looked up when a data file is
//! read; everything that runs holds the ids they resolve to.

use super::{PlayerControlled, Position, Rotation};
use crate::components;
use crate::ecs::*;
use crate::game::ModRegistry;
use crate::render::*;
use crate::resources;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::{Id, OnceInit};
use crate::world::{Lighting, WorldLightsLayout};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use log::error;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::marker::PhantomData;
use std::mem::offset_of;
use wgpu::*;

/// The sprite sets, by name.
static SPRITE_SETS: OnceInit<Registry<SpriteSet>> = OnceInit::new();

components! {
    /// The layers an actor is drawn with.
    ///
    /// A layer is referenced by its position in this vector. Removing a layer
    /// leaves an empty slot instead of moving the layers after it, so the
    /// layers an animation names keep their identity while an actor is alive.
    #[derive(Clone, Serialize, Deserialize)]
    pub struct Appearance(Vec<Option<Sprite>>): Hot, asset, save;
}

/// One layer of an actor's appearance.
///
/// The size is the width and the height the frame is drawn at, in blocks. The
/// offset places the bottom center of the layer relative to the position of
/// the actor in the billboard's local frame: positive z points towards the
/// camera. The uv rectangle selects an area within the frame.
///
/// The frame and uv are state: animations write them and the renderer reads
/// them.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(into = "RawSprite")]
#[serde(from = "RawSprite")]
pub struct Sprite {
    pub set: Id,
    pub frame: Id,
    pub size: Vec2,
    pub offset: Vec3,
    pub uvs: AABB<Vec2>,
}

/// An animation of an actor's appearance, run by the frame system `S`.
///
/// `layers` contains the slots of the [`Appearance`] the system writes. The
/// system type is part of the component type, so the system can declare access
/// to this component directly and different animations remain distinct to the
/// scheduler.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Animation<S: System<2>> {
    pub layers: Vec<usize>,
    #[serde(skip)]
    _system: PhantomData<S>,
}

impl<S: System<2>> Animation<S> {
    /// Creates an animation that writes the named slots of an appearance.
    pub fn new(layers: Vec<usize>) -> Self {
        Self {
            layers,
            _system: PhantomData,
        }
    }
}

impl<S: System<2>> Clone for Animation<S> {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            _system: PhantomData,
        }
    }
}

impl<S: System<2>> Component for Animation<S> {
    const SETTINGS: ComponentSettings = ComponentSettings {
        storage: StorageType::Hot,
        asset: Some(asset_of::<Self>()),
        save: Some(codec_of::<Self>(Self::SAVE_VERSION)),
    };
}

/// One set of sprites: the frames its actors are drawn from and the animations
/// that may change their appearance.
pub struct SpriteSet {
    /// The frames of the set, one per file of its directory, as the layers of
    /// one texture array. The pixels of the frames are dropped once they are
    /// uploaded, so only the ids of the set stay in memory.
    pub frames: BindSet<TextureArraySampler>,
    pub frame_handles: Registry<()>,
    /// The frame systems that may animate actors of this set.
    pub animations: Animations,
}

/// The animations a sprite set is compatible with.
///
/// This is only a whitelist. The behavior stays in the system that runs it,
/// instead of in a callback that bypasses the system's declared access.
pub struct Animations(Vec<TypeId>);

impl Animations {
    /// Returns whether `S` is one of the animations of the set.
    #[inline]
    pub fn contains<S: 'static>(&self) -> bool {
        self.0.contains(&TypeId::of::<S>())
    }

    /// Turns the system names of a data file into the type ids a set holds.
    fn resolve(frame: &SystemManager<2>, names: &[String]) -> anyhow::Result<Self> {
        let ids = names
            .iter()
            .map(|name| {
                frame
                    .type_id_of(name)
                    .ok_or_else(|| anyhow::anyhow!("frame system {name} is not registered"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self(ids))
    }
}

/// A sprite as it is written to a data file: the names of its set and of the
/// frame, so that the sets can be reordered without breaking the files, and the
/// size, placement and uv area it is drawn at, which are not part of the
/// frames.
#[derive(Serialize, Deserialize)]
struct RawSprite<'a> {
    set: &'a str,
    frame: &'a str,
    size: Vec2,
    #[serde(default)]
    offset: Vec3,
    #[serde(default = "full_uv")]
    uvs: AABB<Vec2>,
}

fn full_uv() -> AABB<Vec2> {
    AABB {
        min: Vec2::ZERO,
        max: Vec2::ONE,
    }
}

/// A set as it is written to a data file.
#[derive(Deserialize)]
struct RawSpriteSet {
    /// The systems that may animate an actor of the set.
    #[serde(default)]
    animations: Vec<String>,
}

impl From<RawSprite<'_>> for Sprite {
    fn from(value: RawSprite) -> Self {
        let set = SPRITE_SETS.id_of(value.set);
        let frame = SPRITE_SETS.get(set).frame_handles.id_of(value.frame);

        Self {
            set,
            frame,
            size: value.size,
            offset: value.offset,
            uvs: value.uvs,
        }
    }
}

impl From<Sprite> for RawSprite<'_> {
    fn from(value: Sprite) -> Self {
        let set = SPRITE_SETS.name_of(value.set);
        let frame = SPRITE_SETS
            .get(value.set)
            .frame_handles
            .name_of(value.frame);

        Self {
            set,
            frame,
            size: value.size,
            offset: value.offset,
            uvs: value.uvs,
        }
    }
}

/// Loads the sprite sets.
///
/// This has to run once the systems are registered, because a set names the
/// animations that drive it, and before the actor types are built, because
/// their data names an appearance. It needs a canvas, because the frames of a set
/// are uploaded to the gpu while the set is read.
pub(super) fn init(canvas: &Canvas, frame: &SystemManager<2>, mods: &ModRegistry) {
    if let Err(e) = init_sprite_sets(canvas, frame, mods) {
        error!("failed to load sprite sets: {e}");
    }
}

/// Builds the sets of `assets/sprites` and the frames each of them names.
fn init_sprite_sets(
    canvas: &Canvas,
    frame: &SystemManager<2>,
    mods: &ModRegistry,
) -> anyhow::Result<()> {
    let raw = Registry::<RawSpriteSet>::load_rons(mods, "assets/sprites")?;
    let mut sets = Registry::new();

    for id in 0..raw.entries.len() as Id {
        let name = raw.name_of(id);
        let entry = &raw.entries[id as usize];
        let (frame_handles, _, view) = load_texture_array(
            canvas,
            mods,
            &format!("assets/textures/actor/{name}"),
            name,
            TextureFormat::Rgba8UnormSrgb,
            true,
            |_| (),
        )?;
        let animations = Animations::resolve(frame, &entry.animations)?;

        let sampler = Sampler::new(
            canvas,
            &SamplerConfig {
                name,
                filter: FilterMode::Nearest,
                address_mode: AddressMode::Repeat,
                mipmap_filter: MipmapFilterMode::Linear,
                mip_level_count: 1,
                compare: None,
            },
        );

        let frames = BindSet::new(
            canvas,
            &BindSetConfig {
                name: &format!("{name}_frames"),
                content: (&view, &sampler),
            },
        );

        sets.register(
            id,
            name,
            SpriteSet {
                frames,
                frame_handles,
                animations,
            },
        );
    }

    SPRITE_SETS.init(sets);
    Ok(())
}

/// Selects the frame that shows an actor from the side the camera is on.
pub(super) struct DirectionalFacing;

impl DirectionalFacing {
    const FRAME_STEP: f32 = std::f32::consts::TAU / 8.0;
}

impl System<2> for DirectionalFacing {
    type CompQueries = (
        (
            CompRead<Animation<Self>>,
            CompWrite<Appearance>,
            CompRead<Position>,
            CompRead<Rotation>,
            Without<PlayerControlled>,
        ),
        (),
    );
    type ResQuery = (ResRead<Camera>, ResRead<PartialTick>);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (camera, partial) = res;

        for (_, animation, appearance, position, rotation, _) in comp.0.iter() {
            let pos = position.interpolated(partial.0);
            let to_camera = Vec2::new(camera.eye.x - pos.x, camera.eye.z - pos.z);

            if to_camera.length_squared() <= f32::EPSILON {
                continue;
            }

            let camera_yaw = to_camera.x.atan2(-to_camera.y);
            let turn = (camera_yaw - rotation.0[0]) / Self::FRAME_STEP;
            let rem = turn.round().rem_euclid(8.0) as u32;

            for &layer in &animation.layers {
                if let Some(sprite) = appearance.0.get_mut(layer).and_then(Option::as_mut) {
                    sprite.frame = (sprite.frame / 8) * 8 + rem;
                }
            }
        }
    }
}

/// Registers the animation component and the frame system that drives it.
pub(super) fn register(entities: &mut EntityManager, frame: &mut SystemManager<2>) {
    entities
        .components
        .register::<Animation<DirectionalFacing>>("directional_facing");
    frame.register(3, "directional_facing", DirectionalFacing);
}

/// The instance of one appearance layer: where it is and which texels it shows.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ActorInst {
    /// Where the actor stands, interpolated between the last two ticks.
    pub pos: Vec3,
    /// Width and height of the sprite, in blocks.
    pub size: Vec2,
    /// The billboard-local offset of the bottom center of the layer.
    pub offset: Vec3,
    /// The top-left corner of the uv rectangle.
    pub uv_min: Vec2,
    /// The bottom-right corner of the uv rectangle.
    pub uv_max: Vec2,
    /// The layer of the set the frame is drawn from.
    pub tex: Id,
    pub color: [f32; 4],
}

impl Inst for ActorInst {
    fn layout<'a, V: Vertex>() -> Option<VertexBufferLayout<'a>> {
        Some(VertexBufferLayout {
            array_stride: size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: offset_of!(Self, pos) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: offset_of!(Self, size) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 1,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, offset) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 2,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: offset_of!(Self, uv_min) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 3,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, uv_max) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 4,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, tex) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 5,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: offset_of!(Self, color) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 6,
                    format: VertexFormat::Float32x4,
                },
            ],
        })
    }
}

resources! {
    /// The frames of every sprite set, and the instances of the actors that are
    /// drawn with it.
    pub struct ActorSprites(Vec<SetSprites>);
}

/// The GPU state of one sprite set.
pub struct SetSprites {
    /// The frames of the set, as the layers of one texture array. The registry
    /// of the set owns the array and outlives every actor that draws with it.
    pub frames: &'static BindSet<TextureArraySampler>,
    /// The actors of the set that are drawn in the current frame.
    pub instances: BufferVec<ActorInst>,
}

impl ActorSprites {
    /// Takes the texture array of every set and creates the buffer its
    /// instances are written to.
    fn new(canvas: &Canvas) -> Self {
        let sets = SPRITE_SETS
            .entries
            .iter()
            .map(|set| SetSprites {
                frames: &set.frames,
                instances: BufferVec::instance(canvas, "actor", BufferInit::Size(0)),
            })
            .collect();

        Self(sets)
    }
}

/// Registers the resource and the system that draw the actors, which need a
/// canvas.
pub(super) fn register_canvas(
    resources: &mut ResourceManager,
    frame: &mut SystemManager<2>,
    canvas: &Canvas,
) {
    resources.register("actor_sprites", ActorSprites::new(canvas));

    frame.register(8, "actor_renderer", ActorRenderer::new(canvas));
}

/// Draws the actors of the world as sprites that turn towards the camera.
pub(super) struct ActorRenderer {
    batch: RenderBatch<
        (TextureArraySampler, CameraTransform, WorldLightsLayout),
        (),
        ActorInst,
        ColorDraw<1>,
        DepthDraw,
    >,
}

/// The instances of one sprite set, as the batch draws them.
struct SetList<'a>(&'a BufferVec<ActorInst>);

impl Render<(), ActorInst> for SetList<'_> {
    fn rendered(&self) -> Vec<RenderItem<'_, (), ActorInst>> {
        RenderItem::new(&EMPTY_BUFFER_VEC, &QUAD_INDEX_BUFFER_VEC, self.0)
            .into_iter()
            .collect()
    }
}

impl ActorRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "actor",
                    shader: "actor",
                    // A sprite is drawn opaque and cut out by its alpha, so that
                    // the actors do not have to be sorted.
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: None,
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
        }
    }
}

impl System<2> for ActorRenderer {
    type CompQueries = (
        (
            CompRead<Appearance>,
            CompRead<Position>,
            // The camera stands in the player, so drawing its sprite would fill
            // the screen.
            Without<PlayerControlled>,
        ),
        (),
    );
    type ResQuery = (
        ResRead<Canvas>,
        ResRead<Camera>,
        ResWrite<ActorSprites>,
        ResRead<PartialTick>,
        ResWrite<FrameSlot>,
        ResRead<Lighting>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, camera, sprites, partial_tick, frame, lights) = res;
        let mut instances = vec![Vec::new(); sprites.0.len()];

        for (_, appearance, position, _) in comp.0.iter() {
            let pos = position.interpolated(partial_tick.0);

            for sprite in appearance.0.iter().flatten() {
                let Some(actors) = instances.get_mut(sprite.set as usize) else {
                    continue;
                };

                actors.push(ActorInst {
                    pos,
                    size: sprite.size,
                    offset: sprite.offset,
                    uv_min: sprite.uvs.min,
                    uv_max: sprite.uvs.max,
                    tex: sprite.frame,
                    color: [1.0; 4],
                });
            }
        }

        for (set, instances) in sprites.0.iter_mut().zip(instances) {
            set.instances.set_content(canvas, &instances);
        }

        let Some(frame) = frame.as_mut() else {
            return;
        };

        let colors = [ColorTarget {
            view: &canvas.color_view,
            load: LoadOp::Load,
        }];
        let depth = DepthTarget {
            view: &canvas.depth_view,
            load: LoadOp::Load,
        };

        frame.render::<ColorDraw<1>, DepthDraw>("actor", colors, depth, |mut pass| {
            self.batch.begin(&mut pass);

            // Every set draws from its own texture array, so it is bound and
            // drawn on its own.
            for set in sprites.0.iter() {
                self.batch.push(
                    &mut pass,
                    (set.frames, &camera.transform, &lights.bind_group),
                );
                self.batch.draw(&mut pass, &SetList(&set.instances));
            }
        });
    }
}
