//! The screen space user interface: the screens that are open, the elements on
//! them and the systems that lay them out and draw them.
//!
//! UI entities live in their own [`EntityManager`](crate::ecs::EntityManager),
//! separately from the entities of the world.

mod render;
mod screen;
mod trigger;

use render::{UiRenderer, init_ui_textures};
pub use screen::*;
pub use trigger::*;

use crate::components;
use crate::ecs::{EntityManager, ResourceManager, SystemManager};
use crate::game::ModRegistry;
use crate::render::{Canvas, Inst, Vertex, Viewports};
use crate::util::bounding::AABB;
use crate::world::Block;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec4};
use log::error;
use serde::{Deserialize, Serialize};
use std::mem::offset_of;
use std::time::Duration;
use wgpu::*;

/// Width and height of the UI space that the viewports are laid out in, in UI
/// units.
const UI_WIDTH: f32 = 256.0;
const UI_HEIGHT: f32 = 256.0;

components! {
    /// The area an element occupies, in the UI space of its viewport.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct UiRect(AABB<Vec2>): Hot, asset;
    /// A quad drawn from one layer of the UI texture array.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct UiSprite { pub tex: u32, pub uv: AABB<Vec2>, pub color: Vec4 }: Hot, asset;
    /// A piece of text drawn with the glyph atlas.
    #[derive(Clone, Serialize, Deserialize)]
    pub struct UiText {
        pub text: String,
        pub h_align: UiHAlign,
        pub v_align: UiVAlign,
        pub color: Vec4,
        pub scale: f32,
    }: Cold, asset;
    /// A block drawn inside a UI element, filling the element's rectangle.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct UiBlock(Block): Cold, asset;
}

/// A block as it is written to a data file: the name of its type and its state,
/// so that the block types can be reordered without breaking the files.
///
/// The state is left out for a block in the default state of its type; RON
/// writes `Option` as `Some` or `None`, so any other state is written as
/// `state: Some(n)`.

/// The horizontal alignment of a piece of UI text.
///
/// It mirrors the alignment of the text renderer, so that a text element can be
/// written in data without depending on the renderer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum UiHAlign {
    Left,
    Center,
    Right,
}

/// The vertical alignment of a piece of UI text.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum UiVAlign {
    Top,
    Center,
    Bottom,
}

/// Registers the components, the resources and the systems of the user
/// interface.
pub(crate) fn register(
    entities: &mut EntityManager,
    systems: &mut SystemManager<2>,
    resources: &mut ResourceManager,
) {
    entities.components.register::<UiRect>("rect");
    entities.components.register::<UiSprite>("sprite");
    entities.components.register::<UiText>("text");
    entities.components.register::<Triggers>("trigger_tag");
    entities.components.register::<UiBlock>("block");

    resources.register("active_element", ActiveElement(None));
    resources.register("active_screens", ActiveScreens::default());

    systems.register(0, "cursor_tracker", CursorTracker);
    systems.register(1, "screen_controller", ScreenController);
    systems.register(2, "ui_pointer", UiPointer);
    systems.register(2, "hotbar_controller", HotbarController);
    systems.register(3, "highlighter", Highlighter::new(1.5, 0.5));
    systems.register(3, "frame_rater", FrameRater::new(Duration::from_secs(1)));
}

/// Builds the screens and registers the resources and the system of the
/// interface that need a canvas.
///
/// A screen names the block of an element and the systems that handle the
/// triggers of its elements, so the screens are built once the blocks and the
/// systems exist.
pub(crate) fn register_canvas(
    entities: &mut EntityManager,
    systems: &mut SystemManager<2>,
    resources: &mut ResourceManager,
    canvas: &Canvas,
    mods: &ModRegistry,
) {
    if let Err(e) = init_screen_types(&entities.components, systems, mods) {
        error!("failed to load screen types: {e}");
    }

    let mut viewports = Viewports::new(canvas, UI_WIDTH, UI_HEIGHT);
    viewports.transform(canvas);
    resources.register("viewports", viewports);

    // The interface is the last pass of a frame: it clears the depth buffer to
    // sort its own elements with it, so a pass drawn after it would be tested
    // against that depth instead of the depth of the world.
    if let Err(e) = init_ui_textures(canvas, mods) {
        error!("failed to load ui textures: {e}");
    }
    systems.register(12, "ui_renderer", UiRenderer::new(canvas, mods));
}

/// Spawns the screens the game starts with.
pub(crate) fn open_initial(entities: &mut EntityManager, resources: &ResourceManager) {
    let screens = resources
        .get::<ActiveScreens>()
        .expect("the active screens are registered")
        .get_mut::<ActiveScreens>();
    let mut ui = entities.commander().lock();

    screens.open(SCREEN_TYPES.id_of("hud"), |element| ui.spawn(element));
    screens.open(SCREEN_TYPES.id_of("hotbar"), |element| ui.spawn(element));
}

/// The instance data of one UI quad or glyph.
///
/// The coordinates are in the UI space of the viewport, which the transform of
/// that viewport maps onto the window.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct UiInst {
    /// Layer of the UI texture array to sample.
    pub tex: u32,
    pub min: Vec2,
    pub max: Vec2,
    /// Depth of the element inside its viewport, where a larger value is drawn
    /// on top.
    pub z: f32,
    pub min_uv: Vec2,
    pub max_uv: Vec2,
    pub color: [f32; 4],
}

impl Inst for UiInst {
    fn layout<'a, V: Vertex>() -> Option<VertexBufferLayout<'a>> {
        Some(VertexBufferLayout {
            array_stride: size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: offset_of!(Self, tex) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: offset_of!(Self, min) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 1,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, max) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 2,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, z) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 3,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: offset_of!(Self, min_uv) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 4,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, max_uv) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 5,
                    format: VertexFormat::Float32x2,
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

/// The instance data of one UI block: the rectangle of its element and the
/// range of depths the block itself is drawn in.
///
/// The camera of the block maps the depth of a vertex between `min_z` and
/// `max_z`, so that the block keeps the order of its own faces while it stays
/// inside the layer of its element.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct UiBlockInst {
    pub min: Vec2,
    pub max: Vec2,
    pub min_z: f32,
    pub max_z: f32,
}

impl Inst for UiBlockInst {
    fn layout<'a, V: Vertex>() -> Option<VertexBufferLayout<'a>> {
        Some(VertexBufferLayout {
            array_stride: size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: offset_of!(Self, min) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, max) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 1,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, min_z) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 2,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: offset_of!(Self, max_z) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 3,
                    format: VertexFormat::Float32,
                },
            ],
        })
    }
}
