//! The screen space user interface: the screens that are open, the elements on
//! them and the systems that lay them out and draw them.
//!
//! UI entities live in their own [`EntityManager`](crate::ecs::EntityManager),
//! separately from the entities of the world.

mod trigger;
mod render;
mod screen;

pub use trigger::*;
use render::UiRenderer;
pub use screen::*;

use crate::components;
use crate::ecs::{ComponentManager, EntityManager, ResourceManager, SystemManager};
use crate::id_of;
use crate::render::{Canvas, Inst, Vertex, Viewports};
use crate::util::bounding::AABB;
use crate::util::Id;
use crate::world::Block;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use serde::{Serialize, Deserialize};
use std::mem::offset_of;
use wgpu::*;

/// Width and height of the UI space that the viewports are laid out in, in UI
/// units.
const UI_WIDTH: f32 = 256.0;
const UI_HEIGHT: f32 = 256.0;

components! {
    /// The area an element occupies, in the UI space of its viewport.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct UiRect(AABB<Vec2>): Hot, data;
    /// A quad drawn from one layer of the UI texture array.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct UiSprite { pub tex: u32, pub uv: AABB<Vec2>, pub color: [f32; 4] }: Hot, data;
    /// A piece of text drawn with the glyph atlas.
    #[derive(Clone, Serialize, Deserialize)]
    pub struct UiText {
        pub text: String,
        pub h_align: UiHAlign,
        pub v_align: UiVAlign,
        pub color: [f32; 4],
        pub scale: f32,
    }: Cold, data;
    /// Marks an element as part of a screen, and orders it inside that screen.
    #[derive(Clone, Copy)]
    pub struct ScreenTag { pub screen: Id, pub priority: u32 }: Hot;

    /// A block drawn inside a UI element, filling the element's rectangle.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct UiBlock(Block): Cold, data;
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

/// Registers the components of a screen element.
fn register_components(components: &mut ComponentManager) {
    components.register::<UiRect>("rect");
    components.register::<UiSprite>("sprite");
    components.register::<UiText>("text");
    components.register::<ScreenTag>("screen_tag");
    components.register::<Triggers>("trigger_tag");
    components.register::<UiBlock>("block");
}

/// Registers the components, the resources, the systems and the data of the
/// user interface.
pub(crate) fn register(
    entities: &mut EntityManager,
    systems: &mut SystemManager,
    resources: &mut ResourceManager,
) {
    register_components(&mut entities.components);

    resources.register("active_element", ActiveElement(None));
    resources.register("active_screens", ActiveScreens::default());

    systems.register(0, "cursor_tracker", CursorTracker);
    systems.register(1, "screen_controller", ScreenController);
    systems.register(1, "screen_collector", ScreenCollector);
    systems.register(2, "ui_pointer", UiPointer);
    systems.register(3, "highlighter", Highlighter::new(1.5, 0.5));

    // A screen names the systems that handle the triggers of its elements, so
    // the screens are built once the systems have been registered.
    SCREEN_TYPES.init(build_screens(&entities.components, systems));
}

/// Registers the resources and the system of the interface that need a canvas.
pub(crate) fn register_canvas(
    systems: &mut SystemManager,
    resources: &mut ResourceManager,
    canvas: &Canvas,
) {
    let mut viewports = Viewports::new(canvas, UI_WIDTH, UI_HEIGHT);
    viewports.transform(canvas);
    resources.register("viewports", viewports);

    systems.register(4, "ui_renderer", UiRenderer::new(canvas));
}

/// Spawns the screens the game starts with.
pub(crate) fn open_initial(entities: &mut EntityManager, resources: &ResourceManager) {
    let screens = resources
        .get::<ActiveScreens>()
        .expect("the active screens are registered")
        .get_mut::<ActiveScreens>();
    let mut commands = Vec::new();

    commands.extend(screens.open(id_of!(SCREEN_TYPES, "hud")));
    commands.extend(screens.open(id_of!(SCREEN_TYPES, "hotbar")));

    entities.submit(commands);
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
