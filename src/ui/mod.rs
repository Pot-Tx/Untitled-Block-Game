//! The screen space user interface: the screens that are open, the elements on
//! them and the systems that lay them out and draw them.
//!
//! UI entities live in their own [`EntityManager`](crate::ecs::EntityManager),
//! separately from the entities of the world.

mod interaction;
mod render;
mod screen;

pub use interaction::*;
pub use render::*;
pub use screen::*;

use crate::components;
use crate::render::{Inst, Vertex};
use crate::util::bounding::AABB;
use crate::util::Id;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use glyph_brush::{HorizontalAlign, VerticalAlign};
use wgpu::*;

components! {
    /// The area an element occupies, in the UI space of its viewport.
    #[derive(Clone, Copy)]
    pub struct UiRect { pub rect: AABB<Vec2> }: Hot;
    /// A quad drawn from one layer of the UI texture array.
    #[derive(Clone, Copy)]
    pub struct UiSprite { pub tex: u32, pub uv: AABB<Vec2>, pub color: [f32; 4] }: Hot;
    /// A piece of text drawn with the glyph atlas.
    #[derive(Clone)]
    pub struct UiText {
        pub text: String,
        pub h_align: HorizontalAlign,
        pub v_align: VerticalAlign,
        pub color: [f32; 4],
        pub scale: f32,
    }: Cold;
    /// Marks an element as part of a screen, and orders it inside that screen.
    #[derive(Clone, Copy)]
    pub struct ScreenTag { pub screen: Id, pub priority: u32 }: Hot;
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
                    offset: 0,
                    shader_location: V::ATTRIBUTE_COUNT,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: size_of::<u32>() as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 1,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: (size_of::<u32>() + size_of::<Vec2>()) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 2,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: (size_of::<u32>() + size_of::<Vec2>() * 2) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 3,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: (size_of::<u32>() + size_of::<Vec2>() * 2 + size_of::<f32>())
                        as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 4,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: (size_of::<u32>() + size_of::<Vec2>() * 3 + size_of::<f32>())
                        as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 5,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: (size_of::<u32>() + size_of::<Vec2>() * 4 + size_of::<f32>())
                        as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 6,
                    format: VertexFormat::Float32x4,
                },
            ],
        })
    }
}
