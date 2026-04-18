mod batch;
mod binding;
mod camera;
mod canvas;
mod mesh;
mod texture;
mod vertex;

use crate::ecs::*;
use crate::resources;
use crate::util::OnceInit;
use bytemuck::{Pod, Zeroable};
use glam::*;
use std::sync::RwLock;
use wgpu::util::DeviceExt;
use wgpu::*;

pub use batch::*;
pub use binding::*;
pub use camera::*;
pub use canvas::*;
pub use mesh::*;
pub use texture::*;
pub use vertex::*;

pub static CANVAS: OnceInit<RwLock<Canvas>> = OnceInit::new();

resources! {
    pub struct PartialTick(f32);
}

pub trait FromConfig<C> {
    type Base;

    fn new(base: &Self::Base, config: &C) -> Self;
}

pub enum BufferInit<'a, T: Pod + Zeroable> {
    Content(&'a [T]),
    Size(BufferAddress),
}

pub struct BufferConfig<'a, T: Pod + Zeroable> {
    pub name: &'a str,
    pub init: BufferInit<'a, T>,
    pub usage: BufferUsages,
}

impl<T: Pod + Zeroable> FromConfig<BufferConfig<'_, T>> for Buffer {
    type Base = Canvas;

    #[inline]
    fn new(base: &Self::Base, config: &BufferConfig<T>) -> Self {
        match config.init {
            BufferInit::Content(content) => {
                base.device.create_buffer_init(&util::BufferInitDescriptor {
                    label: Label::from(format!("{}_buffer", config.name).as_str()),
                    contents: bytemuck::cast_slice(content),
                    usage: config.usage,
                })
            }

            BufferInit::Size(size) => base.device.create_buffer(&BufferDescriptor {
                label: Label::from(format!("{}_buffer", config.name).as_str()),
                size,
                usage: config.usage,
                mapped_at_creation: false,
            }),
        }
    }
}
