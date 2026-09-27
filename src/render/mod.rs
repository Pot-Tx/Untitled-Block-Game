//! Thin wrappers around the parts of `wgpu` that the game uses.
//!
//! Every resource is created through [`FromConfig`], which derives the debug
//! labels of the underlying `wgpu` objects from the name in a configuration
//! struct, so that a captured frame can be read back to its source.

mod batch;
mod binding;
mod camera;
mod canvas;
mod mesh;
mod texture;
mod vertex;

use crate::resources;
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use glam::*;
use std::marker::PhantomData;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

use crate::util::OnceInit;
pub use batch::*;
pub use binding::*;
pub use camera::*;
pub use canvas::*;
pub use mesh::*;
pub use texture::*;
pub use vertex::*;

pub const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];
/// A vertex buffer without vertices, used by batches that draw from indices and
/// instances only.
pub static EMPTY_BUFFER_VEC: OnceInit<BufferVec<()>> = OnceInit::new();
/// The index buffer that draws a single quad, shared by every quad batch.
pub static QUAD_INDEX_BUFFER_VEC: OnceInit<BufferVec<u16>> = OnceInit::new();

/// Creates [`EMPTY_BUFFER_VEC`], which needs a [`Canvas`] to exist.
pub fn create_empty_buffer_vec(canvas: &Canvas) -> BufferVec<()> {
    BufferVec::vertex(canvas, "empty", BufferInit::Size(0))
}

/// Creates [`QUAD_INDEX_BUFFER_VEC`], which needs a [`Canvas`] to exist.
pub fn create_quad_index_buffer_vec(canvas: &Canvas) -> BufferVec<u16> {
    BufferVec::index(canvas, "quad", BufferInit::Content(&QUAD_INDICES))
}

/// A drawable geometry, described as vertex, index and instance buffers.
pub trait Render<V: Vertex, I: Inst> {
    fn rendered(&self) -> Vec<RenderItem<'_, V, I>>;
}

/// One draw call: the buffers of a piece of geometry, ready to be bound.
#[derive(Clone)]
pub struct RenderItem<'a, V: Vertex, I: Inst> {
    pub vertices: &'a BufferVec<V>,
    pub indices: &'a BufferVec<u16>,
    pub instances: &'a BufferVec<I>,
    _v_marker: PhantomData<V>,
    _i_marker: PhantomData<I>,
}

/// A vertex, index or instance buffer that grows to fit its content.
///
/// `length` counts the elements in use while `capacity` tracks the size of the
/// allocation, so [`Self::set_content`] only has to reallocate when the content
/// grew.
#[derive(Clone)]
pub struct BufferVec<T: Pod + Zeroable> {
    pub buffer: Buffer,
    pub length: u32,
    capacity: u32,
    label: String,
    usage: BufferUsages,
    _marker: PhantomData<T>,
}

impl<'a, V: Vertex, I: Inst> RenderItem<'a, V, I> {
    /// Builds a draw call from the three buffers of a piece of geometry.
    ///
    /// Fails when the geometry would draw nothing, which keeps the render pass
    /// from setting up an empty draw.
    pub fn new(
        vertices: &'a BufferVec<V>,
        indices: &'a BufferVec<u16>,
        instances: &'a BufferVec<I>,
    ) -> Result<Self> {
        if indices.length == 0 {
            return Err(anyhow!("index buffer should not be empty"));
        }

        if instances.length == 0 {
            return Err(anyhow!("instance buffer should not be empty"));
        }

        Ok(Self {
            vertices,
            indices,
            instances,
            _v_marker: PhantomData,
            _i_marker: PhantomData,
        })
    }
}

impl<T: Pod + Zeroable> FromConfig<BufferConfig<'_, T>> for BufferVec<T> {
    type Base = Canvas;

    #[inline]
    fn new(base: &Self::Base, config: &BufferConfig<T>) -> Self {
        let (length, capacity) = match config.init {
            BufferInit::Content(c) => (c.len() as u32, c.len() as u32),

            BufferInit::Size(size) => {
                let stride = size_of::<T>();
                let capacity = if stride == 0 {
                    0
                } else {
                    (size as usize / stride) as u32
                };
                (0, capacity)
            }
        };

        Self {
            buffer: Buffer::new(base, config),
            length,
            capacity,
            label: format!("{}_buffer", config.name),
            usage: config.usage,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Pod + Zeroable> From<&'a BufferVec<T>> for Option<BufferSlice<'a>> {
    fn from(value: &'a BufferVec<T>) -> Option<BufferSlice<'a>> {
        let bytes = value.length as BufferAddress * size_of::<T>() as BufferAddress;

        if value.length > 0 {
            Some(value.buffer.slice(..bytes))
        } else {
            None
        }
    }
}

impl<T: Pod + Zeroable> BufferVec<T> {
    /// Creates a vertex buffer with the given initial content.
    pub fn vertex(canvas: &Canvas, name: &str, init: BufferInit<T>) -> Self {
        Self::new(
            canvas,
            &BufferConfig {
                name,
                init,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            },
        )
    }

    /// Creates an instance buffer with the given initial content.
    pub fn instance(canvas: &Canvas, name: &str, init: BufferInit<T>) -> Self {
        Self::new(
            canvas,
            &BufferConfig {
                name,
                init,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            },
        )
    }

    /// Creates an index buffer with the given initial content.
    pub fn index(canvas: &Canvas, name: &str, init: BufferInit<T>) -> Self {
        Self::new(
            canvas,
            &BufferConfig {
                name,
                init,
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            },
        )
    }

    /// Replaces the content of the buffer, growing the allocation when needed.
    pub fn set_content(&mut self, canvas: &Canvas, content: &[T]) {
        self.length = content.len() as u32;

        if self.length == 0 {
            return;
        }

        let contents = bytemuck::cast_slice(content);

        if self.length > self.capacity {
            self.buffer = canvas.device.create_buffer_init(&BufferInitDescriptor {
                label: Label::from(self.label.as_str()),
                contents,
                usage: self.usage,
            });
            self.capacity = self.length;
        } else {
            canvas.queue.write_buffer(&self.buffer, 0, contents);
        }
    }
}

resources! {
    /// How far the current frame sits between the last tick and the next one,
    /// in `0..1`; used to interpolate the camera.
    pub struct PartialTick(f32);
}

/// A type that can be built from a configuration and a base object.
pub trait FromConfig<C> {
    type Base;

    fn new(base: &Self::Base, config: &C) -> Self;
}

/// The initial content of a buffer: either the elements themselves or the size
/// of an empty allocation to be filled later.
pub enum BufferInit<'a, T: Pod + Zeroable> {
    Content(&'a [T]),
    Size(BufferAddress),
}

/// The name, initial content and usage of a buffer to create.
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
