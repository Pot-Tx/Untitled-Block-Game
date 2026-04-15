use crate::util::coord::*;
use bytemuck::{Pod, Zeroable};
use glam::*;
use std::fmt::Debug;
use wgpu::*;

pub trait Vertex: Copy + Clone + Sync + Send + Pod + Zeroable + Debug {
	type Pos: Coord;
	const ATTRIBUTE_COUNT: u32;
	const LAYOUT: VertexBufferLayout<'static>;
	
	#[must_use]
	fn translate(self, dpos: Self::Pos) -> Self;
	
	#[must_use]
	fn scale(self, scale: Self::Pos) -> Self;
	
	#[must_use]
	fn multiply(self, scale: <Self::Pos as Coord>::Scalar) -> Self;
}

pub trait Inst: Copy + Clone + Sync + Send + Pod + Zeroable + Debug {
	fn layout<'a, V: Vertex>() -> VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BasicVertex {
	pub pos: Vec3,
}
unsafe impl Pod for BasicVertex {}
unsafe impl Zeroable for BasicVertex {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TexVertex {
	pub pos: Vec3,
	pub tex: u32,
	pub uv: Vec2,
}
unsafe impl Pod for TexVertex {}
unsafe impl Zeroable for TexVertex {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NormTexVertex {
	pub pos: Vec3,
	pub tex: u32,
	pub uv: Vec2,
	pub norm: Vec3,
}
unsafe impl Pod for NormTexVertex {}
unsafe impl Zeroable for NormTexVertex {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AlphaVertex {
	pub pos: Vec3,
	pub alpha: f32,
}
unsafe impl Pod for AlphaVertex {}
unsafe impl Zeroable for AlphaVertex {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TransInst {
	pub pos: Vec3,
}
unsafe impl Pod for TransInst {}
unsafe impl Zeroable for TransInst {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct IntTransInst {
	pub pos: IVec3,
}
unsafe impl Pod for IntTransInst {}
unsafe impl Zeroable for IntTransInst {}


macro_rules! vertex_basics {
    ($pos:ty, $count:expr) => {
        type Pos = $pos;
        const ATTRIBUTE_COUNT: u32 = $count;

        #[inline]
        fn translate(mut self, dpos: Self::Pos) -> Self {
            self.pos += dpos;
            self
        }
        
        #[inline]
        fn scale(mut self, scale: Self::Pos) -> Self {
            self.pos *= scale;
            self
        }
    };
}

impl Vertex for () {
	type Pos = Vec3;
	const ATTRIBUTE_COUNT: u32 = 0;
	const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
		array_stride: 0 as BufferAddress,
		step_mode: VertexStepMode::Vertex,
		attributes: &[],
	};
	
	fn translate(self, _: Self::Pos) -> Self {
		self
	}
	
	fn scale(self, _: Self::Pos) -> Self {
		self
	}
	
	fn multiply(self, _: <Self::Pos as Coord>::Scalar) -> Self {
		self
	}
}

impl Vertex for BasicVertex {
	vertex_basics!(Vec3, 1);
	
	const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
		array_stride: size_of::<Self>() as BufferAddress,
		step_mode: VertexStepMode::Vertex,
		attributes: &[VertexAttribute {
			offset: 0,
			shader_location: 0,
			format: VertexFormat::Float32x3,
		}],
	};
	
	#[inline]
	fn multiply(mut self, scale: <Self::Pos as Coord>::Scalar) -> Self {
		self.pos *= scale;
		self
	}
}

impl BasicVertex {
	pub fn with_texture(&self, tex: u32, uv: Vec2) -> TexVertex {
		TexVertex {
			pos: self.pos,
			tex,
			uv,
		}
	}
}

impl Vertex for TexVertex {
	vertex_basics!(Vec3, 3);
	
	const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
		array_stride: size_of::<Self>() as BufferAddress,
		step_mode: VertexStepMode::Vertex,
		attributes: &[
			VertexAttribute {
				offset: 0,
				shader_location: 0,
				format: VertexFormat::Float32x3,
			},
			VertexAttribute {
				offset: size_of::<Vec3>() as BufferAddress,
				shader_location: 1,
				format: VertexFormat::Uint32,
			},
			VertexAttribute {
				offset: (size_of::<Vec3>() + size_of::<u32>()) as BufferAddress,
				shader_location: 2,
				format: VertexFormat::Float32x2,
			},
		],
	};
	
	#[inline]
	fn multiply(mut self, scale: <Self::Pos as Coord>::Scalar) -> Self {
		self.pos *= scale;
		self.uv *= scale;
		self
	}
}

impl TexVertex {
	pub fn with_texture(&self, tex: u32) -> Self {
		Self {
			pos: self.pos,
			tex,
			uv: self.uv,
		}
	}
	
	pub fn with_normal(&self, norm: Vec3) -> NormTexVertex {
		NormTexVertex {
			pos: self.pos,
			tex: self.tex,
			uv: self.uv,
			norm,
		}
	}
}

impl Vertex for NormTexVertex {
	vertex_basics!(Vec3, 4);
	
	const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
		array_stride: size_of::<Self>() as BufferAddress,
		step_mode: VertexStepMode::Vertex,
		attributes: &[
			VertexAttribute {
				offset: 0,
				shader_location: 0,
				format: VertexFormat::Float32x3,
			},
			VertexAttribute {
				offset: size_of::<Vec3>() as BufferAddress,
				shader_location: 1,
				format: VertexFormat::Uint32,
			},
			VertexAttribute {
				offset: (size_of::<Vec3>() + size_of::<u32>()) as BufferAddress,
				shader_location: 2,
				format: VertexFormat::Float32x2,
			},
			VertexAttribute {
				offset: (size_of::<Vec3>() + size_of::<u32>() + size_of::<Vec2>())
					as BufferAddress,
				shader_location: 3,
				format: VertexFormat::Float32x3,
			},
		],
	};
	
	#[inline]
	fn multiply(mut self, scale: <Self::Pos as Coord>::Scalar) -> Self {
		self.pos *= scale;
		self.uv *= scale;
		self
	}
}

impl NormTexVertex {
	pub fn with_texture(&self, tex: u32) -> Self {
		Self {
			pos: self.pos,
			tex,
			uv: self.uv,
			norm: self.norm,
		}
	}
}

impl Vertex for AlphaVertex {
	vertex_basics!(Vec3, 2);
	
	const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
		array_stride: size_of::<Self>() as BufferAddress,
		step_mode: VertexStepMode::Vertex,
		attributes: &[
			VertexAttribute {
				offset: 0,
				shader_location: 0,
				format: VertexFormat::Float32x3,
			},
			VertexAttribute {
				offset: size_of::<Vec3>() as BufferAddress,
				shader_location: 1,
				format: VertexFormat::Float32,
			},
		],
	};
	
	#[inline]
	fn multiply(mut self, scale: <Self::Pos as Coord>::Scalar) -> Self {
		self.pos *= scale;
		self
	}
}

impl Inst for () {
	fn layout<'a, V: Vertex>() -> VertexBufferLayout<'a> {
		VertexBufferLayout {
			array_stride: 0 as BufferAddress,
			step_mode: VertexStepMode::Instance,
			attributes: &[],
		}
	}
}

impl Inst for TransInst {
	fn layout<'a, V: Vertex>() -> VertexBufferLayout<'a> {
		VertexBufferLayout {
			array_stride: size_of::<Self>() as BufferAddress,
			step_mode: VertexStepMode::Instance,
			attributes: &[VertexAttribute {
				offset: 0,
				shader_location: V::ATTRIBUTE_COUNT,
				format: VertexFormat::Float32x3,
			}],
		}
	}
}

impl Inst for IntTransInst {
	fn layout<'a, V: Vertex>() -> VertexBufferLayout<'a> {
		VertexBufferLayout {
			array_stride: size_of::<Self>() as BufferAddress,
			step_mode: VertexStepMode::Instance,
			attributes: &[VertexAttribute {
				offset: 0,
				shader_location: V::ATTRIBUTE_COUNT,
				format: VertexFormat::Sint32x3,
			}],
		}
	}
}
