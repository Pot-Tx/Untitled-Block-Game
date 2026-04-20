use crate::render::canvas::Canvas;
use crate::render::vertex::*;
use crate::render::{BufferConfig, BufferInit, FromConfig};
use crate::util::coord::*;
use crate::util::Id;
use glam::*;
use std::array;
use std::marker::PhantomData;
use wgpu::*;

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

pub trait Render<V: Vertex, I: Inst> {
    fn rendered(&self) -> Vec<RenderItem<'_, V, I>>;
}

#[derive(Clone)]
pub struct Geometry<V: Vertex> {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    _marker: PhantomData<V>,
}

#[derive(Clone)]
pub struct Instances<I: Inst> {
    pub instance_buffer: Buffer,
    pub instance_count: u32,
    _marker: PhantomData<I>,
}

#[derive(Clone)]
pub struct RenderItem<'a, V: Vertex, I: Inst> {
    pub geometry: &'a Geometry<V>,
    pub instances: &'a Instances<I>,
}

#[derive(Clone)]
pub struct Mesh<V: Vertex> {
    pub vertices: Vec<V>,
    pub indices: Vec<u16>,
}

impl<V: Vertex> Default for Mesh<V> {
    fn default() -> Self {
        Self {
            vertices: Vec::default(),
            indices: Vec::default(),
        }
    }
}

impl<V: Vertex> Mesh<V> {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    #[inline]
    pub fn translate(&mut self, dpos: V::Pos) -> &mut Self {
        self.vertices.iter_mut().for_each(|v| {
            *v = v.translate(dpos);
        });
        self
    }

    #[inline]
    pub fn translated(&self, dpos: V::Pos) -> Self {
        let vertices = self.vertices.iter().map(|v| v.translate(dpos)).collect();

        Self {
            vertices,
            indices: self.indices.clone(),
        }
    }

    #[inline]
    pub fn scale(&mut self, scale: V::Pos) -> &mut Self {
        self.vertices.iter_mut().for_each(|v| {
            *v = v.scale(scale);
        });
        self
    }

    #[inline]
    pub fn scaled(&self, scale: V::Pos) -> Self {
        let vertices = self.vertices.iter().map(|v| v.scale(scale)).collect();

        Self {
            vertices,
            indices: self.indices.clone(),
        }
    }

    #[inline]
    pub fn multiply(&mut self, scale: <V::Pos as Coord>::Scalar) -> &mut Self {
        self.vertices.iter_mut().for_each(|v| {
            *v = v.multiply(scale);
        });
        self
    }

    #[inline]
    pub fn multiplied(&self, scale: <V::Pos as Coord>::Scalar) -> Self {
        let vertices = self.vertices.iter().map(|v| v.multiply(scale)).collect();

        Self {
            vertices,
            indices: self.indices.clone(),
        }
    }

    #[inline]
    pub fn merge(&mut self, other: &Self) -> &mut Self {
        let offset = self.vertices.len() as u16;

        self.vertices.extend(&other.vertices);
        self.indices.extend(
            other
                .indices
                .iter()
                .map(|&i| offset + i)
                .collect::<Vec<_>>(),
        );

        self
    }

    #[inline]
    pub fn merged(&self, other: &Self) -> Self {
        let mut joined = self.clone();
        joined.merge(&other);
        joined
    }

    pub fn geometry<'a>(&self, canvas: &Canvas, name: &'a str) -> Geometry<V> {
        let vertex_buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: &format!("{}_vertex", name),
                init: BufferInit::Content(&self.vertices),
                usage: BufferUsages::VERTEX,
            },
        );
        let index_buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: &format!("{}_index", name),
                init: BufferInit::Content(&self.indices),
                usage: BufferUsages::INDEX,
            },
        );

        Geometry {
            vertex_buffer,
            index_buffer,
            index_count: self.indices.len() as u32,
            _marker: PhantomData,
        }
    }
}

pub trait MeshGroup {
    type Vertex: Vertex;

    fn merge(&self) -> Mesh<Self::Vertex>;
}

impl<V: Vertex> MeshGroup for [Mesh<V>] {
    type Vertex = V;

    #[inline]
    fn merge(&self) -> Mesh<Self::Vertex> {
        let mut joined = Mesh::new();

        for mesh in self {
            joined.merge(mesh);
        }

        joined
    }
}

impl Mesh<BasicVertex> {
    pub fn cuboid(min: Vec3, max: Vec3) -> [Self; 6] {
        let p = Vec3::cuboid(min, max).map(|pos| BasicVertex { pos });

        [
            Self {
                vertices: vec![p[2], p[0], p[1], p[3]],
                indices: Vec::from(QUAD_INDICES),
            },
            Self {
                vertices: vec![p[7], p[5], p[4], p[6]],
                indices: Vec::from(QUAD_INDICES),
            },
            Self {
                vertices: vec![p[5], p[1], p[0], p[4]],
                indices: Vec::from(QUAD_INDICES),
            },
            Self {
                vertices: vec![p[2], p[3], p[7], p[6]],
                indices: Vec::from(QUAD_INDICES),
            },
            Self {
                vertices: vec![p[6], p[4], p[0], p[2]],
                indices: Vec::from(QUAD_INDICES),
            },
            Self {
                vertices: vec![p[3], p[1], p[5], p[7]],
                indices: Vec::from(QUAD_INDICES),
            },
        ]
    }
    
    pub fn frame(min: Vec3, max: Vec3) -> Self {
        let p = Vec3::cuboid(min, max).map(|pos| BasicVertex { pos });
        
        Self {
            vertices: Vec::from(p),
            indices: vec![
                0, 1,
                0, 2,
                0, 4,
                1, 3,
                1, 5,
                2, 3,
                2, 6,
                3, 7,
                4, 5,
                4, 6,
                5, 7,
                6, 7,
            ],
        }
    }

    pub fn with_texture(&self, tex: Id, uvs: Vec<Vec2>) -> Mesh<TexVertex> {
        Mesh {
            vertices: self
                .vertices
                .iter()
                .enumerate()
                .map(|(i, v)| v.with_texture(tex, uvs[i]))
                .collect(),
            indices: self.indices.clone(),
        }
    }
}

impl Mesh<TexVertex> {
    pub fn cuboid(min: Vec3, max: Vec3, texs: [Id; 6], uvs: [Vec<Vec2>; 6]) -> [Self; 6] {
        let cuboid = Mesh::<BasicVertex>::cuboid(min, max);
        array::from_fn(|i| cuboid[i].with_texture(texs[i], uvs[i].clone()))
    }

    pub fn with_texture(&self, tex: Id) -> Self {
        Mesh {
            vertices: self.vertices.iter().map(|v| v.with_texture(tex)).collect(),
            indices: self.indices.clone(),
        }
    }

    pub fn with_normal(&self, norm: Vec3) -> Mesh<NormTexVertex> {
        Mesh {
            vertices: self.vertices.iter().map(|m| m.with_normal(norm)).collect(),
            indices: self.indices.clone(),
        }
    }
}

impl Mesh<NormTexVertex> {
    pub fn cuboid(min: Vec3, max: Vec3, texs: [Id; 6], uvs: [Vec<Vec2>; 6]) -> [Self; 6] {
        let cuboid = Mesh::<TexVertex>::cuboid(min, max, texs, uvs);
        array::from_fn(|i| cuboid[i].with_normal(Direction::by_idx(i).vector()))
    }

    pub fn with_texture(&self, tex: Id) -> Self {
        Mesh {
            vertices: self.vertices.iter().map(|v| v.with_texture(tex)).collect(),
            indices: self.indices.clone(),
        }
    }
}

pub trait InstGroup {
    type Inst: Inst;

    fn instances<'a>(&self, canvas: &Canvas, name: &'a str) -> Instances<Self::Inst>;
}

impl<I: Inst> InstGroup for [I] {
    type Inst = I;

    fn instances<'a>(&self, canvas: &Canvas, name: &'a str) -> Instances<Self::Inst> {
        let instance_buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: &format!("{}_instance", name),
                init: BufferInit::Content(self),
                usage: BufferUsages::VERTEX,
            },
        );

        Instances {
            instance_buffer,
            instance_count: self.len() as u32,
            _marker: PhantomData,
        }
    }
}
