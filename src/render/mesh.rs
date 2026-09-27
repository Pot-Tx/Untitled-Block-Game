use crate::render::canvas::Canvas;
use crate::render::vertex::*;
use crate::render::{BufferInit, BufferVec, QUAD_INDICES};
use crate::util::coord::*;
use crate::util::Id;
use glam::*;
use serde::{Deserialize, Serialize};
use std::array;

/// An indexed triangle mesh of `V` vertices.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
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

    /// Returns whether the mesh has no triangles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Moves every vertex by `dpos`.
    #[inline]
    pub fn translate(&mut self, dpos: V::Pos) -> &mut Self {
        self.vertices.iter_mut().for_each(|v| {
            *v = v.translate(dpos);
        });
        self
    }

    /// The mesh moved by `dpos`.
    #[inline]
    pub fn translated(&self, dpos: V::Pos) -> Self {
        let vertices = self.vertices.iter().map(|v| v.translate(dpos)).collect();

        Self {
            vertices,
            indices: self.indices.clone(),
        }
    }

    /// Scales every vertex position by `scale`.
    #[inline]
    pub fn scale(&mut self, scale: V::Pos) -> &mut Self {
        self.vertices.iter_mut().for_each(|v| {
            *v = v.scale(scale);
        });
        self
    }

    /// The mesh scaled by `scale`.
    #[inline]
    pub fn scaled(&self, scale: V::Pos) -> Self {
        let vertices = self.vertices.iter().map(|v| v.scale(scale)).collect();

        Self {
            vertices,
            indices: self.indices.clone(),
        }
    }

    /// Scales every vertex position by the single factor `scale`.
    #[inline]
    pub fn multiply(&mut self, scale: <V::Pos as Coord>::Scalar) -> &mut Self {
        self.vertices.iter_mut().for_each(|v| {
            *v = v.multiply(scale);
        });
        self
    }

    /// The mesh scaled by the single factor `scale`.
    #[inline]
    pub fn multiplied(&self, scale: <V::Pos as Coord>::Scalar) -> Self {
        let vertices = self.vertices.iter().map(|v| v.multiply(scale)).collect();

        Self {
            vertices,
            indices: self.indices.clone(),
        }
    }

    /// Appends `other`, offsetting its indices by the current vertex count.
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

    /// The mesh with `other` appended to it.
    #[inline]
    pub fn merged(&self, other: &Self) -> Self {
        let mut joined = self.clone();
        joined.merge(other);
        joined
    }

    /// Uploads the vertices into a vertex buffer.
    pub fn vertex_buffer(&self, canvas: &Canvas, name: &str) -> BufferVec<V> {
        BufferVec::vertex(
            canvas,
            &format!("{}_vertex", name),
            BufferInit::Content(&self.vertices),
        )
    }

    /// Uploads the indices into an index buffer.
    pub fn index_buffer_vec(&self, canvas: &Canvas, name: &str) -> BufferVec<u16> {
        BufferVec::index(
            canvas,
            &format!("{}_index", name),
            BufferInit::Content(&self.indices),
        )
    }
}

/// A collection of meshes that can be joined into a single one.
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
    /// The six faces of the box from `min` to `max`, in the order of
    /// [`Direction::ALL`].
    ///
    /// Every face is a separate mesh with its own four vertices and the quad
    /// indices of [`QUAD_INDICES`], which is what the other cuboid constructors
    /// build on.
    pub fn cuboid(min: Vec3, max: Vec3) -> [Self; 6] {
        let p = Vec3::corners(min, max).map(|pos| BasicVertex { pos });

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

    /// The twelve edges of the box from `min` to `max`, as a line list.
    pub fn frame(min: Vec3, max: Vec3) -> Self {
        let p = Vec3::corners(min, max).map(|pos| BasicVertex { pos });

        Self {
            vertices: Vec::from(p),
            indices: vec![
                0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 3, 7, 4, 5, 4, 6, 5, 7, 6, 7,
            ],
        }
    }

    /// The same faces with the given normal on every vertex.
    pub fn with_normal(&self, norm: Vec3) -> Mesh<NormVertex> {
        Mesh {
            vertices: self.vertices.iter().map(|v| v.with_normal(norm)).collect(),
            indices: self.indices.clone(),
        }
    }

    /// The same faces with the given texture and per-vertex uvs.
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

impl Mesh<NormVertex> {
    /// The six faces of the box, in the order of [`Direction::ALL`], with the
    /// outward normal of each face.
    pub fn cuboid(min: Vec3, max: Vec3) -> [Self; 6] {
        let cuboid = Mesh::<BasicVertex>::cuboid(min, max);
        array::from_fn(|i| cuboid[i].with_normal(Direction::by_idx(i).vector()))
    }

    /// The same mesh with the given per-vertex uvs.
    pub fn with_uv(&self, uvs: Vec<Vec2>) -> Mesh<NormUvVertex> {
        Mesh {
            vertices: self
                .vertices
                .iter()
                .zip(uvs)
                .map(|(m, uv)| m.with_uv(uv))
                .collect(),
            indices: self.indices.clone(),
        }
    }
}

impl Mesh<TexVertex> {
    /// The six faces of the box, in the order of [`Direction::ALL`], each with
    /// its own texture and uvs.
    pub fn cuboid(min: Vec3, max: Vec3, texs: [Id; 6], uvs: [Vec<Vec2>; 6]) -> [Self; 6] {
        let cuboid = Mesh::<BasicVertex>::cuboid(min, max);
        array::from_fn(|i| cuboid[i].with_texture(texs[i], uvs[i].clone()))
    }

    /// The same mesh with the given normal on every vertex.
    pub fn with_normal(&self, norm: Vec3) -> Mesh<NormTexVertex> {
        Mesh {
            vertices: self.vertices.iter().map(|m| m.with_normal(norm)).collect(),
            indices: self.indices.clone(),
        }
    }
}

impl Mesh<NormUvVertex> {
    /// The six faces of the box, in the order of [`Direction::ALL`], with the
    /// outward normal of each face.
    pub fn cuboid(min: Vec3, max: Vec3, uvs: [Vec<Vec2>; 6]) -> [Self; 6] {
        let cuboid = Mesh::<NormVertex>::cuboid(min, max);
        array::from_fn(|i| cuboid[i].with_uv(uvs[i].clone()))
    }

    /// The same mesh with the given texture on every vertex.
    pub fn with_texture(&self, tex: Id) -> Mesh<NormTexVertex> {
        Mesh {
            vertices: self.vertices.iter().map(|v| v.with_texture(tex)).collect(),
            indices: self.indices.clone(),
        }
    }
}

impl Mesh<NormTexVertex> {
    /// The six faces of the box, in the order of [`Direction::ALL`], each with
    /// its own texture, uvs and outward normal.
    pub fn cuboid(min: Vec3, max: Vec3, texs: [Id; 6], uvs: [Vec<Vec2>; 6]) -> [Self; 6] {
        let cuboid = Mesh::<TexVertex>::cuboid(min, max, texs, uvs);
        array::from_fn(|i| cuboid[i].with_normal(Direction::by_idx(i).vector()))
    }

    /// The same mesh with the given texture on every vertex.
    pub fn with_texture(&self, tex: Id) -> Self {
        Mesh {
            vertices: self.vertices.iter().map(|v| v.with_texture(tex)).collect(),
            indices: self.indices.clone(),
        }
    }
}
