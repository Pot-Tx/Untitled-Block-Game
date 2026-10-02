//! Block models and the mesher that turns chunks into geometry.
//!
//! A block model is a list of parts, each of which is a mesh template with a
//! texture. Parts of the same model that share a texture are merged into
//! rectangles before they are uploaded, which is what keeps the vertex count of
//! a chunk low.

use crate::ecs::*;
use crate::render::*;
use crate::util::collection::Registry;
use crate::util::coord::{Axis, Coord3, Direction, ICoord3};
use crate::util::Id;
use crate::world::*;
use crossbeam_channel::{Receiver, Sender};
use glam::{U8Vec3, Vec2, Vec3};
use log::error;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::array;
use std::collections::HashMap;
use crate::util::OnceInit;

/// The templates the block models are built from, loaded from
/// `assets/models/block/templates`.
pub(super) static BLOCK_MODEL_TEMPLATES: OnceInit<Registry<BlockModelTemplate>> = OnceInit::new();

/// The textures of the blocks, loaded from `assets/textures/block`.
pub(super) static BLOCK_TEXTURES: OnceInit<Registry<Tex>> = OnceInit::new();

/// Builds the block model templates.
pub(super) fn build_block_model_templates() -> Registry<BlockModelTemplate> {
    Registry::load_rons_from("assets/models/block/templates")
        .expect("failed to load block model templates")
}

/// Builds the block textures.
pub(super) fn build_block_textures() -> Registry<Tex> {
    Registry::<Tex>::load_from("assets/textures/block").expect("failed to load block textures")
}

resources! {
    pub(super) struct BlockTextures(BindSet<TextureArraySampler>);
}

#[derive(Clone, Default)]
pub struct BlockModel {
    meshes: Vec<BlockModelPart>,
    /// Whether the face in the direction at the index is culled by an opaque
    /// neighbour; indexed by [`Direction::idx`].
    cull: [bool; 6],
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockModelPart {
    pub template: Id,
    pub texture: Id,
}

/// A block model part as it is written to a `.ron` file, with names instead of
/// ids so that the registries can be reordered without breaking the files.
#[derive(Serialize, Deserialize)]
struct RawBlockModelPart<'a> {
    template: &'a str,
    texture: &'a str,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
pub struct BlockModelTemplate {
    mesh: Mesh<NormUvVertex>,
    translucent: bool,
    /// The direction the mesh can be culled from, when it covers a whole face of
    /// a block.
    cull: Option<Direction>,
    spans: SmallVec<[MergeSpan; 2]>,
}

/// `ends` lists the vertices that have to be moved when the face is stretched,
/// and `uv_unit` is how far the uv of those vertices advances per block.
#[derive(Serialize, Deserialize)]
pub struct MergeSpan {
    axis: Axis,
    ends: Vec<usize>,
    uv_unit: Vec2,
}

impl Serialize for BlockModel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.meshes.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BlockModel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let meshes = Vec::<BlockModelPart>::deserialize(deserializer)?;
        Ok(Self::new(meshes))
    }
}

impl Serialize for BlockModelPart {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let raw = RawBlockModelPart {
            template: BLOCK_MODEL_TEMPLATES.name_of(self.template),
            texture: BLOCK_TEXTURES.name_of(self.texture),
        };
        raw.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BlockModelPart {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawBlockModelPart::deserialize(deserializer)?;
        Ok(Self {
            template: BLOCK_MODEL_TEMPLATES.id_of(raw.template),
            texture: BLOCK_TEXTURES.id_of(raw.texture),
        })
    }
}

impl BlockModelTemplate {
    /// The six faces of the box from `min` to `max`, in the order of
    /// [`Direction::ALL`], which is also the order of `Mesh::cuboid`.
    pub fn cuboid(min: Vec3, max: Vec3, translucent: [bool; 6]) -> [Self; 6] {
        // The u and v axes of each face, chosen so that the texture is not
        // mirrored when it is seen from outside the block.
        let uvs = Direction::ALL.map(|dir| {
            let (udir, vdir) = match dir {
                Direction::West => (Direction::South, Direction::Down),
                Direction::East => (Direction::North, Direction::Down),
                Direction::Down => (Direction::North, Direction::West),
                Direction::Up => (Direction::South, Direction::East),
                Direction::North => (Direction::West, Direction::Down),
                Direction::South => (Direction::East, Direction::Down),
            };

            let (umin, umax) = match udir.positive() {
                false => (1.0 - max.get(udir.axis()), 1.0 - min.get(udir.axis())),
                true => (min.get(udir.axis()), max.get(udir.axis())),
            };
            let (vmin, vmax) = match vdir.positive() {
                false => (1.0 - max.get(vdir.axis()), 1.0 - min.get(vdir.axis())),
                true => (min.get(vdir.axis()), max.get(vdir.axis())),
            };

            vec![
                Vec2::new(umin, vmin),
                Vec2::new(umin, vmax),
                Vec2::new(umax, vmax),
                Vec2::new(umax, vmin),
            ]
        });

        // A face that covers the whole box can be merged with the faces of the
        // neighbouring blocks, which is what makes a flat wall a single quad.
        let mut merge_axis = Axis::ALL.to_vec();
        merge_axis.retain(|&a| min.get(a) < 0.001 && max.get(a) > 0.999);

        let cuboid = Mesh::<NormUvVertex>::cuboid(min, max, uvs);

        array::from_fn(|i| {
            let dir = Direction::by_idx(i);
            let axis = dir.axis();
            let cull = if if dir.positive() {
                max.get(axis) > 0.999
            } else {
                min.get(axis) < 0.001
            } {
                Some(dir)
            } else {
                None
            };

            let mut spans = SmallVec::new();

            for &maxis in merge_axis.iter() {
                if maxis != axis {
                    let (ends, uv_unit) = match maxis {
                        Axis::X => match dir {
                            Direction::Down => (vec![0, 3], Vec2::new(0.0, -1.0)),
                            Direction::Up | Direction::South => (vec![2, 3], Vec2::new(1.0, 0.0)),
                            Direction::North => (vec![0, 1], Vec2::new(-1.0, 0.0)),
                            // The other directions cannot produce a full x face.
                            _ => unreachable!("{:?} cannot merge along the x axis", dir),
                        },
                        Axis::Y => (vec![0, 3], Vec2::new(0.0, -1.0)),
                        Axis::Z => match dir {
                            Direction::West => (vec![2, 3], Vec2::new(1.0, 0.0)),
                            Direction::East | Direction::Down => (vec![0, 1], Vec2::new(-1.0, 0.0)),
                            Direction::Up => (vec![1, 2], Vec2::new(0.0, 1.0)),
                            // The other directions cannot produce a full z face.
                            _ => unreachable!("{:?} cannot merge along the z axis", dir),
                        },
                    };

                    spans.push(MergeSpan {
                        axis: maxis,
                        ends,
                        uv_unit,
                    });
                }
            }

            Self {
                mesh: cuboid[i].clone(),
                translucent: translucent[i],
                cull,
                spans,
            }
        })
    }
}

impl BlockModel {
    pub fn new(meshes: Vec<BlockModelPart>) -> Self {
        let mut cull = [false; 6];

        for mesh in meshes.iter() {
            let template = BLOCK_MODEL_TEMPLATES.get(mesh.template);
            if let Some(dir) = template.cull {
                cull[dir.idx()] = true;
            }
        }

        Self { meshes, cull }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    fn culls(&self, dir: Direction) -> bool {
        self.cull[dir.idx()]
    }
}

/// A request to mesh a chunk: a single sub-region when `pos` is set, otherwise
/// the whole region.
pub struct MeshingTask {
    pub lod: u8,
    pub pos: Option<SubRegionPos>,
    pub chunk: Chunk,
    pub tx: Sender<MeshingResult>,
}

pub struct MeshingResult {
    pub lod: u8,
    pub pos: Option<SubRegionPos>,
    pub blocks: Mesh<NormTexVertex>,
    pub occlusion: Mesh<AlphaVertex>,
}

pub struct ChunkMesher {
    task_rx: Receiver<MeshingTask>,
}

/// Merges the equally shaped faces of one template into larger rectangles.
///
/// The blocks are stored as one bitmask per line: `lines` is indexed by two of
/// the three axes, and a set bit marks a block along the merge axis
/// `axes[0]`. `two` tells whether the template may also be merged along a second
/// axis, which is what turns a row into a rectangle.
struct MeshMerger {
    lines: Vec<u64>,
    side: u8,
    axes: [Axis; 3],
    two: bool,
    current: (usize, u8),
}

impl Resource for ChunkMesher {}

impl ChunkMesher {
    pub fn new(task_rx: Receiver<MeshingTask>) -> Self {
        Self { task_rx }
    }

    /// Sub-regions of the near levels of detail are meshed by the near pool, so
    /// that a queue of far regions cannot delay the geometry around the player.
    pub fn update(&mut self, threads: &WorldThreads) {
        let WorldThreads(near_threads, far_threads) = threads;

        let mut near_tasks = Vec::new();
        let mut far_tasks = Vec::new();
        for task in self.task_rx.try_iter() {
            match task.pos {
                None => far_tasks.push(task),
                Some(_) => near_tasks.push(task),
            }
        }

        near_threads.spawn(move || {
            near_tasks.into_par_iter().for_each(|task| {
                Self::perform(task);
            })
        });

        far_threads.spawn(move || {
            far_tasks.into_par_iter().for_each(|task| {
                Self::perform(task);
            })
        });
    }

    fn perform(task: MeshingTask) {
        let (mut blocks, mut occlusion) = Self::build_meshes(&task.chunk);

        match task.pos {
            None => {
                let scale = Region::block_size_on_lod(task.lod) as f32;
                blocks.multiply(scale);
                occlusion.multiply(scale);
            }

            Some(pos) => {
                let offset = (pos * SUBREGION_SIZE).as_vec3();
                blocks.translate(offset);
                occlusion.translate(offset);
            }
        }

        let result = MeshingResult {
            lod: task.lod,
            pos: task.pos,
            blocks,
            occlusion,
        };

        if task.tx.try_send(result).is_err() {
            error!("failed to send meshes to region");
        }
    }

    fn build_meshes(chunk: &Chunk) -> (Mesh<NormTexVertex>, Mesh<AlphaVertex>) {
        // Merging needs to see every block of a template, so the positions are
        // collected first and merged afterwards.
        let mut temp_mergers = HashMap::new();
        let [w, h, d] = (chunk.size - 2).to_array();

        let mut block_mesh = Mesh::new();
        let mut occlusion_mesh = Mesh::new();

        for x in 0..w {
            for y in 0..h {
                for z in 0..d {
                    let pos = LocalPos::new(x, y, z);
                    let real_pos = pos + 1;
                    let block = Block::from_meta(*chunk.get(real_pos));

                    for temp_mesh in block.model().meshes.iter() {
                        let template = BLOCK_MODEL_TEMPLATES.get(temp_mesh.template);

                        // A face that a full block covers is never visible, so it
                        // is left out of the mesh.
                        if let Some(dir) = template.cull {
                            let adj_pos = real_pos.step(dir);
                            let adj_block = Block::from_meta(*chunk.get(adj_pos));

                            if adj_block.model().culls(dir.opposite()) {
                                continue;
                            }
                        }

                        if !template.translucent {
                            // The occlusion mesh shades the vertices that touch an
                            // opaque block, which draws the ambient occlusion of
                            // the block corners.
                            let mut vertices = Vec::new();
                            let mut add = false;

                            for vertex in template.mesh.vertices.iter() {
                                let pos = vertex.pos;
                                let mut alpha = 0.0;

                                // Only the opaque blocks that touch this vertex
                                // shade it: the two neighbours beside the face of
                                // a culled template, and the three blocks around
                                // the vertex of an unculled one. Every opaque
                                // neighbour adds a fixed share of occlusion.
                                match template.cull {
                                    Some(dir) => {
                                        let adj_pos = real_pos.step(dir);
                                        for &axis in dir.axis().others() {
                                            let dir = axis.direction(pos.get(axis) > 0.5);
                                            let block =
                                                Block::from_meta(*chunk.get(adj_pos.step(dir)));
                                            if block.block_type.opacity.abs().element_sum() > 2.25 {
                                                alpha += 0.375;
                                            }
                                        }
                                    }

                                    None => {
                                        for &axis in Axis::ALL {
                                            let dir = axis.direction(pos.get(axis) > 0.5);
                                            let block =
                                                Block::from_meta(*chunk.get(real_pos.step(dir)));
                                            if block.block_type.opacity.abs().element_sum() > 2.25 {
                                                alpha += 0.25;
                                            }
                                        }
                                    }
                                }

                                if alpha > 0.0 {
                                    add = true;
                                }

                                vertices.push(AlphaVertex { pos, alpha });
                            }

                            if add {
                                let mesh = Mesh {
                                    vertices,
                                    indices: template.mesh.indices.clone(),
                                }
                                .translated(pos.as_vec3());

                                occlusion_mesh.merge(&mesh);
                            }
                        }

                        if template.spans.is_empty() {
                            block_mesh.merge(
                                &template
                                    .mesh
                                    .with_texture(temp_mesh.texture)
                                    .translated(pos.as_vec3()),
                            );
                        } else {
                            temp_mergers
                                .entry(*temp_mesh)
                                .or_insert_with(|| MeshMerger::new(w, &template.spans))
                                .add(pos);
                        }
                    }
                }
            }
        }

        let merged = temp_mergers
            .into_par_iter()
            .map(|(temp_mesh, merger)| {
                let template = BLOCK_MODEL_TEMPLATES.get(temp_mesh.template);
                let base = template.mesh.with_texture(temp_mesh.texture);

                merger
                    .map(|(pos, extent)| {
                        let mut mesh = base.translated(pos.as_vec3());

                        for (i, dist) in extent.into_iter().enumerate() {
                            let span = &template.spans[i];

                            for &end in span.ends.iter() {
                                let vertex = &mut mesh.vertices[end];

                                vertex.pos = vertex.pos.shift(span.axis, dist as f32);
                                vertex.uv += span.uv_unit * dist as f32;
                            }
                        }

                        mesh
                    })
                    .collect::<Vec<_>>()
                    .merge()
            })
            .collect::<Vec<_>>()
            .merge();

        block_mesh.merge(&merged);

        (block_mesh, occlusion_mesh)
    }
}

impl Iterator for MeshMerger {
    type Item = (U8Vec3, SmallVec<[u8; 2]>);

    /// Takes the run of set bits at the current position, then grows the
    /// rectangle over the following lines for as long as they have a run of the
    /// same length. The item is the corner of the rectangle and how far it
    /// extends along each merge axis.
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.side as usize;

        let (idx, bit) = &mut self.current;

        while *idx < n * n {
            *bit += (self.lines[*idx] >> (*bit)).trailing_zeros() as u8;
            if *bit >= self.side {
                *bit = 0;
                *idx += 1;
            } else {
                break;
            }
        }

        let (idx, bit) = self.current;

        if idx < n * n {
            let pos = self.pos_of_bit(self.current);
            let mut extent = SmallVec::new();

            // The run of set bits is the length of the rectangle along the first
            // merge axis; `trailing_ones` counts it, `dist` is the distance the
            // far vertices have to be moved.
            let dist = (self.lines[idx] >> bit).trailing_ones() as u8 - 1;
            let removal = !(((1u64 << (dist + 1)) - 1) << bit);
            extent.push(dist);
            self.lines[idx] &= removal;

            if self.two {
                let mut dist1 = 0;

                // Only the lines of the same row can extend the rectangle, and
                // they have to cover the run that was just consumed.
                for idx1 in idx + 1..((idx / n) + 1) * n {
                    if (self.lines[idx1] >> bit).trailing_ones() as u8 > dist {
                        dist1 += 1;
                        self.lines[idx1] &= removal;
                    } else {
                        break;
                    }
                }

                extent.push(dist1);
            }

            Some((pos, extent))
        } else {
            None
        }
    }
}

impl MeshMerger {
    fn new(side: u8, spans: &[MergeSpan]) -> Self {
        // The merge axes take the first slots of `axes`, so that the bit of a
        // position is its coordinate along the merge axis.
        let masks = vec![0; (side as usize).pow(2)];
        let mut axes = *Axis::ALL;
        let span_count = spans.len();
        for (i, span) in spans.iter().enumerate().take(span_count) {
            axes.swap(i, span.axis.idx());
        }

        Self {
            lines: masks,
            side,
            axes,
            two: span_count > 1,
            current: (0, 0),
        }
    }

    #[inline]
    fn pos_of_bit(&self, bit: (usize, u8)) -> U8Vec3 {
        let (idx, bit) = bit;
        let n = self.side as usize;

        U8Vec3::ZERO
            .with(self.axes[0], bit)
            .with(self.axes[1], (idx % n) as u8)
            .with(self.axes[2], (idx / n) as u8)
    }

    #[inline]
    fn bit_of_pos(&self, pos: U8Vec3) -> (usize, u8) {
        debug_assert!(pos.x < self.side && pos.y < self.side && pos.z < self.side);

        let n = self.side as usize;
        let idx = pos.get(self.axes[1]) as usize + pos.get(self.axes[2]) as usize * n;
        (idx, pos.get(self.axes[0]))
    }

    fn add(&mut self, pos: U8Vec3) {
        let (idx, bit) = self.bit_of_pos(pos);
        self.lines[idx] |= 1u64 << bit;
    }
}

pub(super) struct ChunkMeshing;

impl System for ChunkMeshing {
    type CompQuery = ();
    type ResQuery = (ResWrite<ChunkMesher>, ResRead<WorldThreads>);

    fn operate<'a>(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'a>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'a>,
    ) -> Option<Vec<Command>> {
        res.0.update(res.1);

        None
    }
}
