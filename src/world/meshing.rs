use crate::ecs::*;
use crate::render::{AlphaVertex, Mesh, MeshGroup, NormTexVertex};
use crate::util::collection::{CubicVec, Registry};
use crate::util::coord::{Axis, Coord3, Direction, ICoord3};
use crate::util::Id;
use crate::world::*;
use crossbeam_channel::{Receiver, Sender};
use glam::{U8Vec3, Vec2, Vec3};
use log::error;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::array;
use std::collections::HashMap;
use std::sync::LazyLock;

pub static BLOCK_MESH_TEMPLATES: LazyLock<Registry<BlockMeshTemplate>> =
    LazyLock::new(|| build_block_mesh_templates());

fn build_block_mesh_templates() -> Registry<BlockMeshTemplate> {
    let mut templates = Registry::new();

    let [cube_w, cube_e, cube_d, cube_u, cube_n, cube_s] =
        BlockMeshTemplate::cuboid(Vec3::ZERO, Vec3::ONE, [false; 6]);

    templates.register(0, cube_w);
    templates.register(1, cube_e);
    templates.register(2, cube_d);
    templates.register(3, cube_u);
    templates.register(4, cube_n);
    templates.register(5, cube_s);

    templates
}

#[derive(Default)]
pub struct BlockModel {
    meshes: Vec<TemplatedMesh>,
    cull: [bool; 6],
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TemplatedMesh {
    pub template: Id,
    pub texture: Id,
}

pub struct BlockMeshTemplate {
    mesh: Mesh<NormTexVertex>,
    translucent: bool,
    cull: Option<Direction>,
    spans: [Option<MergeSpan>; 3],
}

pub struct MergeSpan {
    ends: Vec<usize>,
    uv_unit: Vec2,
}

impl BlockMeshTemplate {
    pub fn cuboid(min: Vec3, max: Vec3, translucent: [bool; 6]) -> [Self; 6] {
        let uvs = array::from_fn(|_| {
            vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(0.0, 1.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(1.0, 0.0),
            ]
        });

        let mut merge_axis = Axis::ALL.to_vec();
        merge_axis.retain(|&a| min.get(a) < 0.001 && max.get(a) > 0.999);

        let cuboid = Mesh::<NormTexVertex>::cuboid(min, max, [0; 6], uvs);

        array::from_fn(|i| {
            let dir = Direction::by_idx(i);
            let axis = dir.axis();
            let cull = if {
                if dir.positive() {
                    max.get(axis) > 0.999
                } else {
                    min.get(axis) < 0.001
                }
            } {
                Some(dir)
            } else {
                None
            };

            let mut spans = [None, None, None];

            for &maxis in merge_axis.iter() {
                if maxis != axis {
                    let (ends, uv_unit) = match maxis {
                        Axis::X => match dir {
                            Direction::Down => (vec![0, 3], Vec2::new(0.0, -1.0)),
                            Direction::Up | Direction::South => (vec![2, 3], Vec2::new(1.0, 0.0)),
                            Direction::North => (vec![0, 1], Vec2::new(-1.0, 0.0)),
                            _ => unreachable!(),
                        },
                        Axis::Y => (vec![0, 3], Vec2::new(0.0, -1.0)),
                        Axis::Z => match dir {
                            Direction::West => (vec![2, 3], Vec2::new(1.0, 0.0)),
                            Direction::East | Direction::Down => (vec![0, 1], Vec2::new(-1.0, 0.0)),
                            Direction::Up => (vec![1, 2], Vec2::new(0.0, 1.0)),
                            _ => unreachable!(),
                        },
                    };

                    spans[maxis.idx()] = Some(MergeSpan { ends, uv_unit });
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
    pub fn new(meshes: Vec<TemplatedMesh>) -> Self {
        let mut cull = [false; 6];

        for mesh in meshes.iter() {
            let template = BLOCK_MESH_TEMPLATES.get(mesh.template);
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

pub struct MeshingTask {
    pub pos: RegionPos,
    pub chunk_pos: Option<ChunkPos>,
    pub chunk: Chunk,
}

pub struct MeshingResult {
    pub pos: RegionPos,
    pub chunk_pos: Option<ChunkPos>,
    pub block_mesh: Mesh<NormTexVertex>,
    pub occlusion_mesh: Mesh<AlphaVertex>,
}

pub struct ChunkMesher {
    task_rs: Receiver<MeshingTask>,
    result_tx: Sender<MeshingResult>,
}

impl Resource for ChunkMesher {}

impl ChunkMesher {
    pub fn new(task_rs: Receiver<MeshingTask>, result_tx: Sender<MeshingResult>) -> Self {
        Self { task_rs, result_tx }
    }

    pub fn update(&mut self, threads: &WorldThreads) {
        let WorldThreads(near_thread, far_thread) = threads;

        while let Ok(task) = self.task_rs.try_recv() {
            let result_tx = self.result_tx.clone();
            let threads = match task.chunk_pos {
                Some(_) => near_thread,
                None => far_thread,
            };

            threads.spawn(move || {
                let result = Self::perform(&task);

                if result_tx.try_send(result).is_err() {
                    error!(
                        "Failed to send Chunk Meshing Result of Region at {} to Rendered World",
                        task.pos
                    );
                }
            });
        }
    }

    fn perform(task: &MeshingTask) -> MeshingResult {
        let (mut block_mesh, mut occlusion_mesh) = Self::build_meshes(&task.chunk);

        if task.chunk_pos.is_none() {
            let scale = (REGION_SIZE / (task.chunk.side - 2)) as f32;
            block_mesh.multiply(scale);
            occlusion_mesh.multiply(scale);
        }

        MeshingResult {
            pos: task.pos,
            chunk_pos: task.chunk_pos,
            block_mesh,
            occlusion_mesh,
        }
    }

    fn build_meshes(chunk: &Chunk) -> (Mesh<NormTexVertex>, Mesh<AlphaVertex>) {
        let mut temp_poses = HashMap::new();
        let n = chunk.side - 2;

        let mut occlusion_mesh = Mesh::new();

        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let pos = RelBlockPos::new(x, y, z);
                    let real_pos = pos + 1;
                    let block = Block::from_meta(*chunk.get(real_pos));

                    for temp_mesh in block.model().meshes.iter() {
                        let template = BLOCK_MESH_TEMPLATES.get(temp_mesh.template);

                        if let Some(dir) = template.cull {
                            let adj_pos = real_pos.step(dir);
                            let adj_block = Block::from_meta(*chunk.get(adj_pos));

                            if adj_block.model().culls(dir.opposite()) {
                                continue;
                            }
                        }

                        if !template.translucent {
                            let mut vertices = Vec::new();
                            let mut add = false;

                            for vertex in template.mesh.vertices.iter() {
                                let pos = vertex.pos;
                                let mut alpha = 0.0;

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

                        temp_poses
                            .entry(*temp_mesh)
                            .or_insert_with(|| CubicVec::<bool>::new(n))
                            .set(pos, true);
                    }
                }
            }
        }

        let block_mesh = temp_poses
            .into_par_iter()
            .map(|(temp_mesh, mut poses)| {
                let template = BLOCK_MESH_TEMPLATES.get(temp_mesh.template);
                let mut merged = Mesh::new();

                for x in 0..n {
                    for y in 0..n {
                        for z in 0..n {
                            let pos = RelBlockPos::new(x, y, z);

                            if !poses.get(pos) {
                                continue;
                            }

                            let mut mesh = template.mesh.translated(pos.as_vec3());

                            let spans = &template.spans;
                            let mut dpos = U8Vec3::ZERO;

                            for &axis in Axis::ALL {
                                let i = axis.idx();

                                if let Some(span) = &spans[i] {
                                    let max = n - pos.get(axis);

                                    let dist = (1..max)
                                        .find(|&d| 'a: {
                                            let pos1 = pos.shift(axis, d);

                                            if !poses.get(pos1) {
                                                break 'a true;
                                            }

                                            for j in 0..i {
                                                let axis1 = Axis::by_idx(j);

                                                for k in 1..=dpos.get(axis1) {
                                                    let pos2 = pos1.shift(axis1, k);
                                                    if !poses.get(pos2) {
                                                        break 'a true;
                                                    }
                                                }
                                            }

                                            false
                                        })
                                        .map(|d| d - 1)
                                        .unwrap_or(max - 1);

                                    for &end in span.ends.iter() {
                                        let vertex = &mut mesh.vertices[end];

                                        vertex.pos = vertex.pos.shift(axis, dist as f32);
                                        vertex.uv += span.uv_unit * dist as f32;
                                    }
                                    dpos = dpos.with(axis, dist);
                                }
                            }

                            poses.fill(pos, pos + dpos + 1, false);

                            merged.merge(&mesh);
                        }
                    }
                }

                merged.with_texture(temp_mesh.texture)
            })
            .collect::<Vec<_>>()
            .merge();

        (block_mesh, occlusion_mesh)
    }
}

pub struct ChunkMeshing;

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
