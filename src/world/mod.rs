mod block;
mod generation;
mod model;
mod region;
mod render;

use crate::actor::{PlayerControlled, Position, Rotation};
use crate::ecs::*;
use crate::render::{
	AlphaVertex, Camera, Canvas, Frame, FromConfig, IntTransInst, NormTexVertex, RenderBatch,
	RenderBatchConfig, RenderDescriptor, TextureArraySampler, Transformation,
};
use crate::resources;
use crate::util::bounding::PlaneGroup;
use crate::util::collection::Volume;
use crate::util::math::L1ShellIter;
pub use block::*;
use crossbeam_channel::Sender;
pub use generation::*;
use glam::IVec3;
use log::error;
pub use model::*;
use rayon::ThreadPool;
pub use region::*;
pub use render::*;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use wgpu::{Color, LoadOp, PrimitiveTopology};

pub type RegionPos = IVec3;
pub type BlockPos = IVec3;
pub type Chunk = Volume<Meta>;
pub const LOD_COUNT: usize = MAX_LOD as usize + 1;
static SAVE_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_mins(5));

pub struct World {
    center: RegionPos,
    lod_radii: SmallVec<[u32; LOD_COUNT]>,
    update_level: usize,
    update_iters: SmallVec<[L1ShellIter<RegionPos>; LOD_COUNT + 1]>,
    regions: HashMap<RegionPos, Region>,
    generating_regions: HashSet<RegionPos>,
    meshing_regions: HashSet<RegionPos>,
    changed_regions: HashSet<RegionPos>,
    save_timer: Instant,

    gen_tx: Sender<GenTask>,
    meshing_tx: Sender<MeshingTask>,
}

impl Resource for World {}

impl World {
    const MAX_UPDATE_COST: usize = 32;

    pub fn new(
        lod_radii: Vec<u32>,
        gen_tx: Sender<GenTask>,
        meshing_tx: Sender<MeshingTask>,
    ) -> Self {
        assert!(!lod_radii.is_empty(), "World's lod radii must not be empty");
        assert!(
            lod_radii.len() <= LOD_COUNT,
            "World's lod radii must not be longer than {}",
            LOD_COUNT,
        );

        let mut update_iters = SmallVec::new();
        update_iters.push(L1ShellIter::new(RegionPos::ZERO, 0));
        for &r in lod_radii.iter() {
            update_iters.push(L1ShellIter::new(RegionPos::ZERO, r as i32));
        }

        Self {
            center: RegionPos::ZERO,
            lod_radii: SmallVec::from_vec(lod_radii),
            update_level: 0,
            update_iters,
            regions: HashMap::new(),
            generating_regions: HashSet::new(),
            meshing_regions: HashSet::new(),
            changed_regions: HashSet::new(),
            save_timer: Instant::now(),

            gen_tx,
            meshing_tx,
        }
    }

    #[inline]
    pub fn cast_pos(pos: BlockPos) -> (RegionPos, LocalPos) {
        let region_size = IVec3::splat(REGION_SIZE as i32);
        let region_pos = pos.div_euclid(region_size);
        let rel_block_pos = pos.rem_euclid(region_size).as_u8vec3();
        (region_pos, rel_block_pos)
    }

    #[inline]
    fn pos_influence(pos: BlockPos) -> SmallVec<[RegionPos; 8]> {
        let mut influenced = SmallVec::new();

        fn range(b: i32) -> (i32, i32) {
            let min = (b - 1).div_euclid(REGION_SIZE as i32);
            let max = (b + 1).div_euclid(REGION_SIZE as i32);
            (min, max)
        }

        let (minx, maxx) = range(pos.x);
        let (miny, maxy) = range(pos.y);
        let (minz, maxz) = range(pos.z);

        for x in minx..=maxx {
            for y in miny..=maxy {
                for z in minz..=maxz {
                    influenced.push(RegionPos::new(x, y, z));
                }
            }
        }

        influenced
    }

    pub fn get_block(&self, pos: BlockPos) -> Block {
        let (region_pos, rel_block_pos) = Self::cast_pos(pos);
        if let Some(region) = self.regions.get(&region_pos) {
            region.get_block(rel_block_pos)
        } else {
            Block::air()
        }
    }

    pub fn set_block(&mut self, pos: BlockPos, block: Block) {
        let (region_pos, rel_block_pos) = Self::cast_pos(pos);
        for influenced_region_pos in Self::pos_influence(pos) {
            if let Some(region) = self.regions.get_mut(&influenced_region_pos) {
                let pos = (rel_block_pos.as_i16vec3()
                    + (region_pos - influenced_region_pos).as_i16vec3() * REGION_SIZE as i16
                    + 1)
                .as_u8vec3();
                if region.set_block(pos, block) {
                    self.meshing_regions.insert(influenced_region_pos);
                    self.changed_regions.insert(influenced_region_pos);
                }
            }
        }
    }

    #[inline]
    fn min_radius(&self, level: usize) -> u32 {
        if level == 0 {
            0
        } else {
            self.lod_radii[level - 1]
        }
    }

    #[inline]
    fn max_radius(&self, level: usize) -> u32 {
        if level == self.lod_radii.len() {
            *self.lod_radii.last().unwrap() * 2
        } else {
            self.lod_radii[level]
        }
    }

    pub fn update(&mut self, canvas: &Canvas, center: RegionPos) {
        let center_displacement = (center - self.center).abs().element_sum() as u32;
        self.center = center;

        if center_displacement > 0 {
            for i in 0..=self.lod_radii.len() {
                let min_radius = self.min_radius(i);
                let iter = &mut self.update_iters[i];
                let radius = (iter.radius as u32)
                    .saturating_sub(center_displacement)
                    .max(min_radius);

                *iter = L1ShellIter::new(center, radius as i32);
            }

            if center_displacement >= self.max_radius(self.lod_radii.len()) {
                self.regions.clear();
            }

            self.update_level = 0;
        }

        let mut cost = 0;

        while self.update_level <= self.lod_radii.len() {
            let level = self.update_level;
            let complexity = (LOD_COUNT - level).pow(2);
            let max_radius = self.max_radius(level);

            let iter = &mut self.update_iters[level];
            let lod = if level < self.lod_radii.len() {
                Some(level as u8)
            } else {
                None
            };

            while cost < Self::MAX_UPDATE_COST {
                match iter.next() {
                    Some(pos) => match lod {
                        None => {
                            if let Some(region) = self.regions.get_mut(&pos) {
                                if let Err(e) = region.save() {
                                    error!("Failed to save Region {}: {}", pos, e);
                                }
                                self.regions.remove(&pos);
                            }
                            cost += 1;
                        }

                        Some(lod) => {
                            if let Some(region) = self.regions.get_mut(&pos) {
                                if region.update(lod) {
                                    self.generating_regions.insert(pos);
                                    cost += complexity;
                                } else {
                                    cost += 1;
                                }
                            } else {
                                let region = Region::new(
                                    canvas,
                                    pos,
                                    lod,
                                    self.gen_tx.clone(),
                                    self.meshing_tx.clone(),
                                );
                                self.regions.insert(pos, region);
                                self.generating_regions.insert(pos);
                                cost += complexity;
                            }
                        }
                    },

                    None => break,
                }
            }

            if cost < Self::MAX_UPDATE_COST {
                let next_radius = iter.radius as u32 + 1;
                if next_radius < max_radius {
                    *iter = L1ShellIter::new(center, next_radius as i32);
                } else {
                    self.update_level += 1;
                }
            } else {
                break;
            }
        }

        self.generating_regions.retain(|pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                if region.poll() {
                    self.meshing_regions.insert(*pos);
                    false
                } else {
                    true
                }
            } else {
                false
            }
        });

        if self.save_timer.elapsed() >= *SAVE_DURATION {
            self.save_timer = Instant::now();

            self.save();
        }
    }

    pub fn pre_render(&mut self, canvas: &Canvas) {
        self.meshing_regions.retain(|pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                !region.pre_render(canvas)
            } else {
                false
            }
        });
    }

    pub fn save(&mut self) {
        for pos in self.changed_regions.drain() {
            if let Some(region) = self.regions.get_mut(&pos) {
                if let Err(e) = region.save() {
                    error!("Failed to save Region {}: {}", pos, e);
                }
            }
        }
    }
}

resources! {
    pub struct WorldThreads(ThreadPool, ThreadPool);
}

pub struct WorldUpdater;

impl System for WorldUpdater {
    type CompQuery = (CompRead<PlayerControlled>, CompRead<Position>);
    type ResQuery = (
        ResWrite<World>,
        ResWrite<Generator>,
        ResRead<Canvas>,
        ResRead<WorldThreads>,
    );

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let pos = entry.2.0.floor().as_ivec3();
        let center = pos.div_euclid(IVec3::splat(REGION_SIZE as i32));

        res.0.update(res.2, center);
        res.1.update(res.3);

        None
    }
}

pub struct WorldRenderer {
    block_desc: RenderDescriptor<'static>,
    block_batch: RenderBatch<(TextureArraySampler, Transformation), NormTexVertex, IntTransInst>,

    occlusion_desc: RenderDescriptor<'static>,
    occlusion_batch: RenderBatch<Transformation, AlphaVertex, IntTransInst>,
}

impl System for WorldRenderer {
    type CompQuery = (
        CompRead<PlayerControlled>,
        CompRead<Position>,
        CompRead<Rotation>,
    );
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<Option<Frame>>,
        ResRead<BlockTextures>,
        ResRead<Camera>,
        ResWrite<World>,
    );

    fn operate(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        res.4.pre_render(res.0);

        let mut regions = res.4.regions.values().collect::<Vec<_>>();
        regions.retain(|&region| res.3.frustum.is_aabb_inside(region.bound()));

        if let Some(frame) = res.1 {
            frame.render(&self.block_desc, |mut pass| {
                self.block_batch.begin(&mut pass);
                self.block_batch
                    .push(&mut pass, (&res.2.0, &res.3.transform));

                regions.iter().for_each(|&region| {
                    self.block_batch.draw(&mut pass, region);
                });
            });

            frame.render(&self.occlusion_desc, |mut pass| {
                self.occlusion_batch.begin(&mut pass);
                self.occlusion_batch.push(&mut pass, &res.3.transform);

                regions.iter().for_each(|&region| {
                    self.occlusion_batch.draw(&mut pass, region);
                });
            })
        }

        None
    }
}

impl WorldRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            block_desc: RenderDescriptor {
                name: "block",
                color_load: LoadOp::Clear(Color {
                    r: 0.375,
                    g: 0.625,
                    b: 1.0,
                    a: 1.0,
                }),
                depth_load: LoadOp::Clear(0.0),
            },
            block_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "block",
                    shader: "block",
                    translucent: false,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),

            occlusion_desc: RenderDescriptor {
                name: "occlusion",
                color_load: LoadOp::Load,
                depth_load: LoadOp::Load,
            },
            occlusion_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "occlusion",
                    shader: "occlusion",
                    translucent: true,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: false,
                },
            ),
        }
    }
}
