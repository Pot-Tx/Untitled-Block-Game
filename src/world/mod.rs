mod block;
mod generation;
mod model;
mod region;

use crate::actor::{PlayerControlled, Position, PrevPos, Rotation};
use crate::ecs::*;
use crate::render::{
    AlphaVertex, Camera, Canvas, Frame, FromConfig, IntTransInst, NormTexVertex, RenderBatch,
    RenderBatchConfig, RenderDescriptor, TextureArraySampler, Transformation,
};
use crate::resources;
use crate::util::bounding::{PlaneGroup, AABB};
use crate::util::collection::Volume;
use crate::util::coord::{Axis, Coord3};
use crate::util::math::CubeShellIter;
pub use block::*;
use crossbeam_channel::Sender;
pub use generation::*;
use glam::{IVec3, U8Vec3};
use log::error;
pub use model::*;
use rayon::ThreadPool;
pub use region::*;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use wgpu::{Color, LoadOp, PrimitiveTopology};

pub type RegionPos = IVec3;
pub type BlockPos = IVec3;
pub type Chunk = Volume<Meta>;
pub const LOD_COUNT: usize = MAX_LOD as usize + 1;
static SAVE_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_mins(5));

pub struct RingVolume<T: Clone> {
    pub volume: Volume<Option<T>>,
    pub bound: AABB<IVec3>,
}

impl<T: Clone> RingVolume<T> {
    pub fn new(center: IVec3, radius: u8) -> Self {
        Self {
            volume: Volume::new(U8Vec3::splat(radius * 2 + 1)),
            bound: AABB {
                min: center - radius as i32,
                max: center + radius as i32 + 1,
            },
        }
    }

    #[inline]
    fn cast_pos(&self, pos: IVec3) -> U8Vec3 {
        pos.rem_euclid(self.volume.size.as_ivec3()).as_u8vec3()
    }

    pub fn get(&self, pos: IVec3) -> Option<&T> {
        if self.bound.is_point_inside(pos) {
            self.volume.get(self.cast_pos(pos)).as_ref()
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, pos: IVec3) -> Option<&mut T> {
        if self.bound.is_point_inside(pos) {
            self.volume.get_mut(self.cast_pos(pos)).as_mut()
        } else {
            None
        }
    }

    pub fn set(&mut self, pos: IVec3, value: T) -> Option<T> {
        if self.bound.is_point_inside(pos) {
            self.volume.set(self.cast_pos(pos), Some(value))
        } else {
            None
        }
    }

    pub fn translate(&mut self, dpos: IVec3) {
        let new_bound = self.bound.translate(dpos);

        for &axis in Axis::ALL {
            let d = dpos.get(axis);
            if d != 0 {
                let (min, max) = match d > 0 {
                    false => (new_bound.max.get(axis), self.bound.max.get(axis)),
                    true => (self.bound.min.get(axis), new_bound.min.get(axis)),
                };

                let size = self.volume.size.get(axis) as i32;
                let min = min.rem_euclid(size) as u8;
                let max = max.rem_euclid(size) as u8;

                if min < max {
                    self.volume.fill(
                        U8Vec3::ZERO.with(axis, min),
                        self.volume.size.with(axis, max),
                        None,
                    );
                } else {
                    self.volume
                        .fill(U8Vec3::ZERO, self.volume.size.with(axis, max), None);
                    self.volume
                        .fill(U8Vec3::ZERO.with(axis, min), self.volume.size, None);
                }
            }
        }

        self.bound = new_bound;
    }
}

pub struct World {
    lod_radii: SmallVec<[u8; LOD_COUNT]>,
    update_lod: u8,
    update_iters: SmallVec<[Option<CubeShellIter<RegionPos>>; LOD_COUNT]>,
    regions: RingVolume<Region>,
    generating_regions: HashSet<RegionPos>,
    meshing_regions: HashSet<RegionPos>,
    changed_regions: HashSet<RegionPos>,
    save_timer: Instant,

    gen_tx: Sender<GenTask>,
    meshing_tx: Sender<MeshingTask>,
}

resources! {
    pub struct Gravity(f32);
}

impl Resource for World {}

impl World {
    const MAX_UPDATE_COST: usize = 1024;

    pub fn new(
        center: RegionPos,
        lod_radii: SmallVec<[u8; LOD_COUNT]>,
        gen_tx: Sender<GenTask>,
        meshing_tx: Sender<MeshingTask>,
    ) -> Self {
        let mut update_iters = SmallVec::new();
        for i in 0..lod_radii.len() {
            let radius = if i == 0 { 0 } else { lod_radii[i - 1] + 1 };
            update_iters.push(Some(CubeShellIter::from_center(center, radius as i32)));
        }

        for &r in lod_radii.iter() {
            update_iters.push(Some(CubeShellIter::from_center(center, r as i32)));
        }
        let regions = RingVolume::new(center, *lod_radii.last().unwrap() as u8);

        Self {
            lod_radii,
            update_lod: 0,
            update_iters,
            regions,
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
        if let Some(region) = self.regions.get(region_pos) {
            region.get_block(rel_block_pos)
        } else {
            Block::air()
        }
    }

    pub fn set_block(&mut self, pos: BlockPos, block: Block) {
        let (region_pos, rel_block_pos) = Self::cast_pos(pos);
        for influenced_region_pos in Self::pos_influence(pos) {
            if let Some(region) = self.regions.get_mut(influenced_region_pos) {
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

    pub fn update(&mut self, canvas: &Canvas, center: RegionPos, translation: IVec3) {
        if translation != IVec3::ZERO {
            self.regions.translate(translation);
            let translation = translation.abs().element_sum();

            for i in 0..self.lod_radii.len() as u8 {
                let iter = &mut self.update_iters[i as usize];

                let radius = match iter {
                    None => self.lod_radii[i as usize] as i32,

                    Some(iter) => {
                        let min_radius = if i == 0 {
                            0
                        } else {
                            self.lod_radii[(i - 1) as usize] + 1
                        };
                        (((iter.max - iter.origin).x / 2) - translation).max(min_radius as i32)
                    }
                };

                *iter = Some(CubeShellIter::from_center(center, radius));
            }

            self.update_lod = 0;
        }

        let mut cost = 0;

        while self.update_lod < self.lod_radii.len() as u8 {
            let level = self.update_lod as usize;
            let complexity = (LOD_COUNT - level).pow(2);
            let max_radius = self.lod_radii[level];

            if let Some(iter) = &mut self.update_iters[level] {
                while cost < Self::MAX_UPDATE_COST {
                    match iter.next() {
                        Some(pos) => {
                            if let Some(region) = self.regions.get_mut(pos) {
                                if region.update(self.update_lod) {
                                    self.generating_regions.insert(pos);
                                    cost += complexity;
                                } else {
                                    cost += 1;
                                }
                            } else {
                                let region = Region::new(
                                    canvas,
                                    pos,
                                    self.update_lod,
                                    self.gen_tx.clone(),
                                    self.meshing_tx.clone(),
                                );
                                self.regions.set(pos, region);
                                self.generating_regions.insert(pos);
                                cost += complexity;
                            }
                        }

                        None => break,
                    }
                }

                if cost < Self::MAX_UPDATE_COST {
                    let next_radius = ((iter.max - iter.origin).x as u8 / 2) + 1;
                    if next_radius <= max_radius {
                        *iter = CubeShellIter::from_center(center, next_radius as i32);
                    } else {
                        self.update_iters[level] = None;
                        self.update_lod += 1;
                    }
                } else {
                    break;
                }
            }
        }

        self.generating_regions.retain(|&pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                if region.poll() {
                    self.meshing_regions.insert(pos);
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
        self.meshing_regions.retain(|&pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                !region.pre_render(canvas)
            } else {
                false
            }
        });
    }

    pub fn save(&mut self) {
        for pos in self.changed_regions.drain() {
            if let Some(region) = self.regions.get_mut(pos) {
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
    type CompQuery = (
        CompRead<PlayerControlled>,
        CompRead<Position>,
        CompRead<PrevPos>,
    );
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
        let prev_pos = entry.3.0.floor().as_ivec3();
        let translation = center - prev_pos.div_euclid(IVec3::splat(REGION_SIZE as i32));

        res.0.update(res.2, center, translation);
        res.1.update(translation, res.3);

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

        let mut regions = res
            .4
            .regions
            .volume
            .vec
            .iter()
            .filter(|&region| region.is_some())
            .map(|region| region.as_ref().unwrap())
            .collect::<Vec<_>>();
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
