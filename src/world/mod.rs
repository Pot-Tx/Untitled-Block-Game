mod block;
mod generation;
mod meshing;
mod region;
mod render;

use crate::actor::{PlayerControlled, Position};
use crate::ecs::*;
use crate::resources;
use crate::util::math::L1ShellIter;
use crossbeam_channel::Sender;
use glam::IVec3;
use rayon::ThreadPool;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;

pub use block::*;
pub use generation::*;
pub use meshing::*;
pub use region::*;
pub use render::*;

pub type RegionPos = IVec3;
pub type BlockPos = IVec3;
const DETAIL_LEVELS: usize = 8;

pub struct World<G: Generate> {
    center: RegionPos,
    lod_radii: SmallVec<[u32; DETAIL_LEVELS]>,
    update_level: usize,
    update_iters: SmallVec<[L1ShellIter<RegionPos>; 10]>,
    regions: HashMap<RegionPos, Region<G>>,
    active_regions: HashSet<RegionPos>,
    generator: Arc<G>,

    meshing_tx: Sender<MeshingTask>,
}

impl<G: Generate> Resource for World<G> {}

impl<G: Generate> World<G> {
    const LOD_DEPTHS: [Option<(u8, u8)>; DETAIL_LEVELS] = [
        None,
        Some((4, 5)),
        Some((3, 5)),
        Some((2, 4)),
        Some((1, 3)),
        Some((0, 2)),
        Some((0, 1)),
        Some((0, 0)),
    ];

    const MAX_UPDATE_COST: usize = 32;

    pub fn new(lod_radii: Vec<u32>, generator: G, meshing_tx: Sender<MeshingTask>) -> Self {
        assert!(!lod_radii.is_empty(), "World's lod radii must not be empty");
        assert!(
            lod_radii.len() <= DETAIL_LEVELS,
            "World's lod radii must not be longer than detail level count"
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
            active_regions: HashSet::new(),
            generator: Arc::new(generator),

            meshing_tx,
        }
    }

    #[inline]
    pub fn cast_pos(pos: BlockPos) -> (RegionPos, RelBlockPos) {
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
                let pos = (
                    rel_block_pos.as_i16vec3() + (region_pos - influenced_region_pos).as_i16vec3() * REGION_SIZE as i16 + 1
                )
                    .as_u8vec3();
                if region.set_block(pos, block) {
                    self.active_regions.insert(influenced_region_pos);
                }
            }
        }
    }

    #[inline]
    fn max_radius(&self) -> u32 {
        *self
            .lod_radii
            .last()
            .expect("World's lod radii is empty. What happened?")
    }

    #[inline]
    fn min_update_radius(&self, level: usize) -> u32 {
        if level == 0 {
            0
        } else {
            self.lod_radii[level - 1]
        }
    }

    #[inline]
    fn max_update_radius(&self, level: usize) -> u32 {
        if level == self.lod_radii.len() {
            self.max_radius() * 2 - 1
        } else {
            self.lod_radii[level]
        }
    }

    pub fn update(&mut self, center: RegionPos, threads: &WorldThreads) {
        let center_displacement = (center - self.center).abs().element_sum() as u32;
        self.center = center;
        let WorldThreads(near_threads, far_threads) = threads;

        if center_displacement > 0 {
            for i in 0..=self.lod_radii.len() {
                let min_radius = self.min_update_radius(i);
                let iter = &mut self.update_iters[i];
                let radius = (iter.radius as u32)
                    .saturating_sub(center_displacement)
                    .max(min_radius);

                *iter = L1ShellIter::new(center, radius as i32);
            }

            if center_displacement >= self.max_update_radius(self.lod_radii.len()) {
                self.regions.clear();
            }

            self.update_level = 0;
        }

        let mut cost = 0;

        while self.update_level <= self.lod_radii.len() {
            let level = self.update_level;
            let complex = (DETAIL_LEVELS - level).pow(2);
            let max_radius = self.max_update_radius(level);

            let iter = &mut self.update_iters[level];
            let context = if level < self.lod_radii.len() {
                Some(RegionContext {
                    generator: self.generator.clone(),
                    meshing_tx: self.meshing_tx.clone(),
                    depths: Self::LOD_DEPTHS[level],
                })
            } else {
                None
            };
            let threads = if level == 0 {
                near_threads
            } else {
                far_threads
            };

            while cost < Self::MAX_UPDATE_COST {
                match iter.next() {
                    Some(pos) => {
                        if let Some(ctx) = &context {
                            if let Some(region) = self.regions.get_mut(&pos) {
                                if let Some(ctx) = &context {
                                    if region.update(ctx, threads) {
                                        self.active_regions.insert(pos);
                                        cost += complex;
                                    } else {
                                        cost += 1;
                                    }
                                }
                            } else {
                                let region = Region::new(pos, ctx, threads);
                                self.regions.insert(pos, region);
                                self.active_regions.insert(pos);
                                cost += complex;
                            }
                        } else {
                            self.regions.remove(&pos);
                            cost += 1;
                        }
                    }

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

        self.active_regions.retain(|pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                let threads = if region.is_near() {
                    near_threads
                } else {
                    far_threads
                };

                !region.poll(threads)
            } else {
                false
            }
        });
    }
}

resources! {
    pub struct WorldThreads(ThreadPool, ThreadPool);
}

pub struct WorldUpdater<G: Generate>(pub PhantomData<G>);

impl<G: Generate> System for WorldUpdater<G> {
    type CompQuery = (CompRead<PlayerControlled>, CompRead<Position>);
    type ResQuery = (ResWrite<World<G>>, ResWrite<RenderedWorld>, ResRead<WorldThreads>);

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let pos = entry.2.0.floor().as_ivec3();
        let center = pos.div_euclid(IVec3::splat(REGION_SIZE as i32));
        
        res.0.update(center, res.2);
        res.1.update(center);

        None
    }
}
