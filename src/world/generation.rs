use crate::ecs::Resource;
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::util::coord::{Axis, Coord3, ICoord3};
use crate::world::block::Meta;
use crate::world::region::*;
use crate::world::{BlockPos, Chunk, RegionPos, WorldThreads, LOD_COUNT};
use anyhow::{anyhow, Result};
use arc_swap::{ArcSwap, ArcSwapOption};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, U8Vec3, Vec3};
use log::error;
use rand::RngExt;
use rand_pcg::Pcg64Mcg;
use rand_seeder::Seeder;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::sync::Arc;

type SampleGrid = Volume<Sample>;
type SampleGridView = Volume<Arc<SampleGrid>>;
const SAMPLE_INTERVAL: u8 = 8;
const GRID_SIZE: u8 = REGION_SIZE / SAMPLE_INTERVAL;
const GRID_STRIDE: u8 = GRID_SIZE + 1;

impl SampleGridView {
    fn interpolate(&self, pos: LocalPos, lod: u8, halo: bool) -> Sample {
        let origin = pos / SAMPLE_INTERVAL;
        let size = Region::block_size_on_lod(lod);
        let offset = (size / SAMPLE_INTERVAL).max(1);
        let corners = U8Vec3::corners(U8Vec3::ZERO, U8Vec3::ONE);
        let samples = corners.map(|corner| {
            let pos = origin + corner * offset;
            let grid_pos = (pos / GRID_SIZE).min(self.size - 1);
            let pos = pos - grid_pos * GRID_SIZE;
            *self.get(grid_pos).get(pos)
        });

        let redirection = if halo {
            let mut min = 1.0;
            let mut min_corner = U8Vec3::ZERO;
            for (&corner, sample) in corners.iter().zip(samples) {
                let d = sample.density.min(-sample.erosion);
                if d < min {
                    min = d;
                    min_corner = corner;
                }
            }
            min_corner * (size - 1)
        } else {
            U8Vec3::splat(size / 2)
        };
        let rate =
            (pos % SAMPLE_INTERVAL + redirection).as_vec3() / (offset * SAMPLE_INTERVAL) as f32;

        let mut value = Sample::default();

        for (&corner, sample) in corners.iter().zip(samples) {
            let weight = (rate - (1 - corner).as_vec3()).abs();
            value += sample * weight.element_product();
        }

        value
    }
}

pub struct ArcRingVolume<T> {
    pub volume: Volume<ArcSwapOption<T>>,
    pub bound: ArcSwap<AABB<IVec3>>,
}

unsafe impl<T> Sync for ArcRingVolume<T> {}

impl<T> ArcRingVolume<T> {
    pub fn new(center: IVec3, radius: u8) -> Self {
        Self {
            volume: Volume::from_fn(U8Vec3::splat(radius * 2 + 1), |_| ArcSwapOption::empty()),
            bound: ArcSwap::new(Arc::new(AABB {
                min: center - radius as i32,
                max: center + radius as i32 + 1,
            })),
        }
    }

    #[inline]
    fn cast_pos(&self, pos: IVec3) -> U8Vec3 {
        pos.rem_euclid(self.volume.size.as_ivec3()).as_u8vec3()
    }

    pub fn get(&self, pos: IVec3) -> Result<Option<Arc<T>>> {
        if self.bound.load().is_point_inside(pos) {
            Ok(self.volume.get(self.cast_pos(pos)).load_full())
        } else {
            Err(anyhow!("position out of bound"))
        }
    }

    pub fn set(&self, pos: IVec3, value: T) -> Result<Option<Arc<T>>> {
        if self.bound.load().is_point_inside(pos) {
            Ok(self
                .volume
                .get(self.cast_pos(pos))
                .swap(Some(Arc::new(value))))
        } else {
            Err(anyhow!("position out of bound"))
        }
    }

    pub fn translate(&self, dpos: IVec3) {
        let bound = self.bound.load();
        let new_bound = bound.translate(dpos);

        for &axis in Axis::ALL {
            let d = dpos.get(axis);
            if d != 0 {
                let (min, max) = match d > 0 {
                    false => (new_bound.max.get(axis), bound.max.get(axis)),
                    true => (bound.min.get(axis), new_bound.min.get(axis)),
                };

                let size = self.volume.size.get(axis) as i32;
                let min = min.rem_euclid(size) as u8;
                let max = max.rem_euclid(size) as u8;

                if min < max {
                    self.volume
                        .iter(
                            U8Vec3::ZERO.with(axis, min),
                            self.volume.size.with(axis, max),
                        )
                        .for_each(|item| {
                            item.swap(None);
                        });
                } else {
                    self.volume
                        .iter(U8Vec3::ZERO, self.volume.size.with(axis, max))
                        .for_each(|item| {
                            item.swap(None);
                        });
                    self.volume
                        .iter(U8Vec3::ZERO.with(axis, min), self.volume.size)
                        .for_each(|item| {
                            item.swap(None);
                        });
                }
            }
        }

        self.bound.swap(Arc::new(new_bound));
    }
}

pub struct Field {
    pub climate: fn(BlockPos) -> Vec3,
    pub density: fn(BlockPos) -> f32,
    pub erosion: fn(BlockPos) -> f32,
}

#[derive(Clone, Copy, Default)]
pub struct Sample {
    pub climate: Vec3,
    pub density: f32,
    pub gradient: Vec3,
    pub erosion: f32,
}

impl Add for Sample {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            climate: self.climate + rhs.climate,
            density: self.density + rhs.density,
            gradient: self.gradient + rhs.gradient,
            erosion: self.erosion + rhs.erosion,
        }
    }
}

impl AddAssign for Sample {
    fn add_assign(&mut self, rhs: Self) {
        self.climate += rhs.climate;
        self.density += rhs.density;
        self.gradient += rhs.gradient;
        self.erosion += rhs.erosion;
    }
}

impl Mul<f32> for Sample {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            climate: self.climate * rhs,
            density: self.density * rhs,
            gradient: self.gradient * rhs,
            erosion: self.erosion * rhs,
        }
    }
}

impl MulAssign<f32> for Sample {
    fn mul_assign(&mut self, rhs: f32) {
        self.climate *= rhs;
        self.density *= rhs;
        self.gradient *= rhs;
        self.erosion *= rhs;
    }
}

pub struct Structure {
    pub blocks: SmallVec<[Volume<Option<Meta>>; LOD_COUNT]>,
    pub condition: fn(&SampleGridView, LocalPos) -> bool,
    pub count: u16,
}

impl Structure {
    pub fn tree() -> Self {
        let mut blocks0 = Volume::new(U8Vec3::new(5, 8, 5));
        blocks0.fill(U8Vec3::new(0, 4, 1), U8Vec3::new(5, 7, 4), Some(5));
        blocks0.fill(U8Vec3::new(1, 3, 1), U8Vec3::new(4, 8, 4), Some(5));
        blocks0.fill(U8Vec3::new(1, 4, 0), U8Vec3::new(4, 7, 5), Some(5));
        blocks0.fill(U8Vec3::new(2, 0, 2), U8Vec3::new(3, 7, 3), Some(4));
        let mut blocks1 = Volume::new(U8Vec3::new(3, 4, 3));
        blocks1.fill(U8Vec3::new(0, 1, 0), U8Vec3::new(3, 4, 3), Some(5));
        blocks1.fill(U8Vec3::new(1, 0, 1), U8Vec3::new(2, 3, 2), Some(4));
        let mut blocks2 = Volume::splat(U8Vec3::new(1, 2, 1), Some(5));
        blocks2.set(U8Vec3::ZERO, Some(4));
        let blocks3 = Volume::splat(U8Vec3::new(1, 1, 1), Some(5));

        Self {
            blocks: smallvec![blocks0, blocks1, blocks2, blocks3,],
            condition: |view, pos| -> bool {
                let base = pos + U8Vec3::new(2, 0, 2);

                let root = view.interpolate(base, 0, false);
                if !(root.density > 0.0 && root.erosion < 0.0 && root.density < 0.125) {
                    return false;
                }

                for dy in 3..6 {
                    let trunk = view.interpolate(base.shift(Axis::Y, dy), 0, false);
                    if trunk.density > 0.0 && trunk.erosion < 0.0 {
                        return false;
                    }
                }

                true
            },
            count: 64,
        }
    }
}

pub struct Generator {
    context: Arc<GenContext>,
    task_rx: Receiver<GenTask>,
}

pub struct GenContext {
    field: Field,
    grids: ArcRingVolume<SampleGrid>,
    sites: ArcRingVolume<Vec<Vec<LocalPos>>>,

    terrain: fn(Sample) -> Meta,
    structures: Vec<Structure>,
    after: fn(Meta, &Chunk, LocalPos) -> Meta,
}

impl Generator {
    pub fn new(
        center: RegionPos,
        radius: u8,
        field: Field,
        terrain: fn(Sample) -> Meta,
        structures: Vec<Structure>,
        after: fn(Meta, &Chunk, LocalPos) -> Meta,
        task_rx: Receiver<GenTask>,
    ) -> Self {
        Self {
            context: Arc::new(GenContext {
                field,
                grids: ArcRingVolume::new(center, radius),
                sites: ArcRingVolume::new(center, radius),

                terrain,
                structures,
                after,
            }),
            task_rx,
        }
    }

    pub fn update(&mut self, translation: IVec3, threads: &WorldThreads) {
        self.context.update(translation);

        let WorldThreads(near_threads, far_threads) = threads;

        let mut near_tasks = Vec::new();
        let mut far_tasks = Vec::new();
        for task in self.task_rx.try_iter() {
            match task.lod {
                0 => near_tasks.push(task),
                1.. => far_tasks.push(task),
            }
        }

        let context = self.context.clone();
        near_threads.spawn(move || {
            near_tasks.into_par_iter().for_each(|task| {
                Self::perform(&context, task);
            })
        });

        let context = self.context.clone();
        far_threads.spawn(move || {
            far_tasks.into_par_iter().for_each(|task| {
                Self::perform(&context, task);
            })
        });
    }

    fn perform(context: &GenContext, task: GenTask) {
        let Ok(view) = context.samples_in_range(task.pos - 1, 3) else {
            return;
        };
        let origin = REGION_SIZE - Region::block_size_on_lod(task.lod);
        let stride = Region::stride_on_lod(task.lod);

        let mut chunk = Chunk::from_fn(U8Vec3::splat(stride), |pos| {
            let sample = view.interpolate(
                origin + pos * Region::block_size_on_lod(task.lod),
                task.lod,
                pos.min_element() == 0 || pos.max_element() == stride - 1,
            );
            (context.terrain)(sample)
        });

        let Ok(sites) = context.sites_in_range(task.pos - 1, 2) else {
            return;
        };
        let s = U8Vec3::splat(REGION_SIZE >> task.lod);
        let bound = if task.lod == 0 {
            AABB {
                min: s - 1,
                max: 2 * s + 1,
            }
        } else {
            AABB { min: s, max: 2 * s }
        };
        let origin = s - 1;

        for (struct_sites, structure) in sites.iter().zip(context.structures.iter()) {
            if let Some(blocks) = structure.blocks.get(task.lod as usize) {
                let struct_bound = AABB {
                    min: U8Vec3::ZERO,
                    max: blocks.size,
                };

                for &site in struct_sites.iter() {
                    let struct_bound = struct_bound.translate(site >> task.lod);

                    if let Some(intersection) = bound.intersection(struct_bound) {
                        chunk
                            .iter_mut(intersection.min - origin, intersection.max - origin)
                            .zip(blocks.iter(
                                intersection.min - struct_bound.min,
                                intersection.max - struct_bound.min,
                            ))
                            .for_each(|(chunk_meta, &struct_meta)| {
                                if let Some(meta) = struct_meta {
                                    *chunk_meta = meta;
                                }
                            });
                    }
                }
            }
        }

        for x in 1..stride - 1 {
            for y in 1..stride - 1 {
                for z in 1..stride - 1 {
                    let pos = LocalPos::new(x, y, z);
                    let meta = (context.after)(*chunk.get(pos), &chunk, pos);
                    chunk.set(pos, meta);
                }
            }
        }

        let result = GenResult {
            lod: task.lod,
            chunk,
        };

        if let Err(e) = task.tx.try_send(result) {
            error!("failed to send generation result to region: {}", e);
        }
    }
}

impl GenContext {
    fn update(&self, translation: IVec3) {
        self.grids.translate(translation);
        self.sites.translate(translation);
    }

    fn samples_in_range(&self, pos: RegionPos, side: u8) -> Result<SampleGridView> {
        let mut vec = Vec::with_capacity((side as usize).pow(3));

        for dx in 0..side {
            for dy in 0..side {
                for dz in 0..side {
                    let offset = U8Vec3::new(dx, dy, dz);
                    let pos = pos + offset.as_ivec3();

                    if self.grids.get(pos)?.is_none() {
                        let origin = pos * REGION_SIZE as i32;

                        let mut grid = Volume::from_fn(U8Vec3::splat(GRID_STRIDE + 2), |pos| {
                            let pos = origin + (pos.as_ivec3() - 1) * SAMPLE_INTERVAL as i32;
                            Sample {
                                climate: (self.field.climate)(pos),
                                density: (self.field.density)(pos),
                                gradient: Vec3::ZERO,
                                erosion: (self.field.erosion)(pos),
                            }
                        });

                        for x in 1..=GRID_STRIDE {
                            for y in 1..=GRID_STRIDE {
                                for z in 1..=GRID_STRIDE {
                                    let pos = U8Vec3::new(x, y, z);
                                    let mut gradient = Vec3::ZERO;
                                    for &axis in Axis::ALL {
                                        gradient = gradient
                                            .shift(
                                                axis,
                                                grid.get(pos.step(axis.direction(true))).density,
                                            )
                                            .shift(
                                                axis,
                                                -grid.get(pos.step(axis.direction(false))).density,
                                            );
                                    }
                                    gradient /= (SAMPLE_INTERVAL * 2) as f32;
                                    grid.get_mut(pos).gradient = gradient;
                                }
                            }
                        }

                        grid = grid.part(U8Vec3::ONE, U8Vec3::splat(GRID_STRIDE));
                        self.grids.set(pos, grid)?;
                    }

                    vec.push(self.grids.get(pos)?.unwrap());
                }
            }
        }

        Ok(SampleGridView {
            size: U8Vec3::splat(side),
            vec,
        })
    }

    fn sites_in_range(&self, pos: RegionPos, side: u8) -> Result<Vec<Vec<LocalPos>>> {
        let mut sites = vec![Vec::new(); self.structures.len()];

        for dx in 0..side {
            for dy in 0..side {
                for dz in 0..side {
                    let offset = U8Vec3::new(dx, dy, dz);
                    let pos = pos + offset.as_ivec3();

                    if self.sites.get(pos)?.is_none() {
                        let grid = self.samples_in_range(pos, 2)?;

                        let sites = self
                            .structures
                            .iter()
                            .enumerate()
                            .map(|(i, structure)| {
                                let mut struct_sites = Vec::new();
                                let mut rand = Seeder::from((pos, i)).into_rng::<Pcg64Mcg>();
                                for _ in 0..structure.count {
                                    let pos = rand.random::<LocalPos>() % REGION_SIZE;
                                    if (structure.condition)(&grid, pos) {
                                        struct_sites.push(pos);
                                    }
                                }
                                struct_sites
                            })
                            .collect::<Vec<_>>();

                        self.sites.set(pos, sites)?;
                    }

                    sites
                        .iter_mut()
                        .zip(self.sites.get(pos)?.unwrap().iter())
                        .for_each(|(struct_sites, region_sites)| {
                            for site in region_sites.iter() {
                                struct_sites.push(site + offset * REGION_SIZE);
                            }
                        });
                }
            }
        }

        Ok(sites)
    }
}

pub struct GenTask {
    pub pos: RegionPos,
    pub lod: u8,
    pub tx: Sender<GenResult>,
}

pub struct GenResult {
    pub lod: u8,
    pub chunk: Chunk,
}

impl Resource for Generator {}
