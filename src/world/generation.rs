use crate::ecs::Resource;
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::util::coord::{Axis, Coord3, ICoord3};
use crate::world::block::Meta;
use crate::world::region::*;
use crate::world::{BlockPos, Chunk, RegionPos, RingVolume, WorldThreads, LOD_COUNT};
use anyhow::{anyhow, Result};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, U8Vec3, Vec3};
use log::error;
use rand::RngExt;
use rand_pcg::Pcg64Mcg;
use rand_seeder::Seeder;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::sync::{Arc, RwLock};

pub type Grid = Volume<Sample>;
pub const SAMPLE_INTERVAL: u8 = 8;
pub const UNIT_GRID_SIZE: u8 = REGION_SIZE / SAMPLE_INTERVAL + 1;

pub struct Field {
    pub temperature: fn(BlockPos) -> f32,
    pub ventilation: fn(BlockPos) -> f32,
    pub humidity: fn(BlockPos) -> f32,
    pub density: fn(BlockPos) -> f32,
}

#[derive(Clone, Copy, Default)]
pub struct Sample {
    pub temperature: f32,
    pub ventilation: f32,
    pub humidity: f32,
    pub density: f32,
    pub gradient: Vec3,
}

impl Add for Sample {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            temperature: self.temperature + rhs.temperature,
            ventilation: self.ventilation + rhs.ventilation,
            humidity: self.humidity + rhs.humidity,
            density: self.density + rhs.density,
            gradient: self.gradient + rhs.gradient,
        }
    }
}

impl AddAssign for Sample {
    fn add_assign(&mut self, rhs: Self) {
        self.temperature += rhs.temperature;
        self.ventilation += rhs.ventilation;
        self.humidity += rhs.humidity;
        self.density += rhs.density;
        self.gradient += rhs.gradient;
    }
}

impl Mul<f32> for Sample {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            temperature: self.temperature * rhs,
            ventilation: self.ventilation * rhs,
            humidity: self.humidity * rhs,
            density: self.density * rhs,
            gradient: self.gradient * rhs,
        }
    }
}

impl MulAssign<f32> for Sample {
    fn mul_assign(&mut self, rhs: f32) {
        self.temperature *= rhs;
        self.ventilation *= rhs;
        self.humidity *= rhs;
        self.density *= rhs;
        self.gradient *= rhs;
    }
}

pub struct Structure {
    pub blocks: SmallVec<[Volume<Option<Meta>>; LOD_COUNT]>,
    pub condition: fn(&Grid, LocalPos) -> bool,
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
            condition: |grid, pos| -> bool {
                let base = pos + U8Vec3::new(2, 0, 2);

                let root = grid.sample(base, 0);
                if !(root.density > 0.0 && root.ventilation < 0.0 && root.density < 0.125) {
                    return false;
                }

                for dy in 3..6 {
                    let trunk = grid.sample(base.shift(Axis::Y, dy), 0);
                    if trunk.density > 0.0 && trunk.ventilation < 0.0 {
                        return false;
                    }
                }

                true
            },
            count: 64,
        }
    }
}

impl Grid {
    fn sample(&self, pos: LocalPos, lod: u8) -> Sample {
        let origin = pos / SAMPLE_INTERVAL;
        let size = Region::block_size_on_lod(lod);
        let offset = (size / SAMPLE_INTERVAL).max(1);
        let corners = U8Vec3::corners(U8Vec3::ZERO, U8Vec3::ONE);

        let pos = if size == 1 {
            pos
        } else {
            let mut min_density = 1.0;
            let mut min_corner = U8Vec3::ZERO;
            for corner in corners {
                let density = self.get(origin + corner * offset).density;
                if density < min_density {
                    min_density = density;
                    min_corner = corner;
                }
            }
            pos + min_corner * (size - 1)
        };
        let interval = offset * SAMPLE_INTERVAL;
        let rate = (pos % interval).as_vec3() / interval as f32;

        let mut sample = Sample {
            temperature: 0.0,
            ventilation: 0.0,
            humidity: 0.0,
            density: 0.0,
            gradient: Vec3::ZERO,
        };

        for corner in corners {
            let offset = corner * offset;
            let weight = (rate - (1 - corner).as_vec3()).abs();

            sample += *self.get(origin + offset) * weight.element_product();
        }

        sample
    }
}

impl Chunk {
    fn place(&mut self, pos: LocalPos, blocks: Volume<Option<Meta>>) {
        let max = pos + blocks.size;
        assert!(max.x <= self.size.x && max.y <= self.size.y && max.z <= self.size.z);

        let [w, h, _] = self.size.as_usizevec3().to_array();
        let [sw, sh, sd] = blocks.size.as_usizevec3().to_array();
        let [dx, dy, dz] = pos.as_usizevec3().to_array();

        for z in 0..sd {
            for y in 0..sh {
                let src = z * sh * sw + y * sw;
                let dst = (z + dz) * h * w + (y + dy) * w + dx;
                for (dst, src) in self.vec[dst..dst + sw]
                    .iter_mut()
                    .zip(blocks.vec[src..src + sw].iter())
                {
                    if let Some(meta) = src {
                        *dst = *meta;
                    }
                }
            }
        }
    }
}

pub struct Generator {
    context: Arc<GenContext>,
    task_rx: Receiver<GenTask>,
}

pub struct GenContext {
    field: Field,
    grids: RwLock<RingVolume<Grid>>,
    sites: RwLock<RingVolume<Vec<Vec<LocalPos>>>>,

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
                grids: RwLock::new(RingVolume::new(center, radius)),
                sites: RwLock::new(RingVolume::new(center, radius)),

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
        let Ok(grid) = context.samples(task.pos - 1, 3) else {
            return;
        };
        let origin = REGION_SIZE - Region::block_size_on_lod(task.lod);
        let size = Region::chunk_size_on_lod(task.lod);

        let mut chunk = Chunk::from_fn(U8Vec3::splat(size), |pos| {
            let sample = grid.sample(origin + pos * Region::block_size_on_lod(task.lod), task.lod);
            (context.terrain)(sample)
        });

        let Ok(sites) = context.sites(task.pos - 1, 2) else {
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
                        let pos = intersection.min - origin;
                        let min = intersection.min - struct_bound.min;
                        let size = intersection.max - intersection.min;
                        chunk.place(pos, blocks.part(min, size));
                    }
                }
            }
        }

        for x in 1..size - 1 {
            for y in 1..size - 1 {
                for z in 1..size - 1 {
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
            error!("Failed to send generation result to Region: {}", e);
        }
    }
}

impl GenContext {
    fn update(&self, translation: IVec3) {
        self.grids.write().unwrap().translate(translation);
        self.sites.write().unwrap().translate(translation);
    }

    fn samples(&self, pos: RegionPos, side: u8) -> Result<Grid> {
        let mut area_grid = Grid::new(U8Vec3::splat((UNIT_GRID_SIZE - 1) * side + 1));

        for dx in 0..side {
            for dy in 0..side {
                for dz in 0..side {
                    let offset = U8Vec3::new(dx, dy, dz);
                    let pos = pos + offset.as_ivec3();

                    {
                        let grids = self.grids.read().unwrap();
                        if !grids.bound.is_point_inside(pos) {
                            return Err(anyhow!("Region pos out of bound"));
                        }
                        if let Some(region_grid) = grids.get(pos) {
                            area_grid.fit(offset * (UNIT_GRID_SIZE - 1), region_grid);
                            continue;
                        }
                    }

                    {
                        let mut grids = self.grids.write().unwrap();
                        let origin = pos * REGION_SIZE as i32;
                        let mut region_grid =
                            Volume::from_fn(U8Vec3::splat(UNIT_GRID_SIZE + 2), |pos| {
                                let pos = origin + (pos.as_ivec3() - 1) * SAMPLE_INTERVAL as i32;
                                Sample {
                                    temperature: (self.field.temperature)(pos),
                                    ventilation: (self.field.ventilation)(pos),
                                    humidity: (self.field.humidity)(pos),
                                    density: (self.field.density)(pos),
                                    gradient: Vec3::ZERO,
                                }
                            });
                        for x in 1..UNIT_GRID_SIZE + 1 {
                            for y in 1..UNIT_GRID_SIZE + 1 {
                                for z in 1..UNIT_GRID_SIZE + 1 {
                                    let pos = U8Vec3::new(x, y, z);
                                    let mut gradient = Vec3::ZERO;
                                    for &axis in Axis::ALL {
                                        gradient = gradient
                                            .shift(
                                                axis,
                                                region_grid
                                                    .get(pos.step(axis.direction(true)))
                                                    .density,
                                            )
                                            .shift(
                                                axis,
                                                -region_grid
                                                    .get(pos.step(axis.direction(false)))
                                                    .density,
                                            );
                                    }
                                    gradient /= (SAMPLE_INTERVAL * 2) as f32;
                                    region_grid.get_mut(pos).gradient = gradient;
                                }
                            }
                        }
                        region_grid = region_grid.part(U8Vec3::ONE, U8Vec3::splat(UNIT_GRID_SIZE));
                        area_grid.fit(offset * (UNIT_GRID_SIZE - 1), &region_grid);
                        grids.set(pos, region_grid);
                    }
                }
            }
        }

        Ok(area_grid)
    }

    fn sites(&self, pos: RegionPos, side: u8) -> Result<Vec<Vec<LocalPos>>> {
        let mut area_sites = vec![Vec::new(); self.structures.len()];

        for dx in 0..side {
            for dy in 0..side {
                for dz in 0..side {
                    let offset = U8Vec3::new(dx, dy, dz);
                    let pos = pos + offset.as_ivec3();

                    {
                        let sites = self.sites.read().unwrap();
                        if !sites.bound.is_point_inside(pos) {
                            return Err(anyhow!("Region pos out of bound"));
                        }
                        if let Some(region_sites) = sites.get(pos) {
                            area_sites.iter_mut().zip(region_sites.iter()).for_each(
                                |(struct_sites, region_sites)| {
                                    for site in region_sites.iter() {
                                        struct_sites.push(site + offset * REGION_SIZE);
                                    }
                                },
                            );
                            continue;
                        }
                    }

                    {
                        let mut sites = self.sites.write().unwrap();
                        let grid = self.samples(pos, 2)?;
                        let region_sites = self
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
                        area_sites.iter_mut().zip(region_sites.iter()).for_each(
                            |(struct_sites, region_sites)| {
                                for site in region_sites.iter() {
                                    struct_sites.push(site + offset * REGION_SIZE);
                                }
                            },
                        );
                        sites.set(pos, region_sites);
                    }
                }
            }
        }

        Ok(area_sites)
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
