use crate::actor::ActorMarker;
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::util::coord::{Axis, ICoord3};
use crate::world::block::{BlockChunk, KEEP};
use crate::world::region::*;
use crate::world::*;
use anyhow::{Context, anyhow};
use arc_swap::{ArcSwap, ArcSwapOption};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, U8Vec3, Vec3};
use log::error;
use rayon::prelude::*;
use std::sync::Arc;

mod program;
mod sample;
mod script;
mod structure;

pub use program::*;
use sample::*;
pub use script::*;
pub use structure::*;

/// The grids of the regions around the player, indexed by region position.
type RegionSamples = Volume<Arc<SampleGrid>>;

/// A [`RingVolume`](crate::world::RingVolume) whose entries can be read and
/// replaced while the worker threads are running.
///
/// The generation of a region fills in the grids of its neighbors, which is
/// why the generator keeps this volume in shared memory rather than owning it.
pub struct ArcRingVolume<T> {
    pub volume: Volume<ArcSwapOption<T>>,
    pub bound: ArcSwap<AABB<IVec3>>,
}

// SAFETY: the entries are `ArcSwapOption` cells and the bound is an `ArcSwap`,
// so every value is published atomically; the volume itself is never resized.
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

    pub fn get(&self, pos: IVec3) -> anyhow::Result<Option<Arc<T>>> {
        if self.bound.load().is_point_inside(pos) {
            Ok(self.volume.get(self.cast_pos(pos)).load_full())
        } else {
            Err(anyhow!("position is out of bounds"))
        }
    }

    pub fn set(&self, pos: IVec3, value: T) -> anyhow::Result<Option<Arc<T>>> {
        if self.bound.load().is_point_inside(pos) {
            Ok(self
                .volume
                .get(self.cast_pos(pos))
                .swap(Some(Arc::new(value))))
        } else {
            Err(anyhow!("position is out of bounds"))
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

pub struct Generator {
    context: Arc<GenContext>,
    task_rx: Receiver<GenTask>,
}

/// The shared state of the generator, which the worker threads read.
pub struct GenContext {
    /// The seed of the world, which is what the sites of the structures are
    /// drawn from.
    seed: u64,

    program: Arc<WorldProgram>,
    script: Arc<ScriptRuntime>,
    grids: ArcRingVolume<SampleGrid>,
    sites: ArcRingVolume<Vec<Vec<RelPos>>>,
}

impl Generator {
    pub fn new(
        center: RegionPos,
        world_radius: u8,
        seed: u64,
        program: Arc<WorldProgram>,
        task_rx: Receiver<GenTask>,
    ) -> anyhow::Result<Self> {
        let mut script = ScriptRuntime::new()?;
        script.prepare(&program)?;
        let radius = world_radius + 2;

        Ok(Self {
            context: Arc::new(GenContext {
                seed,
                program,
                script: Arc::new(script),
                grids: ArcRingVolume::new(center, radius),
                sites: ArcRingVolume::new(center, radius),
            }),
            task_rx,
        })
    }

    /// The near and far tasks are kept apart so that a slow coarse region cannot
    /// hold up the regions around the player.
    pub fn update(&mut self, translation: IVec3, threads: &WorldThreads) {
        self.context.update(translation);

        let WorldThreads {
            near: near_threads,
            far: far_threads,
        } = threads;

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
        let content = Self::fill(context, task.pos, task.lod);

        if let Err(e) = task.tx.try_send(GenResult {
            revision: task.revision,
            content,
        }) {
            error!(
                "failed to send generation result to region {}: {}",
                task.pos, e
            );
        }
    }

    /// Builds the blocks of the region at `pos` and the markers of the actors in
    /// it, which is the content of a generation result.
    fn fill(context: &GenContext, pos: RegionPos, lod: u8) -> anyhow::Result<GenContent> {
        let grids = context
            .samples_in_range(pos - 1, 3)
            .context("sample generation failed")?;
        let block_size = Region::block_size_on_lod(lod);
        let stride = Region::stride_on_lod(lod);
        let view = SampleGridView {
            grids,
            layout: context.program.layout.clone(),
            lod,
            base: pos - 1,
            chunk_origin: U8Vec3::splat(REGION_SIZE - block_size),
            stride,
        };
        let sites = context
            .sites_in_range(pos - 1, 2)
            .context("structure placement failed")?;

        let mut chunk = BlockChunk(Volume::splat(U8Vec3::splat(stride), 0));

        for stage in &context.program.task_stages {
            match stage {
                TaskStage::Chunk(stage) if stage.lods.runs(lod) => {
                    if let Err(error) =
                        context
                            .script
                            .run_chunk(stage, context.seed, pos, lod, &view, &mut chunk)
                    {
                        error!(
                            "chunk stage {} failed in region {pos} at LOD {lod}: {error}",
                            stage.name
                        );
                    }
                }

                TaskStage::Stamp { structure, lods } if lods.runs(lod) => {
                    context.stamp(*structure, &sites[*structure], &mut chunk, lod);
                }

                _ => (),
            }
        }

        let actors = if lod == 0 {
            sites
                .iter()
                .zip(context.program.structures.iter())
                .flat_map(|(sites, structure)| {
                    sites.iter().flat_map(move |site| {
                        structure.actors.iter().filter_map(move |marker| {
                            let pos = site.as_vec3() - REGION_SIZE as f32 + marker.pos;

                            if pos.min_element() < 0.0 || pos.max_element() >= REGION_SIZE as f32 {
                                None
                            } else {
                                Some(ActorMarker {
                                    pos,
                                    actor: marker.actor,
                                })
                            }
                        })
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        if pos == RegionPos::ZERO && lod == 0 {
            let solid = chunk.vec.iter().filter(|&&meta| meta != 0).count();
            let site_count = sites.iter().map(Vec::len).sum::<usize>();
            let per_structure = sites.iter().map(Vec::len).collect::<Vec<_>>();
            let bricks = chunk.vec.iter().filter(|&&meta| meta == 1).count();
            let logs = chunk.vec.iter().filter(|&&meta| meta == 4).count();
            let leaves = chunk.vec.iter().filter(|&&meta| meta == 5).count();
            log::debug!(
                "generated the origin region with {solid} solid blocks, {bricks} bricks, \
                 {logs} logs, {leaves} leaves, {site_count} structure sites, per structure \
                 {per_structure:?} and {} actor markers",
                actors.len()
            );
        }

        Ok(GenContent {
            chunk,
            actors,
            loaded: false,
        })
    }
}

impl GenContext {
    fn update(&self, translation: IVec3) {
        self.grids.translate(translation);
        self.sites.translate(translation);
    }

    /// A view whose positions are the region-local positions the placement
    /// stages are written in. It is not bound to a chunk, so it never asks for a
    /// halo sample.
    fn placement_view(&self, grids: RegionSamples, base: RegionPos) -> SampleGridView {
        SampleGridView {
            grids,
            layout: self.program.layout.clone(),
            lod: 0,
            base,
            chunk_origin: U8Vec3::ZERO,
            stride: 0,
        }
    }

    /// The sample grids of the `side ^ 3` regions starting at `pos`, generating
    /// the grids that are missing.
    fn samples_in_range(&self, pos: RegionPos, side: u8) -> anyhow::Result<RegionSamples> {
        let mut vec = Vec::with_capacity((side as usize).pow(3));
        let layout = &self.program.layout;
        let stride = layout.stride();
        let dims = U8Vec3::splat(GRID_STRIDE + 2);

        for dx in 0..side {
            for dy in 0..side {
                for dz in 0..side {
                    let offset = U8Vec3::new(dx, dy, dz);
                    let region = pos + offset.as_ivec3();

                    if self.grids.get(region)?.is_none() {
                        let mut grid = SampleGrid::new(dims, stride);

                        for stage in &self.program.sampler {
                            self.script
                                .run_sample(stage, self.seed, region, layout, &mut grid)?;
                        }

                        compute_gradients(layout, &mut grid, stride);

                        // Drop the halo, which was only needed for the gradient.
                        grid = grid.part(U8Vec3::ONE, U8Vec3::splat(GRID_STRIDE), stride);
                        self.grids.set(region, grid)?;
                    }

                    vec.push(
                        self.grids
                            .get(region)?
                            .expect("grid should have been generated above"),
                    );
                }
            }
        }

        Ok(Volume {
            size: U8Vec3::splat(side),
            vec,
        })
    }

    /// The candidates of one region, before the candidates that cross a region
    /// border are resolved against the neighbors.
    fn site_candidates(&self, region: RegionPos) -> anyhow::Result<Vec<Vec<RelPos>>> {
        let view = self.placement_view(self.samples_in_range(region, 2)?, region);
        let mut candidates = Vec::with_capacity(self.program.structures.len());

        for (index, structure) in self.program.structures.iter().enumerate() {
            let mut sites = Vec::new();
            let mut bounds = Vec::new();

            if let Some(place) = &self.program.placements[index] {
                let placed = self
                    .script
                    .run_place(place, self.seed, region, structure, &view)?;

                for candidate in placed {
                    if sites.len() >= structure.max_sites as usize {
                        break;
                    }

                    let site = U8Vec3::from(candidate);
                    let bound = structure.bound.translate(site);

                    if bounds.iter().all(|&other| !bound.intersects_with(other)) {
                        sites.push(site);
                        bounds.push(bound);
                    }
                }
            }

            candidates.push(sites);
        }

        Ok(candidates)
    }

    /// The sites of every structure in the `side ^ 3` regions starting at `pos`,
    /// as offsets from the block position of the first region.
    fn sites_in_range(&self, pos: RegionPos, side: u8) -> anyhow::Result<Vec<Vec<RelPos>>> {
        let mut sites = vec![Vec::new(); self.program.structures.len()];

        for dx in 0..side {
            for dy in 0..side {
                for dz in 0..side {
                    let offset = U8Vec3::new(dx, dy, dz);
                    let region = pos + offset.as_ivec3();

                    if self.sites.get(region)?.is_none() {
                        let mut adjacent_bounds = Vec::new();

                        for offset in U8Vec3::corners(U8Vec3::ZERO, U8Vec3::ONE)
                            .into_iter()
                            .skip(1)
                        {
                            let neighbor = region + offset.as_ivec3();
                            let candidates = self.site_candidates(neighbor)?;

                            for (structure_sites, structure) in
                                candidates.into_iter().zip(self.program.structures.iter())
                            {
                                for site in structure_sites {
                                    adjacent_bounds.push(
                                        structure.bound.translate(offset * REGION_SIZE + site),
                                    );
                                }
                            }
                        }

                        let mut candidates = self.site_candidates(region)?;

                        candidates
                            .iter_mut()
                            .zip(self.program.structures.iter())
                            .for_each(|(structure_sites, structure)| {
                                structure_sites.retain(|&site| {
                                    (site + structure.bound.max).max_element() <= REGION_SIZE || {
                                        let bound = structure.bound.translate(site);
                                        adjacent_bounds
                                            .iter()
                                            .all(|&adjacent| !bound.intersects_with(adjacent))
                                    }
                                })
                            });

                        self.sites.set(region, candidates)?;
                    }

                    sites
                        .iter_mut()
                        .zip(
                            self.sites
                                .get(region)?
                                .expect("sites should have been generated above")
                                .iter(),
                        )
                        .for_each(|(structure_sites, region_sites)| {
                            for &site in region_sites.iter() {
                                structure_sites.push(site + offset * REGION_SIZE);
                            }
                        });
                }
            }
        }

        Ok(sites)
    }

    /// Stamps the content of one structure into the chunk of one region.
    fn stamp(&self, structure: usize, sites: &[RelPos], chunk: &mut BlockChunk, lod: u8) {
        let structure = &self.program.structures[structure];
        let Some(blocks) = structure.blocks(lod) else {
            return;
        };

        let s = U8Vec3::splat(REGION_SIZE >> lod);
        let bound = if lod == 0 {
            AABB {
                min: s - 1,
                max: 2 * s + 1,
            }
        } else {
            AABB { min: s, max: 2 * s }
        };
        let origin = s - 1;
        let range = blocks.bound();

        for &site in sites {
            let range = range.translate(site >> lod);

            if let Some(intersection) = bound.intersection(range) {
                chunk
                    .iter_mut(intersection.min - origin, intersection.max - origin)
                    .zip(blocks.iter(intersection.min - range.min, intersection.max - range.min))
                    .for_each(|(chunk_meta, &struct_meta)| {
                        if struct_meta != KEEP {
                            *chunk_meta = struct_meta;
                        }
                    });
            }
        }
    }
}

fn compute_gradients(layout: &SampleLayout, grid: &mut SampleGrid, stride: usize) {
    for &(derived, source) in layout.gradients() {
        for x in 1..=GRID_STRIDE {
            for y in 1..=GRID_STRIDE {
                for z in 1..=GRID_STRIDE {
                    let pos = U8Vec3::new(x, y, z);
                    let mut value = Vec3::ZERO;

                    for &axis in Axis::ALL {
                        value = value
                            .shift(
                                axis,
                                grid.values(pos.step(axis.direction(true)), stride)[source.offset],
                            )
                            .shift(
                                axis,
                                -grid.values(pos.step(axis.direction(false)), stride)
                                    [source.offset],
                            );
                    }

                    value /= (SAMPLE_INTERVAL * 2) as f32;
                    let values = grid.values_mut(pos, stride);
                    values[derived.offset..derived.offset + 3].copy_from_slice(&value.to_array());
                }
            }
        }
    }
}

pub struct GenTask {
    pub pos: RegionPos,
    pub lod: u8,
    /// The generation of the region the task belongs to, which the result
    /// reports back so that the region can drop the results of the requests it
    /// has already superseded.
    pub revision: u64,
    pub tx: Sender<GenResult>,
}

pub struct GenResult {
    pub revision: u64,
    /// The content of the region, or why it could not be built. A region that
    /// gets an error asks for the region again.
    pub content: anyhow::Result<GenContent>,
}

/// What one generation of a region produced.
pub struct GenContent {
    pub chunk: BlockChunk,
    pub actors: Vec<ActorMarker>,
    /// Whether the chunk came from a save instead of the generator.
    pub loaded: bool,
}
