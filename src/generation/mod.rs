//! The world generator: the fields it samples, the blocks their samples turn
//! into, and the structures it places.
//!
//! A region is generated in three phases: the fields of the region and its
//! neighbors are sampled, the terrain rule turns their samples into the blocks of
//! the chunk, and every structure the generator places stamps its sites into it.

use crate::actor::ActorMarker;
use crate::game::ModRegistry;
use crate::resources;
use crate::util::Id;
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::util::coord::{Axis, Coord3};
use crate::world::RegionPos;
use crate::world::WorldThreads;
use crate::world::block::{BlockChunk, KEEP};
use crate::world::region::{REGION_SIZE, Region, RelPos};
use anyhow::anyhow;
use arc_swap::{ArcSwap, ArcSwapOption};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, U8Vec3};
use log::error;
use rand_pcg::Pcg64Mcg;
use rand_pcg::rand_core::Rng;
use rand_seeder::Seeder;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::Path;
use std::sync::Arc;

mod field;
mod program;
mod rule;
mod structure;

use field::*;
pub use program::*;
pub use structure::*;

/// The fields of the regions around the player, one volume per field.
type RegionGrids = Vec<Volume<Arc<FieldGrid>>>;

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

resources! {
    /// The generator that shapes the world, named by its file of
    /// `assets/generators`.
    ///
    /// Only the name is data: a world type names the generator a new world uses
    /// and a save keeps it, while the program and the worker state
    /// are built from the name when the world starts.
    #[derive(Clone, Default)]
    pub struct Generator {
        name: String,
        seed: u64,
        runtime: Option<GeneratorRuntime>,
    }: asset, save;
}
/// The parts of a [`Generator`] that only exist while the game runs.
#[derive(Clone)]
struct GeneratorRuntime {
    context: Arc<GenContext>,
    task_rx: Receiver<GenTask>,
}

impl Serialize for Generator {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.name.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Generator {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            name: String::deserialize(deserializer)?,
            seed: 0,
            runtime: None,
        })
    }
}

/// The directory of the generators inside a mod.
const GENERATOR_DIR: &str = "assets/generators";

/// What a generation of the world hangs on.
pub struct GenContext {
    seed: u64,
    program: Arc<WorldProgram>,
    /// One ring of computed fields per field of the program.
    grids: Vec<ArcRingVolume<FieldGrid>>,
    /// The sites each region proposes, which no other region changes.
    candidates: ArcRingVolume<Vec<Vec<Site>>>,
    /// The sites each region keeps, which is what it stamps.
    sites: ArcRingVolume<Vec<Vec<Site>>>,
}

/// One structure the generator places: where its content starts, and which of
/// its variants it uses.
#[derive(Clone, Copy, Debug)]
pub struct Site {
    /// The content origin of the structure, in the blocks of the region it stands
    /// in.
    pub pos: RelPos,
    pub variant: u8,
}

impl Generator {
    /// The generator `name`, which is all a world type or a save holds of it.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            seed: 0,
            runtime: None,
        }
    }

    /// The name of the generator, which is the file it is read from.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The seed the world is generated with, which the session is saved with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Builds the program and the worker state of the generator.
    pub fn prepare(
        &mut self,
        mods: &ModRegistry,
        center: RegionPos,
        world_radius: u8,
        seed: u64,
        task_rx: Receiver<GenTask>,
    ) -> anyhow::Result<()> {
        let asset = mods.asset(&Path::new(GENERATOR_DIR).join(format!("{}.ron", self.name)))?;
        let program = Arc::new(WorldProgram::load(&asset, mods)?);
        let radius = world_radius + 2;
        let mut grids = Vec::with_capacity(program.graphs.len());

        for _ in &program.graphs {
            grids.push(ArcRingVolume::new(center, radius));
        }

        self.seed = seed;
        self.runtime = Some(GeneratorRuntime {
            context: Arc::new(GenContext {
                seed,
                program,
                grids,
                candidates: ArcRingVolume::new(center, radius),
                sites: ArcRingVolume::new(center, radius),
            }),
            task_rx,
        });

        Ok(())
    }

    /// The near and far tasks are kept apart so that a slow coarse region cannot
    /// hold up the regions around the player.
    pub fn update(&mut self, translation: IVec3, threads: &WorldThreads) {
        let Some(runtime) = &mut self.runtime else {
            return;
        };

        runtime.context.update(translation);

        let WorldThreads {
            near: near_threads,
            far: far_threads,
        } = threads;

        let mut near_tasks = Vec::new();
        let mut far_tasks = Vec::new();
        for task in runtime.task_rx.try_iter() {
            match task.lod {
                0 => near_tasks.push(task),
                1.. => far_tasks.push(task),
            }
        }

        let context = runtime.context.clone();
        near_threads.spawn(move || {
            near_tasks.into_par_iter().for_each(|task| {
                Self::perform(&context, task);
            })
        });

        let context = runtime.context.clone();
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
        let fields = context.grids_in_range(pos - 1, 3)?;
        let view = FieldView {
            graphs: &context.program.graphs,
            regions: &fields,
            base: pos - 1,
        };
        let stride = Region::stride_on_lod(lod);
        let edge = stride as usize;
        let block = 1i32 << lod;
        let origin = pos * REGION_SIZE as i32;
        let mut chunk = BlockChunk(Volume::splat(U8Vec3::splat(stride), 0));
        let mut scratch = vec![0.0f32; context.program.terrain.samples()];

        for x in 0..stride as usize {
            for y in 0..stride as usize {
                for z in 0..stride as usize {
                    let at = origin + (IVec3::new(x as i32, y as i32, z as i32) - 1) * block;
                    let halo = x == 0
                        || y == 0
                        || z == 0
                        || x == stride as usize - 1
                        || y == stride as usize - 1
                        || z == stride as usize - 1;
                    let meta = context
                        .program
                        .terrain
                        .block(&view, at, lod, halo, &mut scratch);

                    chunk.vec[x * edge * edge + y * edge + z] = meta;
                }
            }
        }

        let sites = context.sites_in_range(pos - 1, 2)?;

        for (index, placement) in context.program.placements.iter().enumerate() {
            context.stamp(placement.structure, &sites[index], &mut chunk, lod);
        }

        let actors = if lod == 0 {
            sites
                .iter()
                .zip(context.program.placements.iter())
                .flat_map(|(sites, placement)| {
                    let asset = STRUCTURES.get(placement.structure);

                    sites
                        .iter()
                        .flat_map(move |site| {
                            asset
                                .variant(site.variant)
                                .actors
                                .iter()
                                .filter_map(move |marker| {
                                    let pos = site.pos.as_vec3() - REGION_SIZE as f32 + marker.pos;

                                    if pos.min_element() < 0.0
                                        || pos.max_element() >= REGION_SIZE as f32
                                    {
                                        None
                                    } else {
                                        Some(ActorMarker {
                                            pos,
                                            actor: marker.actor,
                                        })
                                    }
                                })
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            Vec::new()
        };

        if pos == RegionPos::ZERO && lod == 0 {
            let solid = chunk.vec.iter().filter(|&&meta| meta != 0).count();
            let site_count = sites.iter().map(Vec::len).sum::<usize>();
            let per_structure = sites.iter().map(Vec::len).collect::<Vec<_>>();
            log::debug!(
                "generated the origin region with {solid} solid blocks, {site_count} structure \
                 sites, per structure {per_structure:?} and {} actor markers",
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
        for grid in &self.grids {
            grid.translate(translation);
        }

        self.candidates.translate(translation);
        self.sites.translate(translation);
    }

    /// The fields of the `side ^ 3` regions starting at `pos`, computing the
    /// fields that are missing.
    fn grids_in_range(&self, pos: RegionPos, side: u8) -> anyhow::Result<RegionGrids> {
        let mut volumes = Vec::with_capacity(self.grids.len());

        for (field, ring) in self.grids.iter().enumerate() {
            let graph = &self.program.graphs[field];
            let mut vec = Vec::with_capacity((side as usize).pow(3));

            for dx in 0..side {
                for dy in 0..side {
                    for dz in 0..side {
                        let region = pos + U8Vec3::new(dx, dy, dz).as_ivec3();

                        if ring.get(region)?.is_none() {
                            ring.set(region, graph.evaluate(region, self.seed))?;
                        }

                        vec.push(
                            ring.get(region)?
                                .expect("the field should have been computed above"),
                        );
                    }
                }
            }

            volumes.push(Volume {
                size: U8Vec3::splat(side),
                vec,
            });
        }

        Ok(volumes)
    }

    /// The sites one region proposes, which only depend on the seed and the
    /// program of the generator.
    fn candidates_of(&self, region: RegionPos) -> anyhow::Result<Arc<Vec<Vec<Site>>>> {
        if let Some(candidates) = self.candidates.get(region)? {
            return Ok(candidates);
        }

        let fields = self.grids_in_range(region - 1, 3)?;
        let view = FieldView {
            graphs: &self.program.graphs,
            regions: &fields,
            base: region - 1,
        };
        let origin = region * REGION_SIZE as i32;
        let mut candidates = Vec::with_capacity(self.program.placements.len());

        for placement in &self.program.placements {
            let asset = STRUCTURES.get(placement.structure);
            let mut sites = Vec::new();
            let mut bounds = Vec::new();

            if self.gate_holds(placement, &view, region) {
                let mut seeder =
                    Seeder::from((self.seed, region.x, region.y, region.z, placement.structure));
                let mut rng: Pcg64Mcg = seeder.into_rng();
                let mut scratch = vec![0.0f32; placement.condition.samples()];
                let variants = asset.variants().len().max(1) as u32;

                for _ in 0..placement.attempts {
                    if sites.len() >= asset.max_sites as usize {
                        break;
                    }

                    let pos = U8Vec3::new(
                        (rng.next_u32() % REGION_SIZE as u32) as u8,
                        (rng.next_u32() % REGION_SIZE as u32) as u8,
                        (rng.next_u32() % REGION_SIZE as u32) as u8,
                    );

                    if !placement
                        .condition
                        .holds(&view, origin + pos.as_ivec3(), 0, &mut scratch)
                    {
                        continue;
                    }

                    let variant = (rng.next_u32() % variants) as u8;
                    let bound = asset.variant(variant).bound.translate(pos);

                    if bounds.iter().any(|other| bound.intersects_with(*other)) {
                        continue;
                    }

                    sites.push(Site { pos, variant });
                    bounds.push(bound);
                }
            }

            candidates.push(sites);
        }

        let candidates = Arc::new(candidates);
        self.candidates.set(region, (*candidates).clone())?;

        Ok(candidates)
    }

    /// Whether a region is worth trying to place the structure of `placement` in,
    /// which is read at the corners of the region expanded by the structure.
    ///
    /// The corners are clipped to the region and its two neighbors, which is what
    /// the fields of a task cover: a structure is at most one region wide, so the
    /// clip only moves the outermost corner by a block.
    fn gate_holds(&self, placement: &Placement, view: &FieldView, region: RegionPos) -> bool {
        let Some(gate) = &placement.gate else {
            return true;
        };

        let origin = region * REGION_SIZE as i32;
        let min = origin;
        let max = origin + IVec3::splat(REGION_SIZE as i32) + placement.size.as_ivec3();
        let span = max - min;
        let last = IVec3::splat(2 * REGION_SIZE as i32) - IVec3::ONE;
        let mut scratch = vec![0.0f32; gate.samples()];

        U8Vec3::corners(U8Vec3::ZERO, U8Vec3::ONE)
            .into_iter()
            .any(|corner| {
                let at = (span * corner.as_ivec3()).min(last);

                gate.holds(view, min + at, 0, &mut scratch)
            })
    }

    /// The sites one region keeps, which are its candidates without the ones that
    /// overlap a candidate of a neighbor.
    ///
    /// Both are cached, so a region is placed and resolved once: reading only the
    /// candidates of the seven neighbors of the positive octant keeps the result
    /// independent of the order the regions are generated in.
    fn sites_of(&self, region: RegionPos) -> anyhow::Result<Arc<Vec<Vec<Site>>>> {
        if let Some(sites) = self.sites.get(region)? {
            return Ok(sites);
        }

        let candidates = self.candidates_of(region)?;
        let mut adjacent = Vec::new();

        for offset in U8Vec3::corners(U8Vec3::ZERO, U8Vec3::ONE)
            .into_iter()
            .skip(1)
        {
            let neighbor = region + offset.as_ivec3();

            for (sites, placement) in self
                .candidates_of(neighbor)?
                .iter()
                .zip(&self.program.placements)
            {
                let asset = STRUCTURES.get(placement.structure);

                for site in sites {
                    adjacent.push(
                        asset
                            .variant(site.variant)
                            .bound
                            .translate(offset * REGION_SIZE + site.pos),
                    );
                }
            }
        }

        let mut kept = Vec::with_capacity(candidates.len());

        for (sites, placement) in candidates.iter().zip(&self.program.placements) {
            let asset = STRUCTURES.get(placement.structure);

            kept.push(
                sites
                    .iter()
                    .copied()
                    .filter(|site| {
                        let bound = asset.variant(site.variant).bound;

                        (site.pos + bound.max).max_element() <= REGION_SIZE || {
                            let bound = bound.translate(site.pos);
                            adjacent
                                .iter()
                                .all(|&adjacent| !bound.intersects_with(adjacent))
                        }
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let kept = Arc::new(kept);
        self.sites.set(region, (*kept).clone())?;

        Ok(kept)
    }

    /// The sites of every structure in the `side ^ 3` regions starting at `pos`,
    /// as offsets from the block position of the first region.
    fn sites_in_range(&self, pos: RegionPos, side: u8) -> anyhow::Result<Vec<Vec<Site>>> {
        let mut sites = vec![Vec::new(); self.program.placements.len()];

        for dx in 0..side {
            for dy in 0..side {
                for dz in 0..side {
                    let offset = U8Vec3::new(dx, dy, dz);
                    let region = pos + offset.as_ivec3();

                    sites
                        .iter_mut()
                        .zip(self.sites_of(region)?.iter())
                        .for_each(|(structure_sites, region_sites)| {
                            for site in region_sites.iter() {
                                structure_sites.push(Site {
                                    pos: site.pos + offset * REGION_SIZE,
                                    variant: site.variant,
                                });
                            }
                        });
                }
            }
        }

        Ok(sites)
    }

    /// Stamps the content of one structure into the chunk of one region.
    fn stamp(&self, structure: Id, sites: &[Site], chunk: &mut BlockChunk, lod: u8) {
        let asset = STRUCTURES.get(structure);
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

        for site in sites {
            let variant = asset.variant(site.variant);
            let Some(blocks) = variant.blocks(lod) else {
                continue;
            };

            let range = blocks.bound().translate(site.pos >> lod);

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
