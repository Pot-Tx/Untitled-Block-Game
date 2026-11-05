//! The voxel world: a ring of regions around the player, generated and meshed
//! on background threads.
//!
//! Regions are the unit of storage and loading, sub-regions the unit of
//! meshing, and every region is kept at the level of detail that its distance
//! from the player asks for.

mod block;
mod generation;
mod lighting;
mod model;
mod navigation;
mod region;

use crate::actor::{ACTOR_TYPES, ActorMarker, Persistent, PlayerControlled, Position, Rotation};
use crate::ecs::*;
use crate::game::Paused;
use crate::render::*;
use crate::resources;
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::util::coord::{Axis, Coord3};
use crate::util::math::CubeShellIter;
pub use block::*;
use crossbeam_channel::Sender;
pub use generation::*;
use glam::{IVec3, U8Vec3};
pub(crate) use lighting::*;
use log::{error, trace, warn};
pub use model::*;
pub use navigation::*;
use rayon::ThreadPool;
pub use region::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use wgpu::{BlendState, Color, Face, LoadOp, PrimitiveTopology, TextureFormat};

/// Position of a region, in region sized steps.
pub type RegionPos = IVec3;
/// Position of a block, in blocks.
pub type BlockPos = IVec3;
/// The number of levels of detail, including level 0.
pub const LOD_COUNT: usize = MAX_LOD as usize + 1;
/// How often the changed regions are written to disk.
pub(crate) static SAVE_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_mins(5));

/// Registers the resources of the voxel world.
pub(crate) fn register(resources: &mut ResourceManager) {
    resources.register("gravity", Gravity(0.125));
}

/// Registers the system that updates the voxel world, which owns the generator
/// and the mesher that drive its background work.
pub(crate) fn register_updater(
    tick: &mut SystemManager<1>,
    generator: Generator,
    mesher: ChunkMesher,
) {
    tick.register(6, "world_updater", WorldUpdater::new(generator, mesher));
    // The lights of the sky move before the world updates, so that everything
    // that reads them sees the light of the current tick.
    tick.register(1, "sky_updater", SkyUpdater);
}

/// Loads the blocks and registers the system that draws them, which needs a
/// canvas.
pub(crate) fn register_canvas(frame: &mut SystemManager<2>, canvas: &Canvas) {
    if let Err(e) = init_block_textures(canvas) {
        error!("failed to load block textures: {e}");
    }
    if let Err(e) = init_occlusion_textures(canvas) {
        error!("failed to load occlusion textures: {e}");
    }
    if let Err(e) = init_block_model_templates(canvas) {
        error!("failed to load block model templates: {e}")
    }
    if let Err(e) = init_block_types() {
        error!("failed to load block types: {e}");
    }

    // Every light and shadow view is prepared before any pass that reads them.
    frame.register(3, "lighting", LightingUpdater);

    // All prepared shadow views are drawn before the passes that read them.
    frame.register(4, "shadow", ShadowRenderer::new(canvas));

    frame.register(5, "world_renderer", WorldRenderer::new(canvas));

    // The occlusion shades the faces of the blocks the pass above drew.
    frame.register(6, "occlusion", OcclusionRenderer::new(canvas));

    frame.register(7, "hi_z_builder", HiZBuilder);

    frame.register(9, "translucent", TranslucentRenderer::new(canvas));
    frame.register(10, "wboit_resolve", WboitResolver);
}

/// A sliding window onto a larger grid.
///
/// Positions are mapped into a volume of `2 * radius + 1` entries by wrapping
/// around, so moving the window by one step only has to clear the slabs that
/// leave it.
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

    /// The value stored for `pos`, or `None` when `pos` is outside the window or
    /// its entry is empty.
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

    /// Removes and returns the value stored for `pos`, or `None` when `pos` is
    /// outside the window or its entry is empty.
    pub fn take(&mut self, pos: IVec3) -> Option<T> {
        if self.bound.is_point_inside(pos) {
            self.volume.set(self.cast_pos(pos), None)
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

/// A region that left the ring but still has to finish saving.
struct Leaving {
    region: Region,
    /// Whether another save has to be started once the running one finishes,
    /// because that save was started before a later change.
    resave: bool,
    /// How often a failed save has been retried.
    attempts: u8,
}

pub struct World {
    /// Outer radius of each level of detail, ordered from the finest level to
    /// the coarsest one.
    lod_radii: SmallVec<[u8; LOD_COUNT]>,
    /// The level of detail that is currently being filled in.
    update_lod: u8,
    /// The position each level has been filled up to, as a shell around the
    /// player.
    update_iters: SmallVec<[Option<CubeShellIter<RegionPos>>; LOD_COUNT]>,
    regions: RingVolume<Region>,
    /// The regions the camera of the current frame sees, which [`Self::cull`]
    /// rebuilds for every pass that draws them.
    visible: Vec<RegionPos>,
    /// Coarse regions whose generation waits for the current view.
    deferred_regions: HashMap<RegionPos, u8>,
    generating_regions: HashSet<RegionPos>,
    meshing_regions: HashSet<RegionPos>,
    /// The regions whose changes still have to be written back to disk.
    changed_regions: HashSet<RegionPos>,
    /// The regions whose save is queued or running.
    saving_regions: HashSet<RegionPos>,
    /// The regions that left the ring but still have to finish saving.
    leaving: Vec<Leaving>,

    /// The radius, in regions, the entities of the world are kept in. It has to
    /// stay inside the finest level of detail, so that a region leaves the
    /// entity scope while it still keeps its own blocks.
    entity_radius: u8,
    /// The regions whose entities are in memory.
    entity_scope: HashSet<RegionPos>,
    /// The regions that entered and left the scope since the last drain.
    entity_entered: Vec<RegionPos>,
    entity_left: Vec<RegionPos>,
    /// Changes whenever an existing block changes the visible geometry.
    geometry_revision: u64,

    gen_tx: Sender<GenTask>,
    meshing_tx: Sender<MeshingTask>,
}

resources! {
    /// Downwards acceleration applied to every actor that cannot fly.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Gravity(f32): asset, save;
}

impl Resource for World {}

impl World {
    /// Work budget of one call to [`Self::update`], in blocks that may be added
    /// to the world.
    const MAX_UPDATE_COST: usize = 1024;
    /// How often the save of a region that left the ring is restarted before
    /// it is given up on.
    const MAX_LEAVING_ATTEMPTS: u8 = 2;
    /// How many save rounds the exit flush runs: the saves that were already
    /// running, and the ones for the regions edited while they ran.
    const MAX_FLUSH_ROUNDS: usize = 2;

    pub fn new(
        center: RegionPos,
        lod_radii: SmallVec<[u8; LOD_COUNT]>,
        entity_radius: u8,
        gen_tx: Sender<GenTask>,
        meshing_tx: Sender<MeshingTask>,
    ) -> Self {
        assert!(
            entity_radius < lod_radii[0],
            "the entity scope has to stay inside the finest level of detail"
        );

        let mut update_iters = SmallVec::new();
        for i in 0..lod_radii.len() {
            // Each level fills its own annulus: level 0 grows from the centre
            // outwards, and every other level starts just outside the radius of
            // the level above it.
            let radius = if i == 0 { 0 } else { lod_radii[i - 1] + 1 };
            update_iters.push(Some(CubeShellIter::from_center(center, radius as i32)));
        }

        let regions = RingVolume::new(
            center,
            *lod_radii.last().expect("lod_radii should not be empty"),
        );

        Self {
            lod_radii,
            update_lod: 0,
            update_iters,
            regions,
            visible: Vec::new(),
            deferred_regions: HashMap::new(),
            generating_regions: HashSet::new(),
            meshing_regions: HashSet::new(),
            changed_regions: HashSet::new(),
            saving_regions: HashSet::new(),
            leaving: Vec::new(),

            entity_radius,
            entity_scope: HashSet::new(),
            entity_entered: Vec::new(),
            entity_left: Vec::new(),
            geometry_revision: 0,

            gen_tx,
            meshing_tx,
        }
    }

    #[inline]
    pub fn cast_pos(pos: BlockPos) -> (RegionPos, RelPos) {
        let region_size = IVec3::splat(REGION_SIZE as i32);
        let region_pos = pos.div_euclid(region_size);
        let rel_block_pos = pos.rem_euclid(region_size).as_u8vec3();
        (region_pos, rel_block_pos)
    }

    /// A block on the border of a region is part of the halo of its neighbours,
    /// so up to eight regions are influenced by a single change.
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

    /// The block at `pos`, or air when its region is not loaded.
    pub fn get_block(&self, pos: BlockPos) -> Block {
        let (region_pos, rel_block_pos) = Self::cast_pos(pos);
        if let Some(region) = self.regions.get(region_pos) {
            region.get_block(rel_block_pos)
        } else {
            Block::air()
        }
    }

    /// The light emitters of the loaded world, in world coordinates.
    ///
    /// Iteration reports source descriptors only. It does not evaluate light
    /// propagation, attenuation, or which surfaces an emitter can reach.
    pub(crate) fn light_sources(&self) -> impl Iterator<Item = (BlockPos, Emission)> + '_ {
        self.regions
            .volume
            .vec
            .iter()
            .filter_map(|region| region.as_ref())
            .flat_map(|region| {
                let origin = region.origin();
                region
                    .light_sources()
                    .map(move |(pos, emission)| (origin + pos.as_ivec3(), emission))
            })
    }

    #[inline]
    fn region_visible(&self, region: &Region, visibility: &Visibility) -> bool {
        visibility.is_visible(region.bound(), self.geometry_revision)
    }

    pub fn set_block(&mut self, pos: BlockPos, block: Block) {
        let (region_pos, rel_block_pos) = Self::cast_pos(pos);
        let mut changed = false;

        for influenced_region_pos in Self::pos_influence(pos) {
            if let Some(region) = self.regions.get_mut(influenced_region_pos) {
                // The region local position, moved into the halo of the region
                // that is being written: the difference between the two regions
                // converted to blocks, plus the one block wide border.
                let pos: LocalPos = (rel_block_pos.as_i16vec3()
                    + (region_pos - influenced_region_pos).as_i16vec3() * REGION_SIZE as i16
                    + 1)
                .as_u8vec3();
                if region.set_block(pos, block) {
                    changed = true;
                    self.meshing_regions.insert(influenced_region_pos);
                    self.changed_regions.insert(influenced_region_pos);
                }
            }
        }

        if changed {
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
        }
    }

    /// The work per call is bounded by `MAX_UPDATE_COST`, so a fast player
    /// spreads the loading of new regions over several ticks.
    ///
    /// `translation` is the change of `center`, in region units.
    pub(crate) fn update(
        &mut self,
        center: RegionPos,
        translation: IVec3,
        visibility: &Visibility,
    ) -> Vec<ActorMarker> {
        self.update_translation(center, translation);

        let lod_radii = &self.lod_radii;
        for (&pos, lod) in &mut self.deferred_regions {
            let radius = (pos - center).abs().max_element();
            *lod = lod_radii
                .iter()
                .position(|&max_radius| radius <= max_radius as i32)
                .unwrap_or(lod_radii.len() - 1) as u8;
        }

        let mut cost = 0;
        self.admit_deferred(center, visibility, &mut cost);

        while self.update_lod < self.lod_radii.len() as u8 {
            let level = self.update_lod as usize;
            let max_radius = self.lod_radii[level];
            let mut exhausted = false;

            while cost < Self::MAX_UPDATE_COST {
                let next = self.update_iters[level]
                    .as_mut()
                    .and_then(|iter| iter.next());

                match next {
                    Some(pos) => {
                        let lod = self.update_lod;

                        if !self.admits_region(pos, lod, visibility) {
                            self.deferred_regions.insert(pos, lod);
                            cost += 1;
                            continue;
                        }

                        self.deferred_regions.remove(&pos);
                        cost += self.admit_region(pos, lod);
                    }

                    None => {
                        exhausted = true;
                        break;
                    }
                }
            }

            if exhausted && cost < Self::MAX_UPDATE_COST {
                let iter = self.update_iters[level]
                    .as_ref()
                    .expect("a level being filled has an iterator");
                // The shell is exhausted, so widen it by one region unless
                // the level has reached its outer radius.
                let next_radius = ((iter.max - iter.origin).x as u8 / 2) + 1;
                if next_radius <= max_radius {
                    self.update_iters[level] =
                        Some(CubeShellIter::from_center(center, next_radius as i32));
                } else {
                    self.update_iters[level] = None;
                    self.update_lod += 1;
                }
            } else if cost >= Self::MAX_UPDATE_COST {
                break;
            }
        }

        trace!(
            "world visibility deferred {} region(s)",
            self.deferred_regions.len()
        );

        let markers = self.poll_gen();

        self.poll_saves();

        self.update_entity(center);
        markers
    }

    /// Returns whether a region may start or change its level of detail.
    ///
    /// The finest level always remains resident and generated. Coarser levels
    /// wait until they are inside the current frustum and are not conservatively
    /// occluded by the previous frame's Hi-Z pyramid.
    fn admits_region(&self, pos: RegionPos, lod: u8, visibility: &Visibility) -> bool {
        lod == 0 || visibility.is_visible(Region::bound_at(pos), self.geometry_revision)
    }

    /// Starts the admission work for a region that passed the visibility gate.
    fn admit_region(&mut self, pos: RegionPos, lod: u8) -> usize {
        let complexity = (LOD_COUNT - lod as usize).pow(2);

        if let Some(region) = self.regions.get_mut(pos) {
            if region.set_lod(lod) {
                self.generating_regions.insert(pos);
                complexity
            } else {
                1
            }
        } else {
            let region = Region::new(pos, lod, self.gen_tx.clone(), self.meshing_tx.clone());
            self.regions.set(pos, region);
            self.generating_regions.insert(pos);
            complexity
        }
    }

    /// Retries deferred coarse regions nearest to the player first.
    fn admit_deferred(&mut self, center: RegionPos, visibility: &Visibility, cost: &mut usize) {
        let mut ready = self
            .deferred_regions
            .iter()
            .filter_map(|(&pos, &lod)| {
                self.admits_region(pos, lod, visibility)
                    .then_some((pos, lod))
            })
            .collect::<Vec<_>>();

        ready.sort_by_key(|(pos, _)| (pos - center).abs().max_element());

        for (pos, lod) in ready {
            if *cost >= Self::MAX_UPDATE_COST {
                break;
            }

            self.deferred_regions.remove(&pos);
            *cost += self.admit_region(pos, lod);
        }
    }

    /// The regions whose entities entered and left the scope since the last
    /// drain, which is what the store follows.
    pub fn take_entity_scope(&mut self) -> (Vec<RegionPos>, Vec<RegionPos>) {
        (
            std::mem::take(&mut self.entity_entered),
            std::mem::take(&mut self.entity_left),
        )
    }

    fn update_translation(&mut self, center: RegionPos, translation: IVec3) {
        // `translation` is in region units, like `center`.
        if translation != IVec3::ZERO {
            let bound = self.regions.bound.translate(translation);
            self.deferred_regions
                .retain(|pos, _| bound.is_point_inside(*pos));

            // A region that leaves the ring is dropped, so its changes have to
            // be written before the slab it lives in is cleared. A region that
            // still has to finish saving is kept until its save is done.
            let leaving = self
                .changed_regions
                .iter()
                .chain(self.saving_regions.iter())
                .copied()
                .filter(|pos| !bound.is_point_inside(*pos))
                .collect::<HashSet<_>>();

            for pos in leaving {
                // A save that is already running was started before a change
                // that still has the region in `changed_regions`, so another
                // save has to follow it.
                let saving = self.saving_regions.remove(&pos);
                let resave = saving && self.changed_regions.remove(&pos);
                let Some(mut region) = self.regions.take(pos) else {
                    continue;
                };

                if !saving {
                    region.begin_saving();
                }

                self.leaving.push(Leaving {
                    region,
                    resave,
                    attempts: 0,
                });
            }

            self.regions.translate(translation);

            // Restart each level at the shell the translation left off at, so
            // that regions that are already loaded are not generated again.
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
    }

    /// Recomputes the entity scope and records what changed.
    ///
    /// The scope is a cube of a fixed radius around the player, so the whole
    /// cube is walked instead of watching every way a region can enter or leave
    /// it; that also keeps a teleport from being missed.
    fn update_entity(&mut self, center: RegionPos) {
        let radius = self.entity_radius as i32;
        let mut scope = HashSet::new();

        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let pos = center + IVec3::new(x, y, z);

                    if self.regions.get(pos).is_some_and(Region::ready) {
                        scope.insert(pos);
                    }
                }
            }
        }

        self.entity_entered
            .extend(scope.difference(&self.entity_scope).copied());
        self.entity_left
            .extend(self.entity_scope.difference(&scope).copied());
        self.entity_scope = scope;
    }

    /// Asks every region that needs a save to start one. A region whose save is
    /// still running is left until its report comes back.
    fn begin_saving(&mut self) {
        let pending = self
            .changed_regions
            .iter()
            .copied()
            .filter(|pos| !self.saving_regions.contains(pos))
            .collect::<Vec<_>>();

        for pos in pending {
            let Some(region) = self.regions.get_mut(pos) else {
                continue;
            };

            // A dirty region always has its chunk, so the save starts.
            region.begin_saving();
            self.changed_regions.remove(&pos);
            self.saving_regions.insert(pos);
        }
    }

    fn poll_gen(&mut self) -> Vec<ActorMarker> {
        let mut generated = Vec::new();

        self.generating_regions.retain(|&pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                if let Some(mut markers) = region.poll_gen() {
                    self.meshing_regions.insert(pos);

                    markers
                        .iter_mut()
                        .for_each(|m| m.pos += Region::origin_at(pos).as_vec3());
                    generated.extend(markers);

                    false
                } else {
                    true
                }
            } else {
                false
            }
        });

        generated
    }

    /// Applies the finished saves. A save that failed is started again, and a
    /// region edited while its save ran stays in `changed_regions`, so the next
    /// save picks it up.
    fn poll_saves(&mut self) {
        self.saving_regions.retain(|&pos| {
            let Some(region) = self.regions.get_mut(pos) else {
                return false;
            };

            match region.poll_save() {
                // The save is still running.
                None => true,

                // The blocks were written. A region that was edited while the
                // save ran is already back in `changed_regions`.
                Some(true) => false,

                // Writing failed, so the region has to be saved again.
                Some(false) => {
                    self.changed_regions.insert(pos);
                    false
                }
            }
        });

        // The regions that left the ring are polled from their own list, and
        // stay alive until they are saved.
        let mut kept = Vec::new();

        for mut leaving in std::mem::take(&mut self.leaving) {
            match leaving.region.poll_save() {
                None => kept.push(leaving),

                Some(true) if leaving.resave => {
                    leaving.resave = false;
                    leaving.region.begin_saving();
                    kept.push(leaving);
                }

                Some(true) => (),

                Some(false) => {
                    if leaving.attempts < Self::MAX_LEAVING_ATTEMPTS {
                        leaving.attempts += 1;
                        leaving.region.begin_saving();
                        kept.push(leaving);
                    } else {
                        warn!(
                            "failed to save region {} after it left the world",
                            leaving.region.pos()
                        );
                    }
                }
            }
        }

        self.leaving = kept;
    }

    pub fn pre_render(&mut self, canvas: &Canvas) {
        let mut changed = false;

        self.meshing_regions.retain(|&pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                let (finished, region_changed) = region.poll_meshes(canvas);
                changed |= region_changed;
                !finished
            } else {
                false
            }
        });

        if changed {
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
        }
    }

    /// Rebuilds the regions the camera of the current frame sees, which every
    /// pass that draws the world shares so that the frustum and the depth query
    /// run once per frame instead of once per pass.
    ///
    /// Call after [`Self::pre_render`], which applies the meshes and updates the
    /// revision the test reads.
    fn cull(&mut self, visibility: &Visibility) {
        let mut visible = std::mem::take(&mut self.visible);
        visible.clear();
        visible.extend(
            self.regions
                .volume
                .vec
                .iter()
                .flatten()
                .filter(|region| self.region_visible(region, visibility))
                .map(Region::pos),
        );
        self.visible = visible;
    }

    /// The regions [`Self::cull`] found visible, which are valid for the rest of
    /// the frame that reads them.
    fn visible_regions(&self) -> impl Iterator<Item = &Region> + '_ {
        self.visible.iter().filter_map(|&pos| self.regions.get(pos))
    }

    /// Saves everything that still has to be written and waits until it is.
    ///
    /// No edit can happen during the flush, so two rounds are enough: the
    /// saves that were already running, and the ones for the regions that were
    /// edited while those ran. A region that is still dirty after them failed
    /// to be written and is given up on.
    pub fn flush(&mut self) {
        self.begin_saving();

        for _ in 0..Self::MAX_FLUSH_ROUNDS {
            self.begin_saving();

            while !self.saving_regions.is_empty() || !self.leaving.is_empty() {
                self.poll_saves();

                if self.saving_regions.is_empty() && self.leaving.is_empty() {
                    break;
                }

                std::thread::sleep(Duration::from_millis(1));
            }
        }

        if !self.changed_regions.is_empty() {
            warn!(
                "gave up saving {} region(s) before the window closed",
                self.changed_regions.len()
            );
            self.changed_regions.clear();
        }
    }
}

resources! {
    /// The thread pools that generate and mesh regions: `near` serves the near
    /// levels of detail and `far` the far ones.
    pub struct WorldThreads {
        pub near: ThreadPool,
        pub far: ThreadPool,
    };
}

struct WorldUpdater {
    save_timer: Instant,
    generator: Generator,
    mesher: ChunkMesher,
}

impl System<1> for WorldUpdater {
    type CompQueries = ((CompRead<PlayerControlled>, CompRead<Position>),);
    type ResQuery = (
        ResWrite<World>,
        ResRead<WorldThreads>,
        ResRead<Paused>,
        ResRead<Visibility>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        com: [Commander<'_>; 1],
    ) {
        let (world, threads, paused, visibility) = res;

        for entry in comp.0.iter() {
            let position = entry.2;
            let pos = position.cur.floor().as_ivec3();
            let center = pos.div_euclid(IVec3::splat(REGION_SIZE as i32));
            let prev_pos = position.prev.floor().as_ivec3();
            let translation = center - prev_pos.div_euclid(IVec3::splat(REGION_SIZE as i32));

            let markers = world.update(center, translation, visibility);
            {
                let mut guard = com[0].lock();
                markers
                    .into_iter()
                    .map(|m| {
                        ACTOR_TYPES
                            .get(m.actor)
                            .clone()
                            .with::<Position>(m.pos.into())
                            .with(Persistent)
                    })
                    .for_each(|desc| {
                        guard.spawn(desc);
                    });
            }
            self.generator.update(translation, threads);
        }

        if self.save_timer.elapsed() >= *SAVE_DURATION || paused.0 {
            self.save_timer = Instant::now();
            world.begin_saving();
        }

        self.mesher.update(threads);
    }
}

impl WorldUpdater {
    pub fn new(generator: Generator, mesher: ChunkMesher) -> Self {
        Self {
            save_timer: Instant::now(),
            generator,
            mesher,
        }
    }
}

/// Captures the opaque terrain after the world passes and before actors, so
/// that dynamic objects do not become stale occluders.
struct HiZBuilder;

impl System<2> for HiZBuilder {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (
        ResRead<Canvas>,
        ResRead<Camera>,
        ResRead<World>,
        ResWrite<Visibility>,
        ResWrite<FrameSlot>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _: [Commander<'_>; 2],
    ) {
        if comp.0.iter().next().is_none() {
            return;
        }

        let (canvas, camera, world, visibility, frame) = res;
        if let Some(frame) = frame.as_mut() {
            visibility.capture(canvas, frame, camera, world.geometry_revision);
        }
    }
}

struct WorldRenderer {
    opaque_batch: RenderBatch<
        (
            BlockTextureSet,
            CameraTransform,
            WorldLightsLayout,
            BlockTemplates,
        ),
        (),
        BlockRectInst,
        ColorDraw<1>,
        DepthDraw,
    >,
}

impl System<2> for WorldRenderer {
    type CompQueries = (
        (
            CompRead<PlayerControlled>,
            CompRead<Position>,
            CompRead<Rotation>,
        ),
        (),
    );
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<FrameSlot>,
        ResRead<Camera>,
        ResWrite<World>,
        ResRead<Lighting>,
        ResRead<Visibility>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, frame, camera, world, lights, visibility) = res;

        for _ in comp.0.iter() {
            world.pre_render(canvas);

            if let Some(frame) = frame.as_mut() {
                world.cull(visibility);

                let colors = [ColorTarget {
                    view: &canvas.color_view,
                    load: LoadOp::Clear(Color {
                        r: 0.375,
                        g: 0.625,
                        b: 1.0,
                        a: 1.0,
                    }),
                }];
                let depth = DepthTarget {
                    view: &canvas.depth_view,
                    load: LoadOp::Clear(0.0),
                };

                frame.render::<ColorDraw<1>, DepthDraw>("block", colors, depth, |mut pass| {
                    self.opaque_batch.begin(&mut pass);
                    self.opaque_batch.push(
                        &mut pass,
                        (
                            &BLOCK_SURFACES,
                            &camera.transform,
                            &lights.bind_group,
                            &BLOCK_TEMPLATES,
                        ),
                    );

                    for region in world.visible_regions() {
                        if let Some(rects) = region.model().blocks() {
                            self.opaque_batch.draw(&mut pass, &rects);
                        }
                    }
                });
            }
        }
    }
}

impl WorldRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            opaque_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "block",
                    // `block` draws the same geometry without any lighting.
                    shader: "block_lighting",
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: None,
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
        }
    }
}

/// Shades the corners of the blocks that the light of the sky does not reach.
///
/// The geometry is the faces of the blocks a second time, so the pass reads the
/// pattern of each face and darkens the texels of the block with the occlusion
/// of the corners around them.
struct OcclusionRenderer {
    batch: RenderBatch<
        (TextureArraySampler, CameraTransform, BlockTemplates),
        (),
        BlockRectInst,
        ColorDraw<1>,
        DepthDraw,
    >,
}

impl System<2> for OcclusionRenderer {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<FrameSlot>,
        ResRead<Camera>,
        ResRead<World>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, frame, camera, world) = res;

        for _ in comp.0.iter() {
            if let Some(frame) = frame.as_mut() {
                let colors = [ColorTarget {
                    view: &canvas.color_view,
                    load: LoadOp::Load,
                }];
                let depth = DepthTarget {
                    view: &canvas.depth_view,
                    load: LoadOp::Load,
                };

                frame.render::<ColorDraw<1>, DepthDraw>("occlusion", colors, depth, |mut pass| {
                    self.batch.begin(&mut pass);
                    self.batch.push(
                        &mut pass,
                        (&OCCLUSION_TEXTURES, &camera.transform, &BLOCK_TEMPLATES),
                    );

                    for region in world.visible_regions() {
                        if let Some(rects) = region.model().occlusion() {
                            self.batch.draw(&mut pass, &rects);
                        }
                    }
                });
            }
        }
    }
}

impl OcclusionRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "occlusion",
                    shader: "occlusion",
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: false,
                },
            ),
        }
    }
}

/// Draws camera-visible translucent blocks into the WBOIT targets.
struct TranslucentRenderer {
    batch: RenderBatch<
        (
            BlockTextureSet,
            CameraTransform,
            WorldLightsLayout,
            BlockTemplates,
        ),
        (),
        BlockRectInst,
        ColorDraw<2>,
        DepthDraw,
    >,
}

impl System<2> for TranslucentRenderer {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<FrameSlot>,
        ResRead<Camera>,
        ResRead<World>,
        ResRead<Lighting>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, frame, camera, world, lighting) = res;
        for _ in comp.0.iter() {
            if let Some(frame) = frame.as_mut() {
                let colors = [
                    ColorTarget {
                        view: &canvas.wboit_accum_view,
                        load: LoadOp::Clear(Color::TRANSPARENT),
                    },
                    ColorTarget {
                        view: &canvas.wboit_reveal_view,
                        load: LoadOp::Clear(Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                    },
                ];
                let depth = DepthTarget {
                    view: &canvas.depth_view,
                    load: LoadOp::Load,
                };

                frame.render::<ColorDraw<2>, DepthDraw>(
                    "translucent",
                    colors,
                    depth,
                    |mut pass| {
                        self.batch.begin(&mut pass);
                        self.batch.push(
                            &mut pass,
                            (
                                &BLOCK_SURFACES,
                                &camera.transform,
                                &lighting.bind_group,
                                &BLOCK_TEMPLATES,
                            ),
                        );
                        for region in world.visible_regions() {
                            if let Some(rects) = region.model().translucent() {
                                self.batch.draw(&mut pass, &rects);
                            }
                        }
                    },
                );
            }
        }
    }
}

impl TranslucentRenderer {
    fn new(canvas: &Canvas) -> Self {
        Self {
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "translucent",
                    shader: "block_wboit",
                    color_formats: [TextureFormat::Rgba16Float, TextureFormat::R16Float],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: Some(BlendState::ADDITIVE),
                    secondary_blend: Some(WBOIT_REVEAL_BLEND),
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: false,
                },
            ),
        }
    }
}

/// Composites the WBOIT targets over the opaque scene.
struct WboitResolver;

impl System<2> for WboitResolver {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (ResRead<Canvas>, ResWrite<FrameSlot>);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        if comp.0.iter().next().is_none() {
            return;
        }
        let (canvas, frame) = res;
        if let Some(frame) = frame.as_mut() {
            canvas.resolve_wboit(frame);
        }
    }
}
