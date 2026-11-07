use crate::game::SAVE_DIR;
use crate::render::*;
use crate::util;
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::world::block::BlockChunk;
use crate::generation::*;
use crate::world::model::MeshingTask;
use crate::world::*;
use anyhow::anyhow;
use crossbeam_channel::*;
use glam::{U8Vec3, Vec3};
use log::{error, warn};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Position of a block inside the chunk of a region, including the halo.
pub type LocalPos = U8Vec3;
/// A block position relative to the origin of a region, without the halo.
pub type RelPos = U8Vec3;
/// Position of a sub-region inside a region.
pub type SubRegionPos = U8Vec3;
/// Number of blocks along one axis of a region.
pub const REGION_SIZE: u8 = 32;
/// Number of blocks along one axis of a sub-region.
pub const SUBREGION_SIZE: u8 = 16;
/// Number of blocks along one axis of the chunk of a sub-region, including the
/// one block halo that the mesher needs.
pub const SUBREGION_STRIDE: u8 = SUBREGION_SIZE + 2;
/// Number of sub-regions along one axis of a region.
pub const SUBREGION_COUNT: u8 = REGION_SIZE / SUBREGION_SIZE;
/// The coarsest level of detail, at which a region collapses into a single
/// block.
pub const MAX_LOD: u8 = REGION_SIZE.ilog2() as u8;
/// The magic of the file the blocks of a region are saved in.
const CHUNK_MAGIC: &[u8; 4] = b"CHNK";
/// The version of the chunk file format.
const CHUNK_VERSION: u32 = 0;
/// The directory the blocks of the regions are saved in.
static CHUNK_DIR: LazyLock<PathBuf> = LazyLock::new(|| SAVE_DIR.join("chunks"));

/// The file of the region at `pos` inside `dir`.
#[inline]
pub fn region_path(dir: &Path, pos: RegionPos) -> PathBuf {
    dir.join(format!("{}.{}.{}.sav", pos.x, pos.y, pos.z))
}

#[derive(Clone)]
pub struct Region {
    pos: RegionPos,
    lod: u8,
    /// The blocks of the region, or `None` while it is still being generated.
    chunk: Option<BlockChunk>,
    /// Light emitters in this region, which are sources rather than a propagated
    /// light field.
    lights: HashMap<RelPos, Emission>,
    model: RegionModel,
    /// Number of meshing tasks that have not returned yet.
    meshing: u8,
    /// Whether the region has been touched, by loading it from a save or by
    /// editing it. A touched region keeps its blocks and stays at level 0
    /// instead of being regenerated at another level of detail.
    edited: bool,
    /// The generation the region waits for. Every request bumps it, so that the
    /// results of the requests it has superseded are dropped instead of applied.
    gen_revision: u64,
    /// How often the generation of the current level failed and was asked for
    /// again.
    gen_attempts: u8,
    /// The revision of each sub-region, so that a mesh built from blocks that
    /// changed again in the meantime can be discarded.
    mesh_revisions: Volume<u64>,

    gen_tx: Sender<GenTask>,
    chunk_tx: Sender<GenResult>,
    chunk_rx: Receiver<GenResult>,
    meshing_tx: Sender<MeshingTask>,
    mesh_tx: Sender<MeshingResult>,
    mesh_rx: Receiver<MeshingResult>,
    save_tx: Sender<anyhow::Result<()>>,
    save_rx: Receiver<anyhow::Result<()>>,
}

impl Region {
    /// How often a failed generation is asked for again before the region gives
    /// up, which keeps a region that always fails from generating forever.
    const MAX_GEN_RETRIES: u8 = 3;

    /// The positions of the sub-regions of a region, in the same order as the
    /// corners of a [`Volume`].
    const SUBREGION_POSES: [SubRegionPos; SUBREGION_COUNT.pow(3) as usize] = [
        SubRegionPos::new(0, 0, 0),
        SubRegionPos::new(0, 0, 1),
        SubRegionPos::new(0, 1, 0),
        SubRegionPos::new(0, 1, 1),
        SubRegionPos::new(1, 0, 0),
        SubRegionPos::new(1, 0, 1),
        SubRegionPos::new(1, 1, 0),
        SubRegionPos::new(1, 1, 1),
    ];

    pub fn new(
        pos: RegionPos,
        lod: u8,
        gen_tx: Sender<GenTask>,
        meshing_tx: Sender<MeshingTask>,
    ) -> Self {
        // A region can have several generations and several meshes in flight:
        // every level change and every burst of edits starts another one before
        // the previous result comes back. A bounded channel would drop the
        // newer result once it is full, which is the one the region needs. The
        // level of a generation result decides whether it is applied, and the
        // revision of a mesh result whether it still belongs to the blocks.
        let (chunk_tx, chunk_rx) = unbounded();
        let (mesh_tx, mesh_rx) = unbounded();
        let (save_tx, save_rx) = bounded(1);
        let mut new = Self {
            pos,
            lod,
            chunk: None,
            lights: HashMap::new(),
            model: RegionModel::new(lod),
            meshing: 0,
            edited: false,
            gen_revision: 0,
            gen_attempts: 0,
            mesh_revisions: Volume::splat(U8Vec3::splat(SUBREGION_COUNT), 0),

            gen_tx,
            chunk_tx,
            chunk_rx,
            meshing_tx,
            mesh_tx,
            mesh_rx,
            save_tx,
            save_rx,
        };

        new.begin_gen();
        new
    }

    /// The number of blocks one block of level `lod` covers on every axis.
    #[inline]
    pub const fn block_size_on_lod(lod: u8) -> u8 {
        1 << lod
    }

    /// The number of blocks along one axis of a region at `lod`.
    #[inline]
    pub const fn size_on_lod(lod: u8) -> u8 {
        REGION_SIZE >> lod
    }

    /// The number of blocks along one axis of the chunk of a region at `lod`,
    /// including its halo.
    #[inline]
    pub const fn stride_on_lod(lod: u8) -> u8 {
        (REGION_SIZE >> lod) + 2
    }

    /// The mesher reads the blocks around a sub-region, so a block on the border
    /// also changes the mesh of the neighboring sub-regions.
    #[inline]
    fn pos_influence(pos: LocalPos) -> SmallVec<[SubRegionPos; SUBREGION_COUNT.pow(3) as usize]> {
        let mut influenced = SmallVec::new();

        fn range(b: u8) -> (u8, u8) {
            let b = b as i16;
            let s = SUBREGION_SIZE as i16;
            let min = ((b - s - 1) / s).max(0);
            let max = (b / s).min(SUBREGION_COUNT as i16 - 1);
            (min as u8, max as u8)
        }

        let (minx, maxx) = range(pos.x);
        let (miny, maxy) = range(pos.y);
        let (minz, maxz) = range(pos.z);

        for x in minx..=maxx {
            for y in miny..=maxy {
                for z in minz..=maxz {
                    influenced.push(SubRegionPos::new(x, y, z));
                }
            }
        }

        influenced
    }

    #[inline]
    pub fn model(&self) -> &RegionModel {
        &self.model
    }

    /// The world position of the corner of the region at `pos`.
    #[inline]
    pub fn origin_at(pos: RegionPos) -> BlockPos {
        pos * REGION_SIZE as i32
    }

    /// The world position of the corner of this region.
    #[inline]
    pub fn origin(&self) -> BlockPos {
        Self::origin_at(self.pos)
    }

    #[inline]
    pub(crate) fn bound_at(pos: RegionPos) -> AABB<Vec3> {
        AABB {
            min: Self::origin_at(pos).as_vec3(),
            max: Self::origin_at(pos + 1).as_vec3(),
        }
    }

    pub fn bound(&self) -> AABB<Vec3> {
        Self::bound_at(self.pos)
    }

    /// Whether the region has its blocks, which the entities of the region need
    /// before they can be placed in it.
    #[inline]
    pub fn ready(&self) -> bool {
        self.chunk.is_some()
    }

    /// The block at `pos`, which is region local and does not include the halo
    /// of the chunk.
    pub fn get_block(&self, pos: RelPos) -> Block {
        match &self.chunk {
            None => Block::air(),
            Some(chunk) => chunk.get_block(pos + 1),
        }
    }

    /// The light emitters of this region, at their region-local positions.
    pub(crate) fn light_sources(&self) -> impl Iterator<Item = (RelPos, Emission)> + '_ {
        self.lights.iter().map(|(&pos, &emission)| (pos, emission))
    }

    /// Rebuilds the source index from the blocks of this region.
    fn collect_lights(&mut self) {
        self.lights.clear();
        let Some(chunk) = &self.chunk else {
            return;
        };

        // Local lights are only meaningful near the viewer. Keeping the first
        // version at LOD 0 avoids inventing a source strength for a coarse block
        // that may stand for several emitters.
        if self.lod != 0 {
            return;
        }

        for x in 0..REGION_SIZE {
            for y in 0..REGION_SIZE {
                for z in 0..REGION_SIZE {
                    let pos = RelPos::new(x, y, z);
                    let block = Block::from_meta(*chunk.get(pos + 1));

                    if let Some(emission) = block.block_type.emission {
                        self.lights.insert(pos, emission);
                    }
                }
            }
        }
    }

    /// Writes `block` at `pos`, which is chunk local and therefore includes
    /// the halo. The world edits the chunks of its neighbors through it.
    ///
    /// Returns whether the region has a chunk to write to.
    pub fn set_block(&mut self, pos: LocalPos, block: Block) -> bool {
        match &mut self.chunk {
            None => false,

            Some(chunk) => {
                // The world edits the chunk through its halo, while the light
                // index is keyed by the region's interior coordinates.
                if self.lod == 0 && pos.min_element() > 0 && pos.max_element() <= REGION_SIZE {
                    if let Some(emission) = block.block_type.emission {
                        self.lights.insert(pos - 1, emission);
                    } else {
                        self.lights.remove(&(pos - 1));
                    }
                }

                chunk.set_block(pos, block);
                self.edited = true;
                self.begin_meshing(Self::pos_influence(pos));
                true
            }
        }
    }

    /// Loads the region from its save, or asks the generator for it when there
    /// is no save. The read runs on the world I/O runtime, so that creating a
    /// region never blocks the tick on the disk.
    fn begin_gen(&mut self) {
        self.lights.clear();
        self.gen_attempts = 0;
        self.gen_revision = self.gen_revision.wrapping_add(1);
        let revision = self.gen_revision;
        let pos = self.pos;
        let lod = self.lod;
        let gen_tx = self.gen_tx.clone();
        let chunk_tx = self.chunk_tx.clone();

        crate::RUNTIME.spawn_blocking(move || match Self::read_from(pos) {
            Ok(Some(chunk)) => {
                let result = GenResult {
                    revision,
                    content: Ok(GenContent {
                        chunk,
                        actors: Vec::new(),
                        loaded: true,
                    }),
                };

                if let Err(e) = chunk_tx.try_send(result) {
                    error!("failed to send loaded region {} to region: {}", pos, e);
                }
            }

            Ok(None) => Self::begin_generating(pos, lod, revision, gen_tx, chunk_tx),

            Err(e) => {
                warn!("failed to load the region {}: {}", pos, e);
                Self::begin_generating(pos, lod, revision, gen_tx, chunk_tx);
            }
        });

        self.meshing = 0;
    }

    /// Asks the generator for the blocks of a region.
    fn begin_generating(
        pos: RegionPos,
        lod: u8,
        revision: u64,
        gen_tx: Sender<GenTask>,
        chunk_tx: Sender<GenResult>,
    ) {
        if let Err(e) = gen_tx.try_send(GenTask {
            pos,
            lod,
            revision,
            tx: chunk_tx,
        }) {
            error!("failed to send generation task from region {}: {}", pos, e);
        }
    }

    /// Asks the generator for the region again after a generation that failed,
    /// which is the only way a region that could not be built gets a second
    /// chance.
    fn retry_gen(&mut self, error: &anyhow::Error) {
        if self.gen_attempts >= Self::MAX_GEN_RETRIES {
            error!(
                "giving up on region {} at LOD {} after {} retries: {error}",
                self.pos,
                self.lod,
                Self::MAX_GEN_RETRIES
            );
            return;
        }

        self.gen_attempts += 1;
        self.gen_revision = self.gen_revision.wrapping_add(1);
        let revision = self.gen_revision;

        warn!(
            "regenerating region {} at LOD {}, retry {} of {}: {error}",
            self.pos,
            self.lod,
            self.gen_attempts,
            Self::MAX_GEN_RETRIES
        );

        Self::begin_generating(
            self.pos,
            self.lod,
            revision,
            self.gen_tx.clone(),
            self.chunk_tx.clone(),
        );
    }

    /// At level 0 every sub-region is meshed on its own, so that a change only
    /// remeshes the sub-regions around it. Coarser levels collapse the region
    /// into a few blocks, so the whole region is meshed in one task.
    fn begin_meshing(&mut self, poses: SmallVec<[SubRegionPos; SUBREGION_COUNT.pow(3) as usize]>) {
        /// Bumps the revision of `pos`, so that a mesh built from blocks that
        /// changed again before it arrived can be discarded.
        #[inline]
        fn bump(revisions: &mut Volume<u64>, pos: SubRegionPos) -> u64 {
            let revision = revisions.get_mut(pos);
            *revision = revision.wrapping_add(1);
            *revision
        }

        match self.lod {
            0 => {
                let revisions = poses
                    .iter()
                    .map(|&pos| bump(&mut self.mesh_revisions, pos))
                    .collect::<SmallVec<[u64; SUBREGION_COUNT.pow(3) as usize]>>();

                if let Some(chunk) = &self.chunk {
                    for (pos, revision) in poses.into_iter().zip(revisions) {
                        match self.meshing_tx.try_send(MeshingTask {
                            lod: self.lod,
                            pos: Some(pos),
                            chunk: chunk
                                .part(pos * SUBREGION_SIZE, U8Vec3::splat(SUBREGION_STRIDE)),
                            origin: self.origin(),
                            tx: self.mesh_tx.clone(),
                            revision,
                        }) {
                            Ok(_) => self.meshing += 1,

                            Err(e) => error!(
                                "failed to send meshing task from region {}: {}",
                                self.pos, e
                            ),
                        }
                    }
                } else {
                    error!("region {} has no chunk to mesh", self.pos);
                }
            }

            1.. => {
                let revision = *self.mesh_revisions.get(U8Vec3::ZERO);

                if let Some(chunk) = self.chunk.take() {
                    match self.meshing_tx.try_send(MeshingTask {
                        lod: self.lod,
                        pos: None,
                        chunk,
                        origin: self.origin(),
                        tx: self.mesh_tx.clone(),
                        revision,
                    }) {
                        Ok(_) => self.meshing += 1,
                        Err(e) => error!(
                            "failed to send meshing task from region {}: {}",
                            self.pos, e
                        ),
                    }
                } else {
                    error!("region {} has no chunk to mesh", self.pos);
                }
            }
        }
    }

    /// Starts writing the region to disk on the game runtime.
    ///
    /// A dirty region always has a chunk to write, so the world only calls this
    /// while the region is not being meshed.
    pub fn begin_saving(&mut self) {
        let Some(chunk) = &self.chunk else {
            error!("region {} has no chunk to save", self.pos);
            return;
        };

        let pos = self.pos;
        let chunk = chunk.clone();
        let tx = self.save_tx.clone();

        crate::RUNTIME.spawn_blocking(move || {
            let result = Self::write_to(pos, &chunk);

            if let Err(e) = tx.try_send(result) {
                error!("failed to report the save of region {}: {}", pos, e);
            }
        });
    }

    /// Returns whether the level changed; a region that has been touched, by
    /// loading it from a save or by editing it, keeps its blocks and is never
    /// regenerated at another level.
    pub fn set_lod(&mut self, lod: u8) -> bool {
        if self.lod == lod || self.edited {
            false
        } else {
            self.lod = lod;
            // The blocks were generated at the level they belong to, and every
            // reader of them takes the block size from the current level, so a
            // chunk must not outlive the level it was built for.
            self.chunk = None;
            self.lights.clear();
            self.begin_gen();
            true
        }
    }

    /// Applies the result of the fill and returns the markers of a newly
    /// generated LOD0 region. Markers are not retained by the region.
    ///
    /// Every request of the region may still have a result on its way, so the
    /// ones of an earlier request are dropped, and a failed generation of the
    /// newest request is asked for again.
    pub fn poll_gen(&mut self) -> Option<Vec<ActorMarker>> {
        let mut actors = None;

        while let Ok(result) = self.chunk_rx.try_recv() {
            if result.revision != self.gen_revision {
                continue;
            }

            match result.content {
                Ok(content) => {
                    if content.loaded {
                        // A saved region belongs to the player, so it keeps full
                        // detail instead of the level the world asked for.
                        self.lod = 0;
                        self.edited = true;
                    }

                    self.chunk = Some(content.chunk);
                    self.collect_lights();
                    self.begin_meshing(SmallVec::from(Self::SUBREGION_POSES));
                    actors = Some(content.actors);
                }

                Err(error) => self.retry_gen(&error),
            }
        }

        actors
    }

    /// Applies the completed meshes and reports whether meshing finished and
    /// whether any completed mesh changed the geometry of the region.
    pub fn poll_meshes(&mut self, canvas: &Canvas) -> (bool, bool) {
        let mut changed = false;

        for result in self.mesh_rx.try_iter() {
            if result.lod != self.lod {
                continue;
            }

            self.meshing = self.meshing.saturating_sub(1);

            // A mesh that a later change already superseded must not replace the mesh of
            // the newer blocks.
            if match result.pos {
                Some(pos) => *self.mesh_revisions.get(pos) == result.revision,
                None => *self.mesh_revisions.get(U8Vec3::ZERO) == result.revision,
            } {
                self.model.receive(result);
                changed = true;
            }
        }

        if self.meshing == 0 {
            self.model.update(canvas, self.lod);
            (true, changed)
        } else {
            (false, changed)
        }
    }

    /// Takes the report of the finished save, if there is one.
    ///
    /// `None` means the save is still running, `Some(true)` that the blocks
    /// were written, and `Some(false)` that writing failed.
    pub fn poll_save(&mut self) -> Option<bool> {
        let mut outcome = None;

        for result in self.save_rx.try_iter() {
            outcome = Some(match result {
                Ok(()) => true,
                Err(e) => {
                    error!("failed to save region {}: {}", self.pos, e);
                    false
                }
            });
        }

        outcome
    }

    pub fn pos(&self) -> RegionPos {
        self.pos
    }

    /// Reads the blocks of `pos`, or `None` when the region has no save.
    fn read_from(pos: RegionPos) -> anyhow::Result<Option<BlockChunk>> {
        let path = region_path(&CHUNK_DIR, pos);
        let chunk: Option<BlockChunk> = util::read_file(&path, CHUNK_MAGIC, CHUNK_VERSION)?;

        if let Some(chunk) = &chunk {
            let expected = U8Vec3::splat(REGION_SIZE + 2);

            if chunk.size != expected {
                return Err(anyhow!(
                    "region file holds a {} chunk instead of a {expected} one",
                    chunk.size
                ));
            }
        }

        Ok(chunk)
    }

    /// Writes the blocks of `chunk`.
    fn write_to(pos: RegionPos, chunk: &BlockChunk) -> anyhow::Result<()> {
        util::write_file(
            &region_path(&CHUNK_DIR, pos),
            CHUNK_MAGIC,
            CHUNK_VERSION,
            chunk,
        )
    }
}

/// The rectangles of one region, which each pass draws in a single call.
///
/// The mesher reports the rectangles of one sub-region at a time at the finest
/// level of detail, and those of the whole region above it. Since a pass draws the
/// region in one call, the rectangles of the level that is drawn are concatenated
/// into the buffer of that call, which is rebuilt when the mesher reports new
/// ones.
#[derive(Clone)]
pub struct RegionModel {
    /// The rectangles of each sub-region of the finest level of detail, once the
    /// mesher has reported them.
    near: Option<Volume<ChunkRects>>,
    /// The rectangles of the whole region, which is one mesh above the finest
    /// level of detail.
    far: Option<ChunkRects>,
    /// The buffer the passes draw, which holds the rectangles of the level that
    /// is drawn.
    model: Option<ChunkModel>,
    /// Whether [`Self::model`] holds the level above the finest one.
    on_far: bool,
}

/// The rectangles of one kind of a region, as the passes draw them: one quad each, cut out
/// of the template of the rectangle.
#[derive(Clone)]
pub struct ChunkModel {
    blocks: BufferVec<BlockRectInst>,
    translucent: BufferVec<BlockRectInst>,
    occlusion: BufferVec<BlockRectInst>,
    optical: BufferVec<BlockRectInst>,
}

/// The rectangles of a chunk as the mesher reports them, which the buffer of a region is
/// concatenated from.
#[derive(Clone, Default)]
struct ChunkRects {
    blocks: Vec<BlockRectInst>,
    translucent: Vec<BlockRectInst>,
    occlusion: Vec<BlockRectInst>,
    optical: Vec<BlockRectInst>,
}

impl From<MeshingResult> for ChunkRects {
    fn from(result: MeshingResult) -> Self {
        Self {
            blocks: result.blocks,
            translucent: result.translucent,
            occlusion: result.occlusion,
            optical: result.optical,
        }
    }
}

impl ChunkRects {
    /// Appends the rectangles of `other`, which is how the rectangles of a level are
    /// collected into the buffer of one call.
    fn extend_from(&mut self, other: &Self) {
        self.blocks.extend_from_slice(&other.blocks);
        self.translucent.extend_from_slice(&other.translucent);
        self.occlusion.extend_from_slice(&other.occlusion);
        self.optical.extend_from_slice(&other.optical);
    }
}

impl ChunkModel {
    /// Creates the buffers of a region, which hold the rectangles of its blocks and of
    /// its occlusion.
    fn new(canvas: &Canvas, rects: &ChunkRects) -> Self {
        Self {
            blocks: BufferVec::instance(canvas, "block_rect", BufferInit::Content(&rects.blocks)),
            translucent: BufferVec::instance(
                canvas,
                "translucent_rect",
                BufferInit::Content(&rects.translucent),
            ),
            occlusion: BufferVec::instance(
                canvas,
                "occlusion_rect",
                BufferInit::Content(&rects.occlusion),
            ),
            optical: BufferVec::instance(
                canvas,
                "optical_rect",
                BufferInit::Content(&rects.optical),
            ),
        }
    }

    /// Replaces the rectangles, which only reallocates the buffer that grew.
    fn set(&mut self, canvas: &Canvas, rects: &ChunkRects) {
        self.blocks.set_content(canvas, &rects.blocks);
        self.translucent.set_content(canvas, &rects.translucent);
        self.occlusion.set_content(canvas, &rects.occlusion);
        self.optical.set_content(canvas, &rects.optical);
    }
}

/// The rectangles of a region, which are drawn with the index buffer of a single quad and
/// the corners of the template every rectangle names.
pub struct RectModel<'a>(&'a BufferVec<BlockRectInst>);

impl Render<(), BlockRectInst> for RectModel<'_> {
    fn rendered(&self) -> Vec<RenderItem<'_, (), BlockRectInst>> {
        RenderItem::new(&EMPTY_BUFFER_VEC, &QUAD_INDEX_BUFFER_VEC, self.0)
            .ok()
            .into_iter()
            .collect()
    }
}

impl RegionModel {
    /// The rectangles of the blocks of the region, which one call draws, or `None` while
    /// the region has none of them to draw.
    pub fn blocks(&self) -> Option<RectModel<'_>> {
        self.model
            .as_ref()
            .filter(|model| model.blocks.length > 0)
            .map(|model| RectModel(&model.blocks))
    }

    /// The camera-visible rectangles of translucent materials.
    pub fn translucent(&self) -> Option<RectModel<'_>> {
        self.model
            .as_ref()
            .filter(|model| model.translucent.length > 0)
            .map(|model| RectModel(&model.translucent))
    }

    /// Every translucent face needed by the shadow transmission pass.
    pub fn optical(&self) -> Option<RectModel<'_>> {
        self.model
            .as_ref()
            .filter(|model| model.optical.length > 0)
            .map(|model| RectModel(&model.optical))
    }

    /// The rectangles of the occlusion of the region, which a pass of its own draws over
    /// the blocks, or `None` while the region has none of them to draw.
    pub fn occlusion(&self) -> Option<RectModel<'_>> {
        self.model
            .as_ref()
            .filter(|model| model.occlusion.length > 0)
            .map(|model| RectModel(&model.occlusion))
    }

    fn new(lod: u8) -> Self {
        Self {
            near: None,
            far: None,
            model: None,
            on_far: lod > 0,
        }
    }

    /// Stores the rectangles the mesher reported and rebuilds the buffer of the level the
    /// region draws from them, and reports whether the region has finished, which is when
    /// it draws the level it asks for and none of the `outstanding` meshes of that level
    /// is still on its way.
    ///
    /// The buffer is rebuilt from every rectangle of the level, so a region that gets
    /// several meshes at once is still one concatenation. The level a region switches to
    /// waits until its meshes are all there, so that the mesh of the level it leaves
    /// behind keeps being drawn instead of popping.
    fn receive(&mut self, result: MeshingResult) {
        let pos = result.pos;
        let rects = ChunkRects::from(result);

        if let Some(pos) = pos {
            self.near
                .get_or_insert_with(|| Volume::new(U8Vec3::splat(SUBREGION_COUNT)))
                .set(pos, rects);
        } else {
            self.far = Some(rects);
        }
    }

    fn update(&mut self, canvas: &Canvas, lod: u8) {
        let is_far = lod > 0;

        if self.on_far != is_far {
            self.on_far = is_far;

            if is_far {
                self.near = None;
            } else {
                self.far = None;
            }
        }

        let rects = if self.on_far {
            self.far.take()
        } else {
            self.near.as_ref().map(|near| {
                let mut rects = ChunkRects::default();
                for chunk in near.vec.iter() {
                    rects.extend_from(chunk);
                }
                rects
            })
        }
        .filter(|r| {
            !r.blocks.is_empty()
                || !r.translucent.is_empty()
                || !r.occlusion.is_empty()
                || !r.optical.is_empty()
        });

        if let Some(rects) = rects {
            if let Some(model) = &mut self.model {
                model.set(canvas, &rects);
            } else {
                self.model = Some(ChunkModel::new(canvas, &rects));
            }
        }
    }
}
