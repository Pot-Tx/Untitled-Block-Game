use crate::render::{
    AlphaVertex, BufferInit, BufferVec, Canvas, IntTransInst, NormTexVertex, Render, RenderItem,
};
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::util::SwapPair;
use crate::world::block::Meta;
use crate::world::generation::*;
use crate::world::model::MeshingTask;
use crate::world::{Block, Chunk, MeshingResult, RegionPos};
use anyhow::{anyhow, Result};
use crossbeam_channel::*;
use glam::{U8Vec3, Vec3};
use log::error;
use smallvec::SmallVec;
use std::fs::File;
use std::io::{Read, Write};

/// Position of a block inside a region, without the halo.
pub type LocalPos = U8Vec3;
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

/// One region of the world: the blocks, the model it is drawn with, and the
/// state of its loading, generation and meshing.
#[derive(Clone)]
pub struct Region {
    pos: RegionPos,
    /// The level of detail the chunk is currently stored at.
    lod: u8,
    /// The blocks of the region, or `None` while it is still being generated.
    chunk: Option<Chunk>,
    model: RegionModel,
    generating: bool,
    /// Number of meshing tasks that have not returned yet.
    meshing: u8,
    /// Whether the chunk has to be written back to disk: `Some(true)` after a
    /// change, `Some(false)` once it has been saved, and `None` while the chunk
    /// has not been loaded or generated yet.
    changed: Option<bool>,

    gen_tx: Sender<GenTask>,
    chunk_tx: Sender<GenResult>,
    chunk_rx: Receiver<GenResult>,
    meshing_tx: Sender<MeshingTask>,
    mesh_tx: Sender<MeshingResult>,
    mesh_rx: Receiver<MeshingResult>,
}

impl Render<NormTexVertex, IntTransInst> for Region {
    fn rendered(&self) -> Vec<RenderItem<'_, NormTexVertex, IntTransInst>> {
        self.model.rendered()
    }
}

impl Render<AlphaVertex, IntTransInst> for Region {
    fn rendered(&self) -> Vec<RenderItem<'_, AlphaVertex, IntTransInst>> {
        self.model.rendered()
    }
}

impl Region {
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

    /// Creates a region, loading it from disk or starting its generation.
    pub fn new(
        canvas: &Canvas,
        pos: RegionPos,
        lod: u8,
        gen_tx: Sender<GenTask>,
        meshing_tx: Sender<MeshingTask>,
    ) -> Self {
        let (chunk_tx, chunk_rx) = bounded(1);
        let (mesh_tx, mesh_rx) = bounded(8);
        let mut new = Self {
            pos,
            lod,
            chunk: None,
            model: RegionModel::new(canvas, pos, lod),
            generating: false,
            meshing: 0,
            changed: None,

            gen_tx,
            chunk_tx,
            chunk_rx,
            meshing_tx,
            mesh_tx,
            mesh_rx,
        };

        if new.load().is_err() {
            new.begin_generation();
        }
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

    /// The sub-regions whose chunk has to be remeshed when the block at `pos`
    /// changes.
    ///
    /// The mesher reads the blocks around a sub-region, so a block on the border
    /// also changes the mesh of the neighbouring sub-regions.
    #[inline]
    fn pos_influence(pos: U8Vec3) -> SmallVec<[SubRegionPos; SUBREGION_COUNT.pow(3) as usize]> {
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

    /// The box the region covers, in world coordinates.
    pub fn bound(&self) -> AABB<Vec3> {
        AABB {
            min: (self.pos * REGION_SIZE as i32).as_vec3(),
            max: ((self.pos + 1) * REGION_SIZE as i32).as_vec3(),
        }
    }

    /// The block at `pos`, which is region local and does not include the halo
    /// of the chunk.
    pub fn get_block(&self, pos: LocalPos) -> Block {
        match &self.chunk {
            None => Block::air(),
            Some(chunk) => Block::from_meta(*chunk.get(pos + 1)),
        }
    }

    /// Sets the block at `pos` and queues the affected sub-regions for
    /// remeshing.
    ///
    /// Returns whether the region has a chunk to write to.
    pub fn set_block(&mut self, pos: LocalPos, block: Block) -> bool {
        match &mut self.chunk {
            None => false,

            Some(chunk) => {
                chunk.set(pos, block.to_meta());
                self.changed = Some(true);
                self.begin_meshing(Self::pos_influence(pos));
                true
            }
        }
    }

    /// Asks the generator for the chunk of this region.
    fn begin_generation(&mut self) {
        match self.gen_tx.try_send(GenTask {
            pos: self.pos,
            lod: self.lod,
            tx: self.chunk_tx.clone(),
        }) {
            Ok(_) => {
                self.generating = true;
                self.meshing = 0;
            }

            Err(e) => error!(
                "failed to send generation task from region {}: {}",
                self.pos, e
            ),
        }
    }

    /// Asks the mesher for the meshes of `poses`.
    ///
    /// At level 0 every sub-region is meshed on its own, so that a change only
    /// remeshes the sub-regions around it. Coarser levels collapse the region
    /// into a few blocks, so the whole region is meshed in one task.
    fn begin_meshing(&mut self, poses: SmallVec<[SubRegionPos; SUBREGION_COUNT.pow(3) as usize]>) {
        match self.lod {
            0 => {
                if let Some(chunk) = &self.chunk {
                    for pos in poses {
                        match self.meshing_tx.try_send(MeshingTask {
                            lod: self.lod,
                            pos: Some(pos),
                            chunk: chunk
                                .part(pos * SUBREGION_SIZE, U8Vec3::splat(SUBREGION_STRIDE)),
                            tx: self.mesh_tx.clone(),
                        }) {
                            Ok(_) => self.meshing += 1,
                            Err(e) => error!(
                                "failed to send meshing task from region {}: {}",
                                self.pos, e
                            ),
                        }
                    }
                }
            }

            1.. => {
                if let Some(chunk) = self.chunk.take() {
                    match self.meshing_tx.try_send(MeshingTask {
                        lod: self.lod,
                        pos: None,
                        chunk,
                        tx: self.mesh_tx.clone(),
                    }) {
                        Ok(_) => self.meshing += 1,
                        Err(e) => error!(
                            "failed to send meshing task from region {}: {}",
                            self.pos, e
                        ),
                    }
                }
            }
        }
    }

    /// Switches the region to `lod`, restarting its generation.
    ///
    /// Returns whether the level changed; a region that has been edited by the
    /// player keeps its blocks and is never regenerated at another level.
    pub fn update(&mut self, lod: u8) -> bool {
        if self.lod == lod || self.changed.is_some() {
            false
        } else {
            self.lod = lod;
            self.begin_generation();
            true
        }
    }

    /// Takes the generated chunk when it arrived and starts meshing it.
    ///
    /// Returns whether the region has its chunk, so that the world can stop
    /// polling it.
    pub fn poll(&mut self) -> bool {
        if self.generating
            && let Ok(result) = self.chunk_rx.try_recv()
            && result.lod == self.lod
        {
            self.chunk = Some(result.chunk);
            self.generating = false;
        }

        if !self.generating {
            self.begin_meshing(SmallVec::from(Self::SUBREGION_POSES));
            true
        } else {
            false
        }
    }

    /// Uploads the meshes that finished and reports whether the model is ready
    /// to be drawn at the level of detail of the region.
    pub fn pre_render(&mut self, canvas: &Canvas) -> bool {
        if self.meshing > 0 {
            for result in self.mesh_rx.try_iter() {
                if result.lod == self.lod {
                    self.model.update(canvas, result);
                    self.meshing = self.meshing.saturating_sub(1);
                }
            }
        }

        self.meshing == 0 && self.model.poll(self.lod)
    }

    /// Writes the chunk to `saves/<x>.<y>.<z>.regn`.
    ///
    /// The file holds a magic, the version of the format, and then the blocks
    /// run length encoded as a meta value followed by how often it repeats.
    pub fn save(&mut self) -> Result<()> {
        if let Some(changed) = self.changed
            && changed
        {
            let mut file = File::create(format!(
                "saves/{}.{}.{}.regn",
                self.pos.x, self.pos.y, self.pos.z
            ))?;

            file.write_all(b"REGN")?;

            file.write_all(&0u32.to_le_bytes())?;

            let mut block_data = Vec::new();
            let mut meta = Meta::MAX;
            let mut count: u16 = 0;

            if let Some(chunk) = &self.chunk {
                for &m in chunk.vec.iter() {
                    if m == meta && count < u16::MAX {
                        count += 1;
                    } else {
                        if count > 0 {
                            block_data.extend_from_slice(&meta.to_le_bytes());
                            block_data.extend_from_slice(&count.to_le_bytes());
                        }

                        meta = m;
                        count = 1;
                    }
                }

                if count > 0 {
                    block_data.extend_from_slice(&meta.to_le_bytes());
                    block_data.extend_from_slice(&count.to_le_bytes());
                }

                file.write_all(&block_data)?;
                file.sync_all()?;

                self.changed = Some(false);
            } else {
                return Err(anyhow!("region {} has no chunk to save", self.pos));
            }
        }

        Ok(())
    }

    /// Reads the chunk back from `saves/<x>.<y>.<z>.regn`.
    pub fn load(&mut self) -> Result<()> {
        let mut file = File::open(format!(
            "saves/{}.{}.{}.regn",
            self.pos.x, self.pos.y, self.pos.z
        ))?;

        let mut magic_data = [0; 4];
        file.read_exact(&mut magic_data)?;
        let magic = str::from_utf8(&magic_data).unwrap_or("ERROR");

        let mut version_data = [0; 4];
        file.read_exact(&mut version_data)?;
        let version = u32::from_le_bytes(version_data);

        if magic != "REGN" {
            return Err(anyhow!("region file has an invalid magic"));
        }

        if version != 0 {
            return Err(anyhow!(
                "region file has an unsupported version {}",
                version
            ));
        }

        let mut block_data = Vec::new();
        file.read_to_end(&mut block_data)?;

        if block_data.len() % 4 != 0 {
            return Err(anyhow!("region file is corrupted"));
        }

        let mut blocks = Vec::new();

        for i in 0..block_data.len() / 4 {
            let j = i * 4;
            let meta = Meta::from_le_bytes(block_data[j..j + 2].try_into()?);
            let count = u16::from_le_bytes(block_data[j + 2..j + 4].try_into()?);
            blocks.resize(blocks.len() + count as usize, meta);
        }

        self.chunk = Some(Chunk {
            size: U8Vec3::splat(REGION_SIZE + 2),
            vec: blocks,
        });
        self.changed = Some(false);

        Ok(())
    }
}

/// The meshes of one region.
///
/// At level 0 the region holds one entry per sub-region, above that a single
/// entry for the whole region. Both are kept around while one of them is being
/// replaced, so that the mesh does not pop when a region changes its level of
/// detail; `on_far` selects which of the two is drawn.
#[derive(Clone)]
pub struct RegionModel {
    near: Option<Volume<SwapPair<ChunkModel>>>,
    far: Option<SwapPair<ChunkModel>>,
    pos: BufferVec<IntTransInst>,
    on_far: bool,
}

/// The buffers of one meshed chunk: the blocks and the occlusion mesh.
#[derive(Clone)]
pub struct ChunkModel {
    block_vertices: BufferVec<NormTexVertex>,
    block_indices: BufferVec<u16>,
    occlusion_vertices: BufferVec<AlphaVertex>,
    occlusion_indices: BufferVec<u16>,
}

impl Render<NormTexVertex, IntTransInst> for RegionModel {
    fn rendered(&self) -> Vec<RenderItem<'_, NormTexVertex, IntTransInst>> {
        let mut items = Vec::new();

        match self.on_far {
            false => {
                if let Some(models) = &self.near {
                    for pair in models.vec.iter() {
                        if let Some(model) = pair.get()
                            && let Ok(item) = RenderItem::new(
                                &model.block_vertices,
                                &model.block_indices,
                                &self.pos,
                            )
                        {
                            items.push(item);
                        }
                    }
                }
            }

            true => {
                if let Some(pair) = &self.far
                    && let Some(model) = pair.get()
                    && let Ok(item) =
                        RenderItem::new(&model.block_vertices, &model.block_indices, &self.pos)
                {
                    items.push(item);
                }
            }
        }

        items
    }
}

impl Render<AlphaVertex, IntTransInst> for RegionModel {
    fn rendered(&self) -> Vec<RenderItem<'_, AlphaVertex, IntTransInst>> {
        let mut items = Vec::new();

        match self.on_far {
            false => {
                if let Some(models) = &self.near {
                    for pair in models.vec.iter() {
                        if let Some(model) = pair.get()
                            && let Ok(item) = RenderItem::new(
                                &model.occlusion_vertices,
                                &model.occlusion_indices,
                                &self.pos,
                            )
                        {
                            items.push(item);
                        }
                    }
                }
            }

            true => {
                if let Some(pair) = &self.far
                    && let Some(model) = pair.get()
                    && let Ok(item) = RenderItem::new(
                        &model.occlusion_vertices,
                        &model.occlusion_indices,
                        &self.pos,
                    )
                {
                    items.push(item);
                }
            }
        }

        items
    }
}

impl RegionModel {
    /// Creates the instance buffer that places the region in the world.
    fn new(canvas: &Canvas, pos: RegionPos, lod: u8) -> Self {
        Self {
            near: None,
            far: None,
            pos: BufferVec::instance(
                canvas,
                "region",
                BufferInit::Content(&[IntTransInst {
                    pos: pos * REGION_SIZE as i32,
                }]),
            ),
            on_far: lod > 0,
        }
    }

    /// Stores the mesh of `result`, which becomes visible after a few frames so
    /// that buffers are not replaced while a recorded frame still uses them.
    fn update(&mut self, canvas: &Canvas, result: MeshingResult) {
        let model = ChunkModel {
            block_vertices: result.blocks.vertex_buffer(canvas, "block"),
            block_indices: result.blocks.index_buffer_vec(canvas, "block"),
            occlusion_vertices: result.occlusion.vertex_buffer(canvas, "occlusion"),
            occlusion_indices: result.occlusion.index_buffer_vec(canvas, "occlusion"),
        };

        match result.pos {
            None => self.far.get_or_insert_default().set(model, 4),
            Some(pos) => {
                self.near
                    .get_or_insert(Volume::new(U8Vec3::splat(SUBREGION_COUNT)))
                    .get_mut(pos)
                    .set(model, 4);
            }
        }
    }

    fn poll(&mut self, lod: u8) -> bool {
        let is_far = lod > 0;

        // Wait for the meshes of the new level before switching over and
        // dropping the meshes of the old one.
        let finished = match is_far {
            false => {
                if let Some(models) = &mut self.near {
                    let mut finished = true;
                    for pair in models.vec.iter_mut() {
                        finished &= pair.update();
                    }
                    finished
                } else {
                    true
                }
            }

            true => {
                if let Some(pair) = &mut self.far {
                    pair.update()
                } else {
                    true
                }
            }
        };

        if self.on_far != is_far && finished {
            self.on_far = is_far;
            if is_far {
                self.near = None;
            } else {
                self.far = None;
            }
        }

        finished
    }
}
