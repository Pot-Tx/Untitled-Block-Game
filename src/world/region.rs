use crate::render::{
    AlphaVertex, Canvas, Geometry, InstGroup, Instances, IntTransInst, NormTexVertex, Render,
    RenderItem,
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

pub type LocalPos = U8Vec3;
pub type SubRegionPos = U8Vec3;
pub const REGION_SIZE: u8 = 32;
pub const SUBREGION_SIZE: u8 = 16;
pub const SUBREGION_STRIDE: u8 = SUBREGION_SIZE + 2;
pub const SUBREGION_COUNT: u8 = REGION_SIZE / SUBREGION_SIZE;
pub const MAX_LOD: u8 = REGION_SIZE.ilog2() as u8;

#[derive(Clone)]
pub struct Region {
    pos: RegionPos,
    lod: u8,
    chunk: Option<Chunk>,
    model: RegionModel,
    generating: bool,
    meshing: u8,
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

    #[inline]
    pub const fn block_size_on_lod(lod: u8) -> u8 {
        1 << lod
    }

    #[inline]
    pub const fn size_on_lod(lod: u8) -> u8 {
        REGION_SIZE >> lod
    }

    #[inline]
    pub const fn stride_on_lod(lod: u8) -> u8 {
        (REGION_SIZE >> lod) + 2
    }

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

    pub fn bound(&self) -> AABB<Vec3> {
        AABB {
            min: (self.pos * REGION_SIZE as i32).as_vec3(),
            max: ((self.pos + 1) * REGION_SIZE as i32).as_vec3(),
        }
    }

    pub fn get_block(&self, pos: LocalPos) -> Block {
        match &self.chunk {
            None => Block::air(),
            Some(chunk) => Block::from_meta(*chunk.get(pos + 1)),
        }
    }

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
                "Failed to send Generation Task from Region {}: {}",
                self.pos, e
            ),
        }
    }

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
                                "Failed to send Meshing Task from Region {}: {}",
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
                            "Failed to send Meshing Task from Region {}: {}",
                            self.pos, e
                        ),
                    }
                }
            }
        }
    }

    pub fn update(&mut self, lod: u8) -> bool {
        if self.lod == lod || self.changed.is_some() {
            false
        } else {
            self.lod = lod;
            self.begin_generation();
            true
        }
    }

    pub fn poll(&mut self) -> bool {
        if self.generating
            && let Ok(result) = self.chunk_rx.try_recv()
        {
            if result.lod == self.lod {
                self.chunk = Some(result.chunk);
                self.generating = false;
            }
        }

        if !self.generating {
            self.begin_meshing(SmallVec::from(Self::SUBREGION_POSES));
            true
        } else {
            false
        }
    }

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

    pub fn save(&mut self) -> Result<()> {
        if let Some(changed) = self.changed
            && changed
        {
            let mut file = File::create(&format!(
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
                return Err(anyhow!("no chunk found in region"));
            }
        }

        Ok(())
    }

    pub fn load(&mut self) -> Result<()> {
        let mut file = File::open(&format!(
            "saves/{}.{}.{}.regn",
            self.pos.x, self.pos.y, self.pos.z
        ))?;

        let mut magic_data = [0; 4];
        file.read_exact(&mut magic_data)?;
        let magic = str::from_utf8(&magic_data).unwrap_or("ERROR");

        let mut version_data = [0; 4];
        file.read_exact(&mut version_data)?;
        let version = u32::from_le_bytes(version_data);

        if magic != "REGN" || version != 0 {
            return Err(anyhow!("file invalid"));
        }

        let mut block_data = Vec::new();
        file.read_to_end(&mut block_data)?;

        if block_data.len() % 4 != 0 {
            return Err(anyhow!("file corrupted"));
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

#[derive(Clone)]
pub struct RegionModel {
    near: Option<Volume<SwapPair<ChunkModel>>>,
    far: Option<SwapPair<ChunkModel>>,
    pos: Instances<IntTransInst>,
    on_far: bool,
}

#[derive(Clone)]
pub struct ChunkModel {
    blocks: Option<Geometry<NormTexVertex>>,
    occlusion: Option<Geometry<AlphaVertex>>,
}

impl Render<NormTexVertex, IntTransInst> for RegionModel {
    fn rendered(&self) -> Vec<RenderItem<'_, NormTexVertex, IntTransInst>> {
        let mut items = Vec::new();

        match self.on_far {
            false => {
                if let Some(models) = &self.near {
                    for pair in models.vec.iter() {
                        if let Some(model) = pair.get()
                            && let Some(blocks) = &model.blocks
                        {
                            items.push(RenderItem {
                                geometry: blocks,
                                instances: &self.pos,
                            });
                        }
                    }
                }
            }

            true => {
                if let Some(pair) = &self.far {
                    if let Some(model) = pair.get()
                        && let Some(blocks) = &model.blocks
                    {
                        items.push(RenderItem {
                            geometry: blocks,
                            instances: &self.pos,
                        });
                    }
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
                            && let Some(occlusion) = &model.occlusion
                        {
                            items.push(RenderItem {
                                geometry: occlusion,
                                instances: &self.pos,
                            });
                        }
                    }
                }
            }

            true => {
                if let Some(pair) = &self.far {
                    if let Some(model) = pair.get()
                        && let Some(occlusion) = &model.occlusion
                    {
                        items.push(RenderItem {
                            geometry: occlusion,
                            instances: &self.pos,
                        });
                    }
                }
            }
        }

        items
    }
}

impl RegionModel {
    fn new(canvas: &Canvas, pos: RegionPos, lod: u8) -> Self {
        Self {
            near: None,
            far: None,
            pos: [IntTransInst {
                pos: pos * REGION_SIZE as i32,
            }]
            .instances(canvas, "region"),
            on_far: lod > 0,
        }
    }

    fn update(&mut self, canvas: &Canvas, result: MeshingResult) {
        let model = ChunkModel {
            blocks: if result.blocks.is_empty() {
                None
            } else {
                Some(result.blocks.geometry(canvas, "block"))
            },
            occlusion: if result.occlusion.is_empty() {
                None
            } else {
                Some(result.occlusion.geometry(canvas, "occlusion"))
            },
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
