//! Helpers for a world generation stage. They wrap the raw ABI in the
//! `block-game-abi` crate so that a stage reads like ordinary Rust.

use block_game_abi::worldgen::*;
use glam::U8Vec3;
use rand_pcg::Pcg64Mcg;
use rand_pcg::rand_core::Rng;
use serde::de::DeserializeOwned;
use std::slice;

pub use block_game_abi::worldgen as abi;

const REGION_SIZE: u8 = 32;
const SAMPLE_INTERVAL: u8 = 8;
const GRID_SIZE: u8 = REGION_SIZE / SAMPLE_INTERVAL;

/// One named run of components inside a sample.
#[derive(Clone, Copy, Debug)]
pub struct Channel {
    pub offset: usize,
    pub components: usize,
}

/// The channels of a sample, in the order their components are stored.
pub struct SampleLayout {
    channels: Vec<(String, Channel)>,
    stride: usize,
}

impl SampleLayout {
    fn from_raw(raw: Slice) -> Self {
        let descs = unsafe {
            slice::from_raw_parts(
                raw.ptr as *const ChannelDesc,
                raw.len as usize / std::mem::size_of::<ChannelDesc>(),
            )
        };
        let mut channels = Vec::with_capacity(descs.len());
        let mut stride = 0;

        for desc in descs {
            let name = str::from_utf8(bytes(desc.name)).unwrap_or("").to_owned();
            let channel = Channel {
                offset: desc.offset as usize,
                components: desc.components as usize,
            };
            stride = stride.max(channel.offset + channel.components);
            channels.push((name, channel));
        }

        Self { channels, stride }
    }

    pub fn channel(&self, name: &str) -> Option<Channel> {
        self.channels
            .iter()
            .find(|(entry, _)| entry == name)
            .map(|(_, channel)| *channel)
    }

    pub fn stride(&self) -> usize {
        self.stride
    }
}

/// One sample of the field.
#[derive(Clone, Copy)]
pub struct Sample {
    values: [f32; MAX_SAMPLE_FLOATS as usize],
    stride: usize,
}

impl Sample {
    fn new(stride: usize) -> Self {
        Self {
            values: [0.0; MAX_SAMPLE_FLOATS as usize],
            stride,
        }
    }

    pub fn scalar(&self, channel: Channel) -> f32 {
        self.values[channel.offset]
    }

    pub fn component(&self, channel: Channel, component: usize) -> f32 {
        self.values[channel.offset + component]
    }

    pub fn vector(&self, channel: Channel) -> glam::Vec3 {
        glam::Vec3::new(
            self.values[channel.offset],
            self.values[channel.offset + 1],
            self.values[channel.offset + 2],
        )
    }

    pub fn values(&self) -> &[f32] {
        &self.values[..self.stride]
    }
}

/// The coarse grids a stage reads, in the coordinates of the task view.
pub struct GridView<'a> {
    grids: &'a [GridDesc],
    layout: SampleLayout,
    lod: u8,
    halo_density: Option<usize>,
    halo_erosion: Option<usize>,
}

impl<'a> GridView<'a> {
    fn new(raw: &'a GridViewAbi, info: &StageInfo) -> Self {
        let grids = unsafe {
            slice::from_raw_parts(
                raw.grids.ptr as *const GridDesc,
                raw.grids.len as usize / std::mem::size_of::<GridDesc>(),
            )
        };
        let layout = SampleLayout::from_raw(info.channels);
        let halo_density = (info.halo_density >= 0).then_some(info.halo_density as usize);
        let halo_erosion = (info.halo_erosion >= 0).then_some(info.halo_erosion as usize);

        Self {
            grids,
            layout,
            lod: raw.lod as u8,
            halo_density,
            halo_erosion,
        }
    }

    pub fn channel(&self, name: &str) -> Option<Channel> {
        self.layout.channel(name)
    }

    pub fn sample(&self, local: U8Vec3, halo: bool, chunk_origin: U8Vec3, stride: u8) -> Sample {
        let block_size = 1u8 << self.lod;
        let pos = chunk_origin + local * block_size;
        let halo = halo && stride != 0;
        let origin = pos / SAMPLE_INTERVAL;
        let offset = (block_size / SAMPLE_INTERVAL).max(1);
        let corners = [
            U8Vec3::new(0, 0, 0),
            U8Vec3::new(0, 0, 1),
            U8Vec3::new(0, 1, 0),
            U8Vec3::new(0, 1, 1),
            U8Vec3::new(1, 0, 0),
            U8Vec3::new(1, 0, 1),
            U8Vec3::new(1, 1, 0),
            U8Vec3::new(1, 1, 1),
        ];
        let side = (self.grids.len() as f64).cbrt().round() as u8;
        let mut samples = [[0.0f32; MAX_SAMPLE_FLOATS as usize]; 8];

        for (i, &corner) in corners.iter().enumerate() {
            let sample_pos = origin + corner * offset;
            let grid_pos = (sample_pos / GRID_SIZE).min(U8Vec3::splat(side.saturating_sub(1)));
            let sample_pos = sample_pos - grid_pos * GRID_SIZE;
            let grid = &self.grids[grid_index(grid_pos, side)];
            let values = unsafe {
                slice::from_raw_parts(
                    grid.data.ptr as *const f32,
                    grid.data.len as usize / std::mem::size_of::<f32>(),
                )
            };
            let start = grid_index(sample_pos, grid.dims[0] as u8) * self.layout.stride();
            samples[i][..self.layout.stride()]
                .copy_from_slice(&values[start..start + self.layout.stride()]);
        }

        let redirection = match (halo, self.halo_density, self.halo_erosion) {
            (true, Some(density), Some(erosion)) => {
                let mut min = 1.0;
                let mut min_corner = U8Vec3::ZERO;

                for (&corner, sample) in corners.iter().zip(samples.iter()) {
                    let d = sample[density].min(-sample[erosion]);
                    if d < min {
                        min = d;
                        min_corner = corner;
                    }
                }

                min_corner * (block_size - 1)
            }
            _ => U8Vec3::splat(block_size / 2),
        };
        let rate =
            (pos % SAMPLE_INTERVAL + redirection).as_vec3() / (offset * SAMPLE_INTERVAL) as f32;

        let mut value = Sample::new(self.layout.stride());

        for (&corner, sample) in corners.iter().zip(samples.iter()) {
            let weight = (rate - (1 - corner).as_vec3()).abs().element_product();

            for i in 0..self.layout.stride() {
                value.values[i] += sample[i] * weight;
            }
        }

        value
    }
}

type GridViewAbi = block_game_abi::worldgen::GridView;

fn grid_index(pos: U8Vec3, side: u8) -> usize {
    pos.x as usize * (side as usize).pow(2) + pos.y as usize * side as usize + pos.z as usize
}

fn bytes(slice: Slice) -> &'static [u8] {
    unsafe { slice::from_raw_parts(slice.ptr as *const u8, slice.len as usize) }
}

/// The blocks a stage may name.
pub struct BlockTable<'a> {
    entries: &'a [BlockEntry],
}

impl<'a> BlockTable<'a> {
    fn new(raw: Slice) -> Self {
        let entries = unsafe {
            slice::from_raw_parts(
                raw.ptr as *const BlockEntry,
                raw.len as usize / std::mem::size_of::<BlockEntry>(),
            )
        };
        Self { entries }
    }

    pub fn meta(&self, name: &str) -> u16 {
        self.entries
            .iter()
            .find(|entry| str::from_utf8(bytes(entry.name)) == Ok(name))
            .map(|entry| entry.meta)
            .unwrap_or(0)
    }
}

/// A stage that writes the coarse samples of one region.
pub struct SampleCtx<'a> {
    raw: &'a SampleCtxAbi,
    layout: SampleLayout,
    blocks: BlockTable<'a>,
}

type SampleCtxAbi = block_game_abi::worldgen::SampleCtx;

impl<'a> SampleCtx<'a> {
    pub unsafe fn new(raw: &'a SampleCtxAbi) -> Self {
        Self {
            raw,
            layout: SampleLayout::from_raw(raw.info.channels),
            blocks: BlockTable::new(raw.info.blocks),
        }
    }

    pub fn layout(&self) -> &SampleLayout {
        &self.layout
    }

    pub fn channel(&self, name: &str) -> Option<Channel> {
        self.layout.channel(name)
    }

    pub fn params<T: DeserializeOwned>(&self) -> Option<T> {
        ron::from_str(str::from_utf8(bytes(self.raw.info.params)).ok()?).ok()
    }

    pub fn block(&self, name: &str) -> u16 {
        self.blocks.meta(name)
    }

    pub fn dims(&self) -> U8Vec3 {
        U8Vec3::new(
            self.raw.dims[0] as u8,
            self.raw.dims[1] as u8,
            self.raw.dims[2] as u8,
        )
    }

    pub fn origin(&self) -> glam::IVec3 {
        glam::IVec3::from(self.raw.origin)
    }

    pub fn world_pos(&self, pos: U8Vec3) -> glam::IVec3 {
        self.origin() + pos.as_ivec3() * self.raw.spacing as i32
    }

    pub fn positions(&self) -> impl Iterator<Item = U8Vec3> {
        let dims = self.dims();
        (0..dims.x).flat_map(move |x| {
            (0..dims.y).flat_map(move |y| (0..dims.z).map(move |z| U8Vec3::new(x, y, z)))
        })
    }

    pub fn set(&mut self, channel: Channel, pos: U8Vec3, component: usize, value: f32) {
        let stride = self.layout.stride();
        let dims = self.raw.dims[0] as u8;
        let index = grid_index(pos, dims) * stride;
        let grid = self.grid_mut();
        grid[index + channel.offset + component] = value;
    }

    pub fn set_channel(&mut self, channel: Channel, pos: U8Vec3, values: &[f32]) {
        let stride = self.layout.stride();
        let dims = self.raw.dims[0] as u8;
        let index = grid_index(pos, dims) * stride + channel.offset;
        let grid = self.grid_mut();
        grid[index..index + values.len()].copy_from_slice(values);
    }

    fn grid_mut(&mut self) -> &mut [f32] {
        unsafe {
            slice::from_raw_parts_mut(
                self.raw.grid.ptr as *mut f32,
                self.raw.grid.len as usize / std::mem::size_of::<f32>(),
            )
        }
    }
}

/// A stage that writes or edits the blocks of one chunk.
pub struct ChunkCtx<'a> {
    raw: &'a ChunkCtxAbi,
    view: GridView<'a>,
    blocks: BlockTable<'a>,
}

type ChunkCtxAbi = block_game_abi::worldgen::ChunkCtx;

impl<'a> ChunkCtx<'a> {
    pub unsafe fn new(raw: &'a ChunkCtxAbi) -> Self {
        Self {
            raw,
            view: GridView::new(&raw.view, &raw.info),
            blocks: BlockTable::new(raw.info.blocks),
        }
    }

    pub fn params<T: DeserializeOwned>(&self) -> Option<T> {
        ron::from_str(str::from_utf8(bytes(self.raw.info.params)).ok()?).ok()
    }

    pub fn block(&self, name: &str) -> u16 {
        self.blocks.meta(name)
    }

    pub fn channel(&self, name: &str) -> Option<Channel> {
        self.view.channel(name)
    }

    pub fn stride(&self) -> u8 {
        self.raw.stride as u8
    }

    pub fn positions(&self) -> impl Iterator<Item = U8Vec3> {
        let stride = self.stride();
        (0..stride).flat_map(move |x| {
            (0..stride).flat_map(move |y| (0..stride).map(move |z| U8Vec3::new(x, y, z)))
        })
    }

    pub fn get(&self, pos: U8Vec3) -> u16 {
        let chunk = self.chunk();
        chunk[grid_index(pos, self.stride())]
    }

    pub fn set(&mut self, pos: U8Vec3, meta: u16) {
        let stride = self.stride();
        let chunk = unsafe {
            slice::from_raw_parts_mut(
                self.raw.chunk.ptr as *mut u16,
                self.raw.chunk.len as usize / std::mem::size_of::<u16>(),
            )
        };
        chunk[grid_index(pos, stride)] = meta;
    }

    pub fn view(&self) -> &GridView<'a> {
        &self.view
    }

    pub fn sample(&self, local: U8Vec3) -> Sample {
        let block_size = 1u8 << self.raw.lod;
        let halo = local.min_element() == 0 || local.max_element() == self.stride() - 1;
        self.view.sample(
            local,
            halo,
            U8Vec3::splat(REGION_SIZE - block_size),
            self.stride(),
        )
    }

    fn chunk(&self) -> &[u16] {
        unsafe {
            slice::from_raw_parts(
                self.raw.chunk.ptr as *const u16,
                self.raw.chunk.len as usize / std::mem::size_of::<u16>(),
            )
        }
    }
}

/// A stage that decides where a structure may be placed.
pub struct PlaceCtx<'a> {
    raw: &'a PlaceCtxAbi,
    view: GridView<'a>,
    rng: Pcg64Mcg,
    sites: usize,
}

type PlaceCtxAbi = block_game_abi::worldgen::PlaceCtx;

impl<'a> PlaceCtx<'a> {
    pub unsafe fn new(raw: &'a PlaceCtxAbi) -> Self {
        Self {
            raw,
            view: GridView::new(&raw.view, &raw.info),
            rng: {
                use rand_pcg::rand_core::SeedableRng;
                Pcg64Mcg::seed_from_u64(raw.info.rng_seed)
            },
            sites: 0,
        }
    }

    pub fn params<T: DeserializeOwned>(&self) -> Option<T> {
        ron::from_str(str::from_utf8(bytes(self.raw.info.params)).ok()?).ok()
    }

    pub fn structure(&self) -> &StructureInfo {
        &self.raw.structure
    }

    pub fn channel(&self, name: &str) -> Option<Channel> {
        self.view.channel(name)
    }

    pub fn random_site(&mut self) -> U8Vec3 {
        U8Vec3::new(
            (self.rng.next_u32() % REGION_SIZE as u32) as u8,
            (self.rng.next_u32() % REGION_SIZE as u32) as u8,
            (self.rng.next_u32() % REGION_SIZE as u32) as u8,
        )
    }

    pub fn sample(&self, local: U8Vec3) -> Sample {
        self.view.sample(local, false, U8Vec3::ZERO, 0)
    }

    pub fn push_site(&mut self, pos: U8Vec3) -> bool {
        if self.sites >= self.raw.capacity as usize {
            return false;
        }

        let candidates = unsafe {
            slice::from_raw_parts_mut(
                self.raw.candidates.ptr as *mut Candidate,
                self.raw.candidates.len as usize / std::mem::size_of::<Candidate>(),
            )
        };
        candidates[self.sites] = Candidate {
            pos: pos.to_array().map(u32::from),
            flags: 0,
        };
        self.sites += 1;
        true
    }

    pub fn sites(&self) -> u32 {
        self.sites as u32
    }
}
