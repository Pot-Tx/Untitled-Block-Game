//! The wasm runtime that executes the stages of a world program.

use crate::game::AssetId;
use crate::world::RegionPos;
use crate::world::block::{BLOCK_TYPES, BlockChunk, BlockMeta};
use crate::world::generation::sample::{Channel, SampleGrid, SampleGridView, SampleLayout};
use crate::world::generation::structure::StructureAsset;
use crate::world::generation::{ChunkStage, PlaceStage, SampleStage};
use crate::world::region::REGION_SIZE;
use anyhow::anyhow;
use block_game_abi::worldgen::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;
use std::slice;
use std::thread;
use std::time::Duration;
use wasmtime::{
    Config, Engine, Instance, Memory, Module, OptLevel, Store, StoreLimits, StoreLimitsBuilder,
    TypedFunc,
};

/// How much memory one module instance may use.
const MEMORY_LIMIT: usize = 64 << 20;
/// How many epoch ticks a stage may run. The epoch is advanced every 100 ms, so
/// a stage that runs for ten seconds is interrupted.
const EPOCH_DEADLINE: u64 = 100;

struct HostState {
    limits: StoreLimits,
}

struct ThreadState {
    store: Store<HostState>,
    instance: Instance,
    memory: Memory,
    alloc: TypedFunc<u32, u32>,
    dealloc: Option<TypedFunc<(u32, u32), ()>>,
}

struct CompiledModule {
    module: Module,
}

/// The compiled stage modules of one world program.
pub struct ScriptRuntime {
    engine: Engine,
    modules: HashMap<String, CompiledModule>,
}

thread_local! {
    static STATES: RefCell<HashMap<String, ThreadState>> = RefCell::new(HashMap::new());
}

impl ScriptRuntime {
    pub fn new() -> anyhow::Result<Self> {
        let mut config = Config::new();
        config.cranelift_opt_level(OptLevel::Speed);
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(wasm)?;
        let ticker = engine.clone();
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_millis(100));
                ticker.increment_epoch();
            }
        });

        Ok(Self {
            engine,
            modules: HashMap::new(),
        })
    }

    pub fn prepare(
        &mut self,
        program: &crate::world::generation::WorldProgram,
    ) -> anyhow::Result<()> {
        for stage in &program.sampler {
            self.compile(&stage.module.asset, &stage.module.bytes)?;
        }

        for stage in &program.task_stages {
            match stage {
                crate::world::generation::TaskStage::Chunk(stage) => {
                    self.compile(&stage.module.asset, &stage.module.bytes)?;
                }
                crate::world::generation::TaskStage::Stamp { .. } => (),
            }
        }

        for stage in program.placements.iter().flatten() {
            self.compile(&stage.module.asset, &stage.module.bytes)?;
        }

        Ok(())
    }

    fn compile(&mut self, asset: &AssetId, bytes: &[u8]) -> anyhow::Result<()> {
        let key = asset.display();
        if self.modules.contains_key(&key) {
            return Ok(());
        }

        let module = Module::from_binary(&self.engine, bytes)
            .map_err(|error| anyhow!("failed to compile the stage module {key}: {error}"))?;
        self.modules.insert(key, CompiledModule { module });
        Ok(())
    }

    fn module(&self, asset: &AssetId) -> anyhow::Result<String> {
        let key = asset.display();
        if !self.modules.contains_key(&key) {
            return Err(anyhow!("the stage module {key} is not compiled"));
        }
        Ok(key)
    }

    fn with_state<R>(
        &self,
        key: &str,
        f: impl FnOnce(&mut ThreadState) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        STATES.with(|states| {
            let mut states = states.borrow_mut();

            if !states.contains_key(key) {
                let module = self
                    .modules
                    .get(key)
                    .map(|module| module.module.clone())
                    .ok_or_else(|| anyhow!("the stage module {key} is not compiled"))?;
                let mut store = Store::new(
                    &self.engine,
                    HostState {
                        limits: StoreLimitsBuilder::new()
                            .memory_size(MEMORY_LIMIT)
                            .build(),
                    },
                );
                store.limiter(|state| &mut state.limits);
                store.set_epoch_deadline(EPOCH_DEADLINE);
                let instance = Instance::new(&mut store, &module, &[]).map_err(wasm)?;
                let version = instance
                    .get_typed_func::<(), u32>(&mut store, "abi_version")
                    .map_err(wasm)?
                    .call(&mut store, ())
                    .map_err(wasm)?;

                if version != ABI_VERSION {
                    return Err(anyhow!(
                        "the stage module {key} uses ABI {version}, but this build uses {ABI_VERSION}"
                    ));
                }

                let memory = instance
                    .get_memory(&mut store, "memory")
                    .ok_or_else(|| anyhow!("the stage module {key} exports no memory"))?;
                let alloc = instance
                    .get_typed_func::<u32, u32>(&mut store, "alloc")
                    .map_err(wasm)?;
                let dealloc = instance
                    .get_typed_func::<(u32, u32), ()>(&mut store, "dealloc")
                    .ok();

                states.insert(
                    key.to_owned(),
                    ThreadState {
                        store,
                        instance,
                        memory,
                        alloc,
                        dealloc,
                    },
                );
            }

            f(states.get_mut(key).expect("the state was just inserted"))
        })
    }

    pub fn run_sample(
        &self,
        stage: &SampleStage,
        seed: u64,
        region: RegionPos,
        layout: &SampleLayout,
        grid: &mut SampleGrid,
    ) -> anyhow::Result<()> {
        let key = self.module(&stage.module.asset)?;

        self.with_state(&key, |state| {
            let mut arena = Arena::default();
            let info = build_info(
                &mut arena,
                seed,
                hash(seed, region, 0),
                region,
                0,
                layout,
                &stage.params,
            );
            let dims = grid.size;
            let grid_offset = arena.push(&floats_to_bytes(&grid.data));
            let origin = [
                region.x * REGION_SIZE as i32 - 8,
                region.y * REGION_SIZE as i32 - 8,
                region.z * REGION_SIZE as i32 - 8,
            ];
            let ctx_offset = arena.reserve(size_of::<SampleCtx>());
            let base = alloc_arena(state, &arena)?;
            info.fill(base, &mut arena);
            let ctx = SampleCtx {
                info: info.into_info(base),
                origin,
                spacing: 8,
                dims: [dims.x as u32, dims.y as u32, dims.z as u32],
                grid: Slice::new(base + grid_offset as u32, (grid.data.len() * 4) as u32),
            };
            write_arena(state, base, &arena.data)?;
            write_struct(state, base + ctx_offset as u32, &ctx);
            call_entry(state, &stage.entry, base + ctx_offset as u32)?;
            let data = read_floats(state, base + grid_offset as u32, grid.data.len());
            grid.data.copy_from_slice(&data);
            free_arena(state, base, arena.data.len());
            Ok(())
        })
    }

    pub fn run_chunk(
        &self,
        stage: &ChunkStage,
        seed: u64,
        region: RegionPos,
        lod: u8,
        view: &SampleGridView,
        chunk: &mut BlockChunk,
    ) -> anyhow::Result<()> {
        let key = self.module(&stage.module.asset)?;

        self.with_state(&key, |state| {
            let mut arena = Arena::default();
            let info = build_info(
                &mut arena,
                seed,
                hash(seed, region, 0),
                region,
                lod,
                &view.layout,
                &stage.params,
            );
            let (grids_offset, grids) = build_grids(&mut arena, view);
            let chunk_offset = arena.push(&meta_to_bytes(&chunk.vec));
            let ctx_offset = arena.reserve(size_of::<ChunkCtx>());
            let base = alloc_arena(state, &arena)?;
            info.fill(base, &mut arena);
            fill_grids(&mut arena, base, grids_offset, view, &grids);
            let grid_view = GridView {
                grids: Slice::new(
                    base + grids_offset as u32,
                    (grids.len() * size_of::<GridDesc>()) as u32,
                ),
                origin: [
                    view.base.x * REGION_SIZE as i32,
                    view.base.y * REGION_SIZE as i32,
                    view.base.z * REGION_SIZE as i32,
                ],
                lod: lod as u32,
            };
            let block_size = 1u32 << lod;
            let origin = [
                region.x * REGION_SIZE as i32 + REGION_SIZE as i32 - block_size as i32,
                region.y * REGION_SIZE as i32 + REGION_SIZE as i32 - block_size as i32,
                region.z * REGION_SIZE as i32 + REGION_SIZE as i32 - block_size as i32,
            ];
            let ctx = ChunkCtx {
                info: info.into_info(base),
                view: grid_view,
                chunk: Slice::new(base + chunk_offset as u32, (chunk.vec.len() * 2) as u32),
                origin,
                stride: (REGION_SIZE >> lod) as u32 + 2,
                lod: lod as u32,
            };
            write_arena(state, base, &arena.data)?;
            write_struct(state, base + ctx_offset as u32, &ctx);
            call_entry(state, &stage.entry, base + ctx_offset as u32)?;
            let data = read_metas(state, base + chunk_offset as u32, chunk.vec.len());
            chunk.vec.copy_from_slice(&data);
            free_arena(state, base, arena.data.len());
            Ok(())
        })
    }

    pub fn run_place(
        &self,
        stage: &PlaceStage,
        seed: u64,
        region: RegionPos,
        structure: &StructureAsset,
        view: &SampleGridView,
    ) -> anyhow::Result<Vec<[u8; 3]>> {
        let key = self.module(&stage.module.asset)?;

        self.with_state(&key, |state| {
            let mut arena = Arena::default();
            let info = build_info(
                &mut arena,
                seed,
                hash(seed, region, structure.name_hash),
                region,
                0,
                &view.layout,
                &stage.params,
            );
            let (grids_offset, grids) = build_grids(&mut arena, view);
            let capacity = structure.max_sites as usize;
            let candidates_offset = arena.reserve(capacity * size_of::<Candidate>());
            let ctx_offset = arena.reserve(size_of::<PlaceCtx>());
            let base = alloc_arena(state, &arena)?;
            info.fill(base, &mut arena);
            fill_grids(&mut arena, base, grids_offset, view, &grids);
            let block_size = [
                structure.blocks[0].size.x,
                structure.blocks[0].size.y,
                structure.blocks[0].size.z,
            ];
            let collision_min = structure.bound.min - structure.anchor;
            let collision_max = structure.bound.max - structure.anchor;
            let grid_view = GridView {
                grids: Slice::new(
                    base + grids_offset as u32,
                    (grids.len() * size_of::<GridDesc>()) as u32,
                ),
                origin: [
                    view.base.x * REGION_SIZE as i32,
                    view.base.y * REGION_SIZE as i32,
                    view.base.z * REGION_SIZE as i32,
                ],
                lod: 0,
            };
            let ctx = PlaceCtx {
                info: info.into_info(base),
                view: grid_view,
                structure: StructureInfo {
                    name_hash: structure.name_hash,
                    size: block_size.map(u32::from),
                    anchor: structure.anchor.to_array().map(u32::from),
                    collision_min: collision_min.to_array().map(u32::from),
                    collision_max: collision_max.to_array().map(u32::from),
                    max_sites: structure.max_sites as u32,
                },
                candidates: Slice::new(
                    base + candidates_offset as u32,
                    (capacity * size_of::<Candidate>()) as u32,
                ),
                capacity: capacity as u32,
            };
            write_arena(state, base, &arena.data)?;
            write_struct(state, base + ctx_offset as u32, &ctx);
            let count = call_entry_count(state, &stage.entry, base + ctx_offset as u32)? as usize;
            let count = count.min(capacity);
            let candidates = read_candidates(state, base + candidates_offset as u32, count);
            free_arena(state, base, arena.data.len());

            Ok(candidates
                .into_iter()
                .filter(|candidate| candidate.pos.iter().all(|&pos| pos < REGION_SIZE as u32))
                .map(|candidate| {
                    [
                        candidate.pos[0] as u8,
                        candidate.pos[1] as u8,
                        candidate.pos[2] as u8,
                    ]
                })
                .collect())
        })
    }
}

struct InfoOffsets {
    params: usize,
    blocks: usize,
    block_count: usize,
    channels: usize,
    channel_count: usize,
    halo_density: i32,
    halo_erosion: i32,
}

impl InfoOffsets {
    fn into_info(self, base: u32) -> StageInfo {
        StageInfo {
            abi_version: ABI_VERSION,
            size: size_of::<StageInfo>() as u32,
            seed: 0,
            rng_seed: 0,
            region: [0; 3],
            lod: 0,
            params: Slice::new(base + self.params as u32, 0),
            blocks: Slice::new(
                base + self.blocks as u32,
                (self.block_count * size_of::<BlockEntry>()) as u32,
            ),
            channels: Slice::new(
                base + self.channels as u32,
                (self.channel_count * size_of::<ChannelDesc>()) as u32,
            ),
            halo_density: self.halo_density,
            halo_erosion: self.halo_erosion,
        }
    }
}

fn build_info(
    arena: &mut Arena,
    seed: u64,
    rng_seed: u64,
    region: RegionPos,
    lod: u8,
    layout: &SampleLayout,
    params: &[u8],
) -> RuntimeInfo {
    let params_offset = arena.push(params);
    let mut names = Vec::new();

    for (name, _) in layout.channels() {
        names.push((arena.push(name.as_bytes()), name.len()));
    }

    let mut channel_offsets = Vec::new();
    for ((_, channel), (name_offset, name_len)) in layout.channels().iter().zip(names.iter()) {
        channel_offsets.push((*name_offset, *name_len, *channel));
    }
    let channels_offset = arena.reserve(channel_offsets.len() * size_of::<ChannelDesc>());
    let mut block_names = Vec::new();

    for (id, name) in BLOCK_TYPES.names.iter() {
        block_names.push((arena.push(name.as_bytes()), name.len(), *id));
    }
    let blocks_offset = arena.reserve(block_names.len() * size_of::<BlockEntry>());
    let block_count = block_names.len();
    let channel_count = channel_offsets.len();
    let halo = layout.halo();

    RuntimeInfo {
        seed,
        rng_seed,
        region,
        lod,
        params_len: params.len(),
        params: params_offset,
        blocks: blocks_offset,
        block_names,
        channels: channels_offset,
        channel_offsets,
        base_offsets: InfoOffsets {
            params: params_offset,
            blocks: blocks_offset,
            block_count,
            channels: channels_offset,
            channel_count,
            halo_density: halo.map_or(-1, |halo| halo.density.offset as i32),
            halo_erosion: halo.map_or(-1, |halo| halo.erosion.offset as i32),
        },
    }
}

struct RuntimeInfo {
    seed: u64,
    rng_seed: u64,
    region: RegionPos,
    lod: u8,
    params: usize,
    params_len: usize,
    blocks: usize,
    block_names: Vec<(usize, usize, crate::util::Id)>,
    channels: usize,
    channel_offsets: Vec<(usize, usize, Channel)>,
    base_offsets: InfoOffsets,
}

impl RuntimeInfo {
    fn fill(&self, base: u32, arena: &mut Arena) {
        let mut entries = Vec::with_capacity(self.block_names.len());

        for &(offset, len, id) in &self.block_names {
            entries.push(BlockEntry {
                name: Slice::new(base + offset as u32, len as u32),
                meta: id as u16,
                default_state: 0,
                _padding: 0,
            });
        }
        arena.write_at(self.blocks, structs_to_bytes(&entries));

        let mut channels = Vec::with_capacity(self.channel_offsets.len());

        for &(offset, len, channel) in &self.channel_offsets {
            channels.push(ChannelDesc {
                name: Slice::new(base + offset as u32, len as u32),
                offset: channel.offset as u32,
                components: channel.components as u32,
            });
        }
        arena.write_at(self.channels, structs_to_bytes(&channels));
    }

    fn into_info(self, base: u32) -> StageInfo {
        let mut info = self.base_offsets.into_info(base);
        info.seed = self.seed;
        info.rng_seed = self.rng_seed;
        info.region = self.region.to_array();
        info.lod = self.lod as u32;
        info.params = Slice::new(base + self.params as u32, self.params_len as u32);
        info
    }
}

struct GridEntry {
    x: u8,
    y: u8,
    z: u8,
    data: usize,
    len: usize,
    size: [u32; 3],
}

fn build_grids(arena: &mut Arena, view: &SampleGridView) -> (usize, Vec<GridEntry>) {
    let mut entries = Vec::new();

    for x in 0..view.grids.size.x {
        for y in 0..view.grids.size.y {
            for z in 0..view.grids.size.z {
                let grid = view.grids.get(glam::U8Vec3::new(x, y, z));
                let data = arena.push(&floats_to_bytes(&grid.data));
                entries.push(GridEntry {
                    x,
                    y,
                    z,
                    data,
                    len: grid.data.len() * 4,
                    size: [grid.size.x as u32, grid.size.y as u32, grid.size.z as u32],
                });
            }
        }
    }

    let offset = arena.reserve(entries.len() * size_of::<GridDesc>());

    (offset, entries)
}

fn fill_grids(
    arena: &mut Arena,
    base: u32,
    offset: usize,
    view: &SampleGridView,
    entries: &[GridEntry],
) {
    let descs = entries
        .iter()
        .map(|entry| {
            let region =
                view.base + glam::IVec3::new(entry.x as i32, entry.y as i32, entry.z as i32);
            GridDesc {
                origin: [
                    region.x * REGION_SIZE as i32,
                    region.y * REGION_SIZE as i32,
                    region.z * REGION_SIZE as i32,
                ],
                dims: entry.size,
                data: Slice::new(base + entry.data as u32, entry.len as u32),
            }
        })
        .collect::<Vec<_>>();

    arena.write_at(offset, structs_to_bytes(&descs));
}

fn alloc_arena(state: &mut ThreadState, arena: &Arena) -> anyhow::Result<u32> {
    let len = arena.data.len().max(1) as u32;
    state.store.set_epoch_deadline(EPOCH_DEADLINE);
    state
        .alloc
        .call(&mut state.store, len)
        .map_err(|error| anyhow!("failed to allocate {len} bytes in the stage: {error}"))
}

fn free_arena(state: &mut ThreadState, base: u32, len: usize) {
    if let Some(dealloc) = state.dealloc.as_ref() {
        let _ = dealloc.call(&mut state.store, (base, len.max(1) as u32));
    }
}

fn call_entry_count(state: &mut ThreadState, entry: &str, ctx: u32) -> anyhow::Result<u32> {
    let function = state
        .instance
        .get_typed_func::<u32, u32>(&mut state.store, entry)
        .map_err(|error| anyhow!("the stage has no entry {entry}: {error}"))?;
    state.store.set_epoch_deadline(EPOCH_DEADLINE);
    let result = function.call(&mut state.store, ctx).map_err(wasm)?;

    if result == u32::MAX {
        return Err(anyhow!("the stage {entry} failed"));
    }

    Ok(result)
}

fn call_entry(state: &mut ThreadState, entry: &str, ctx: u32) -> anyhow::Result<()> {
    call_entry_count(state, entry, ctx).map(|_| ())
}

fn wasm(error: wasmtime::Error) -> anyhow::Error {
    anyhow!("{error}")
}

fn write_struct<T>(state: &mut ThreadState, ptr: u32, value: &T) {
    let bytes = unsafe { slice::from_raw_parts(value as *const T as *const u8, size_of::<T>()) };
    let memory = state.memory.data_mut(&mut state.store);
    let start = ptr as usize;
    memory[start..start + bytes.len()].copy_from_slice(bytes);
}

fn write_arena(state: &mut ThreadState, base: u32, data: &[u8]) -> anyhow::Result<()> {
    let memory = state.memory.data_mut(&mut state.store);
    let start = base as usize;
    if start + data.len() > memory.len() {
        return Err(anyhow!("the stage allocation is outside its memory"));
    }
    memory[start..start + data.len()].copy_from_slice(data);
    Ok(())
}

fn read_floats(state: &mut ThreadState, ptr: u32, len: usize) -> Vec<f32> {
    let memory = state.memory.data(&state.store);
    let start = ptr as usize;
    let bytes = &memory[start..start + len * 4];
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_metas(state: &mut ThreadState, ptr: u32, len: usize) -> Vec<BlockMeta> {
    let memory = state.memory.data(&state.store);
    let start = ptr as usize;
    let bytes = &memory[start..start + len * 2];
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_candidates(state: &mut ThreadState, ptr: u32, len: usize) -> Vec<Candidate> {
    let memory = state.memory.data(&state.store);
    let start = ptr as usize;
    let bytes = &memory[start..start + len * size_of::<Candidate>()];
    bytes
        .chunks_exact(size_of::<Candidate>())
        .map(|chunk| unsafe { *(chunk.as_ptr() as *const Candidate) })
        .collect()
}

fn floats_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn meta_to_bytes(values: &[BlockMeta]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[derive(Default)]
struct Arena {
    data: Vec<u8>,
}

impl Arena {
    fn reserve(&mut self, len: usize) -> usize {
        self.align(8);
        let offset = self.data.len();
        self.data.resize(offset + len, 0);
        offset
    }

    fn push(&mut self, bytes: &[u8]) -> usize {
        self.align(8);
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        self.data.resize(offset + bytes.len().div_ceil(8) * 8, 0);
        offset
    }

    fn align(&mut self, align: usize) {
        while self.data.len() % align != 0 {
            self.data.push(0);
        }
    }

    fn write_at(&mut self, offset: usize, bytes: Vec<u8>) {
        self.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
    }
}

fn structs_to_bytes<T>(values: &[T]) -> Vec<u8> {
    unsafe {
        slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * size_of::<T>()).to_vec()
    }
}

fn hash(seed: u64, region: RegionPos, structure: u64) -> u64 {
    fn mix(mut hash: u64, value: u64) -> u64 {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
        hash
    }

    let mut hash = 0xcbf29ce484222325u64;
    hash = mix(hash, seed);
    hash = mix(hash, region.x as u32 as u64);
    hash = mix(hash, region.y as u32 as u64);
    hash = mix(hash, region.z as u32 as u64);
    hash = mix(hash, structure);
    hash
}
