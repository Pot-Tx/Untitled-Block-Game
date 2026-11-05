//! The types that cross the boundary between the game and a world generation
//! script. They are plain `repr(C)` values so that they can be shared by the
//! host and by the SDK a mod is built with.

/// The version of the world generation ABI. A module that exports another
/// version is rejected before any of its stages run.
pub const ABI_VERSION: u32 = 1;

/// The number of floats one sample may hold. A sample layout that asks for more
/// is rejected when the world program is compiled.
pub const MAX_SAMPLE_FLOATS: u32 = 32;

/// A range of bytes in the linear memory of a module.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Slice {
    pub ptr: u32,
    pub len: u32,
}

impl Slice {
    pub const EMPTY: Self = Self { ptr: 0, len: 0 };

    pub const fn new(ptr: u32, len: u32) -> Self {
        Self { ptr, len }
    }
}

/// One named channel of a sample.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChannelDesc {
    /// The UTF-8 name of the channel.
    pub name: Slice,
    pub offset: u32,
    pub components: u32,
}

/// One block type the script may name.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockEntry {
    pub name: Slice,
    pub meta: u16,
    pub default_state: u8,
    pub _padding: u8,
}

/// What every stage receives.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StageInfo {
    pub abi_version: u32,
    pub size: u32,
    pub seed: u64,
    pub rng_seed: u64,
    pub region: [i32; 3],
    pub lod: u32,
    /// The stage's own RON parameters, as UTF-8.
    pub params: Slice,
    /// `BlockEntry[]`.
    pub blocks: Slice,
    /// `ChannelDesc[]`.
    pub channels: Slice,
    /// The channel offset the halo redirection reads, or `-1` when the layout
    /// has no halo rule.
    pub halo_density: i32,
    pub halo_erosion: i32,
}

/// The coarse grids a chunk or a placement stage reads.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridDesc {
    /// The world block position of the first sample of the grid.
    pub origin: [i32; 3],
    pub dims: [u32; 3],
    /// `dims` product times the sample stride, as `f32`.
    pub data: Slice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridView {
    /// `GridDesc[]`.
    pub grids: Slice,
    /// The world block position of the lower corner of the grid volume.
    pub origin: [i32; 3],
    pub lod: u32,
}

/// A stage that writes the coarse samples of one region.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SampleCtx {
    pub info: StageInfo,
    /// The world block position the first sample of the grid stands for.
    pub origin: [i32; 3],
    /// The number of blocks between two samples.
    pub spacing: u32,
    /// The number of samples along each axis, including the halo.
    pub dims: [u32; 3],
    /// The sample grid, as `f32`, channel interleaved.
    pub grid: Slice,
}

/// A stage that writes or edits the blocks of one chunk.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChunkCtx {
    pub info: StageInfo,
    pub view: GridView,
    /// `u16` block metas, `stride ^ 3` of them.
    pub chunk: Slice,
    /// The world block position of chunk-local (0, 0, 0).
    pub origin: [i32; 3],
    pub stride: u32,
    pub lod: u32,
}

/// A structure as a placement stage sees it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StructureInfo {
    pub name_hash: u64,
    pub size: [u32; 3],
    pub anchor: [u32; 3],
    pub collision_min: [u32; 3],
    pub collision_max: [u32; 3],
    pub max_sites: u32,
}

/// One candidate site a placement stage returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub pos: [u32; 3],
    pub flags: u32,
}

/// A stage that decides where a structure may be placed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlaceCtx {
    pub info: StageInfo,
    pub view: GridView,
    pub structure: StructureInfo,
    /// `Candidate[]`, filled by the stage.
    pub candidates: Slice,
    pub capacity: u32,
}

/// The error code a stage returns when it succeeded.
pub const STAGE_OK: u32 = 0;
