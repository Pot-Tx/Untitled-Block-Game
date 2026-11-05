use crate::id_of;
use crate::render::{Mesh, NormTexVertex};
use crate::util::Id;
use crate::util::OnceInit;
use crate::util::bounding::AABB;
use crate::util::collection::{Registry, Volume};
use crate::world::model::BlockModel;
use crate::world::{BlockPos, Emission};
use glam::{U8Vec3, Vec3};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

/// The block types of the game, in the order their ids are registered in.
pub static BLOCK_TYPES: OnceInit<Registry<BlockType>> = OnceInit::new();

/// Builds the block types, taking the models of the blocks that have one from
/// [`BLOCK_MODEL_TEMPLATES`](crate::world::model::BLOCK_MODEL_TEMPLATES).
pub(super) fn init_block_types() -> anyhow::Result<()> {
    let mut block_types = Registry::new();
    let models = Registry::<BlockModel>::load_rons_from("assets/models/block")?;

    let air = BlockType {
        models: vec![BlockModel::empty()],
        bounds: vec![vec![]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: None,
        default_state: BlockState::DEFAULT,
    };

    let bricks = BlockType {
        models: vec![models.by_name("bricks").clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: None,
        default_state: BlockState::DEFAULT,
    };

    let dirt = BlockType {
        models: vec![models.by_name("dirt").clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: None,
        default_state: BlockState::DEFAULT,
    };

    let grass = BlockType {
        models: vec![models.by_name("grass").clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: None,
        default_state: BlockState::DEFAULT,
    };

    let log = BlockType {
        models: vec![models.by_name("log").clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: None,
        default_state: BlockState::DEFAULT,
    };

    let leaves = BlockType {
        models: vec![models.by_name("leaves").clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: None,
        default_state: BlockState::DEFAULT,
    };

    let lamp = BlockType {
        models: vec![models.by_name("lamp").clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: Some(Emission {
            offset: Vec3::splat(0.5),
            direction: Vec3::ZERO,
            color: Vec3::new(1.0, 0.55, 0.2),
            intensity: 1.5,
            range: 12.0,
            inner_cos: 1.0,
            outer_cos: -1.0,
        }),
        default_state: BlockState::DEFAULT,
    };

    let red_glass = BlockType {
        models: vec![models.by_name("red_glass").clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        state_models: [0; 16],
        state_bounds: [0; 16],
        emission: None,
        default_state: BlockState::DEFAULT,
    };

    block_types.register(0, "air", air);
    block_types.register(1, "bricks", bricks);
    block_types.register(2, "dirt", dirt);
    block_types.register(3, "grass", grass);
    block_types.register(4, "log", log);
    block_types.register(5, "leaves", leaves);
    block_types.register(6, "lamp", lamp);
    block_types.register(7, "red_glass", red_glass);

    BLOCK_TYPES.init(block_types);
    Ok(())
}

/// A block type and its state, packed into the value that a chunk stores.
///
/// The lowest 12 bits hold the id of the block type and the upper 4 bits its
/// state.
pub type BlockMeta = u16;

/// The state of a block: the combination of its properties.
///
/// A meta keeps the state in four bits, so only the values up to
/// [`Self::MAX`] are valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
#[repr(transparent)]
pub struct BlockState(u8);

impl BlockState {
    /// The state a block type creates its blocks with.
    pub const DEFAULT: Self = Self(0);

    /// The highest state a block can have, because a meta keeps it in four
    /// bits.
    pub const MAX: Self = Self(0xF);

    /// The state the four bits of a [`BlockMeta`] stand for.
    #[inline]
    const fn from_meta_bits(bits: u8) -> Self {
        Self(bits & 0xF)
    }

    /// The state as the index of the per-state tables of a [`BlockType`].
    #[inline]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
}

impl Default for BlockState {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TryFrom<u8> for BlockState {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> anyhow::Result<Self> {
        if value > Self::MAX.0 {
            anyhow::bail!("a block state of {value} does not fit in four bits");
        }

        Ok(Self(value))
    }
}

impl From<BlockState> for u8 {
    #[inline]
    fn from(state: BlockState) -> Self {
        state.0
    }
}

/// The block value a structure uses to leave the block that is already there
/// untouched while it is stamped.
pub const KEEP: BlockMeta = BlockMeta::MAX;

/// The blocks of one rectangular volume, stored as one meta per block.
///
/// This is the block storage shared by the chunks of the world and by the
/// content of a structure. The values are packed metas rather than [`Block`]s,
/// so a chunk can be saved and loaded without resolving the block types.
#[derive(Clone, Serialize, Deserialize)]
#[serde(into = "RawBlockChunk", try_from = "RawBlockChunk")]
pub struct BlockChunk(pub Volume<BlockMeta>);

/// The on-disk representation of a [`BlockChunk`].
///
/// The blocks are run length encoded, because a chunk is mostly air or one kind
/// of rock. The size is part of the document because a structure chunk and the
/// chunk of a region have different sizes.
#[derive(Serialize, Deserialize)]
pub(crate) struct RawBlockChunk {
    pub size: [u8; 3],
    pub runs: Vec<(BlockMeta, u16)>,
}

/// The largest side of a chunk a file may describe, which bounds how much a
/// malformed document can allocate while it is decoded.
const MAX_CHUNK_SIDE: u8 = 64;

impl From<BlockChunk> for RawBlockChunk {
    fn from(chunk: BlockChunk) -> Self {
        let mut runs = Vec::<(BlockMeta, u16)>::new();

        for &block in chunk.0.vec.iter() {
            match runs.last_mut() {
                Some((last, count)) if *last == block && *count < u16::MAX => *count += 1,
                _ => runs.push((block, 1)),
            }
        }

        Self {
            size: chunk.0.size.to_array(),
            runs,
        }
    }
}

impl TryFrom<RawBlockChunk> for BlockChunk {
    type Error = anyhow::Error;

    fn try_from(raw: RawBlockChunk) -> anyhow::Result<Self> {
        let size = U8Vec3::from(raw.size);
        if size.min_element() == 0 || size.max_element() > MAX_CHUNK_SIDE {
            anyhow::bail!("a chunk size of {size} is not supported");
        }

        let expected = size.as_usizevec3().element_product();
        let mut vec = Vec::with_capacity(expected);

        for (block, count) in raw.runs {
            let len = vec.len() + count as usize;

            if len > expected {
                anyhow::bail!("a chunk holds more blocks than its size allows");
            }

            vec.resize(len, block);
        }

        if vec.len() != expected {
            anyhow::bail!(
                "a chunk holds {} blocks instead of the {expected} its size asks for",
                vec.len()
            );
        }

        Ok(Self(Volume { size, vec }))
    }
}

impl Deref for BlockChunk {
    type Target = Volume<BlockMeta>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BlockChunk {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Volume<BlockMeta>> for BlockChunk {
    fn from(volume: Volume<BlockMeta>) -> Self {
        Self(volume)
    }
}

impl BlockChunk {
    #[inline]
    pub fn from_fn<F: Fn(U8Vec3) -> BlockMeta>(size: U8Vec3, pos_to_item: F) -> Self {
        Self(Volume::from_fn(size, pos_to_item))
    }

    #[inline]
    pub fn get_block(&self, pos: U8Vec3) -> Block {
        Block::from_meta(*self.0.get(pos))
    }

    #[inline]
    pub fn set_block(&mut self, pos: U8Vec3, block: Block) -> Block {
        Block::from_meta(self.0.set(pos, block.to_meta()))
    }

    #[inline]
    pub fn part(&self, min: U8Vec3, size: U8Vec3) -> Self {
        Self(self.0.part(min, size))
    }

    #[inline]
    pub fn bound(&self) -> AABB<U8Vec3> {
        self.0.bound()
    }
}

/// One block type: the models it is drawn with, the boxes it collides with and
/// how much light it lets through.
pub struct BlockType {
    /// The models this block can be drawn with; the state selects one of them.
    pub models: Vec<BlockModel>,
    /// The collision boxes for every state, in block coordinates.
    pub bounds: Vec<Vec<AABB<Vec3>>>,
    /// Selects the model that belongs to a state.
    pub state_models: [u8; 16],
    /// Selects the collision boxes that belong to a state.
    pub state_bounds: [u8; 16],
    /// The light this block emits, if any.
    pub emission: Option<Emission>,
    /// The state a block of this type is created with.
    pub default_state: BlockState,
}

/// One block in the world: its type, its state and a reference to its type.
#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(into = "RawBlock")]
#[serde(from = "RawBlock")]
pub struct Block {
    pub type_id: Id,
    pub block_type: &'static BlockType,
    pub state: BlockState,
}

#[derive(Serialize, Deserialize)]
pub struct RawBlock<'a> {
    block: &'a str,
    #[serde(default)]
    state: BlockState,
}

impl Eq for Block {}

impl PartialEq<Self> for Block {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id && self.state == other.state
    }
}

impl From<RawBlock<'_>> for Block {
    fn from(value: RawBlock) -> Self {
        let type_id = BLOCK_TYPES.id_of(value.block);
        Self {
            type_id,
            block_type: BLOCK_TYPES.get(type_id),
            state: value.state,
        }
    }
}

impl From<Block> for RawBlock<'_> {
    fn from(value: Block) -> Self {
        Self {
            block: BLOCK_TYPES.name_of(value.type_id),
            state: value.state,
        }
    }
}

impl Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Block")
            .field("type_id", &self.type_id)
            .field("type_name", &BLOCK_TYPES.name_of(self.type_id))
            .field("state", &self.state)
            .finish()
    }
}

impl Block {
    /// The block of the air type, which has no collisions and no model.
    #[inline]
    pub fn air() -> Self {
        Self::default_of(id_of!(BLOCK_TYPES, "air"))
    }

    /// The block of type `type_id` in its default state.
    #[inline]
    pub fn default_of(type_id: Id) -> Self {
        let block_type = BLOCK_TYPES.get(type_id);

        Self {
            type_id,
            block_type,
            state: block_type.default_state,
        }
    }

    /// The block a chunk entry stands for.
    #[inline]
    pub fn from_meta(meta: BlockMeta) -> Self {
        let type_id = (meta & 0xFFF) as Id;
        Self {
            type_id,
            block_type: BLOCK_TYPES.get(type_id),
            state: BlockState::from_meta_bits((meta >> 12) as u8),
        }
    }

    /// The chunk entry that stands for this block.
    #[inline]
    pub fn to_meta(self) -> BlockMeta {
        debug_assert!(
            self.type_id <= 0xFFF,
            "block type {} does not fit in a meta",
            self.type_id
        );
        self.type_id as BlockMeta + ((u8::from(self.state) as BlockMeta) << 12)
    }

    /// The block with the value of `P` set, leaving this one unchanged.
    #[inline]
    #[must_use]
    pub fn with_state(mut self, state: BlockState) -> Self {
        self.state = state;
        self
    }

    /// The model this block is drawn with.
    #[inline]
    pub fn model(&self) -> &BlockModel {
        let block_type = self.block_type;
        &block_type.models[block_type.state_models[self.state.idx()] as usize]
    }

    /// The mesh of this block in its current state, in block coordinates.
    #[inline]
    pub fn mesh(&self) -> Mesh<NormTexVertex> {
        self.model().mesh()
    }

    /// The collision boxes of this block, in world coordinates.
    #[inline]
    pub fn bounds(&self, pos: BlockPos) -> Vec<AABB<Vec3>> {
        let block_type = self.block_type;
        block_type.bounds[block_type.state_bounds[self.state.idx()] as usize]
            .iter()
            .map(|b| b.translate(pos.as_vec3()))
            .collect()
    }

    /// Whether the block collides with anything, which is what a walker has to
    /// step over or walk around.
    ///
    /// Unlike [`Self::bounds`] this allocates nothing, because the walkers of
    /// the world ask it for the blocks around them every tick.
    #[inline]
    pub fn is_solid(&self) -> bool {
        let block_type = self.block_type;
        !block_type.bounds[block_type.state_bounds[self.state.idx()] as usize].is_empty()
    }
}
