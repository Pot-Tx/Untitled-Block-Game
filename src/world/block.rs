use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::Id;
use crate::world::model::BlockModel;
use crate::world::BlockPos;
use glam::Vec3;
use std::fmt;
use std::fmt::Debug;
use std::sync::LazyLock;
pub static BLOCK_TYPES: LazyLock<Registry<BlockType>> = LazyLock::new(|| build_block_types());

fn build_block_types() -> Registry<BlockType> {
    let mut block_types = Registry::new();
    let models = Registry::<BlockModel>::load_rons_from("assets/models/block")
        .expect("failed to load block models");

    let air = BlockType {
        models: vec![BlockModel::empty()],
        bounds: vec![vec![]],
        model_idx_of_state: |_| -> usize { 0 },
        bounds_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ZERO,
        default_state: 0,
    };

    let bricks = BlockType {
        models: vec![models.get(models.id_of("bricks")).clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        model_idx_of_state: |_| -> usize { 0 },
        bounds_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ONE,
        default_state: 0,
    };

    let dirt = BlockType {
        models: vec![models.get(models.id_of("dirt")).clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        model_idx_of_state: |_| -> usize { 0 },
        bounds_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ONE,
        default_state: 0,
    };

    let grass = BlockType {
        models: vec![models.get(models.id_of("grass")).clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        model_idx_of_state: |_| -> usize { 0 },
        bounds_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ONE,
        default_state: 0,
    };

    let log = BlockType {
        models: vec![models.get(models.id_of("log")).clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        model_idx_of_state: |_| -> usize { 0 },
        bounds_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ONE,
        default_state: 0,
    };

    let leaves = BlockType {
        models: vec![models.get(models.id_of("leaves")).clone()],
        bounds: vec![vec![AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        }]],
        model_idx_of_state: |_| -> usize { 0 },
        bounds_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ONE,
        default_state: 0,
    };

    block_types.register(0, "air", air);
    block_types.register(1, "bricks", bricks);
    block_types.register(2, "dirt", dirt);
    block_types.register(3, "grass", grass);
    block_types.register(4, "log", log);
    block_types.register(5, "leaves", leaves);

    block_types
}

pub type Meta = u16;
pub type State = u8;

pub struct BlockType {
    pub models: Vec<BlockModel>,
    pub bounds: Vec<Vec<AABB<Vec3>>>,
    pub model_idx_of_state: fn(State) -> usize,
    pub bounds_idx_of_state: fn(State) -> usize,
    pub opacity: Vec3,
    pub default_state: State,
}

#[derive(Copy, Clone)]
pub struct Block {
    pub type_id: Id,
    pub block_type: &'static BlockType,
    pub state: State,
}

pub trait Property {
    type Output;

    fn default_value() -> Self::Output;
    fn get_value_from_state(state: State) -> Self::Output;
    fn push_value_to_state(value: Self::Output, state: State) -> State;
}

impl Eq for Block {}

impl PartialEq<Self> for Block {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id && self.state == other.state
    }
}

impl Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Block")
            .field("type_id", &self.type_id)
            .field("state", &self.state)
            .finish()
    }
}

impl Block {
    #[inline]
    pub fn air() -> Self {
        Self::default_of(0)
    }

    #[inline]
    pub fn default_of(type_id: Id) -> Self {
        let block_type = BLOCK_TYPES.get(type_id);

        Self {
            type_id,
            block_type,
            state: block_type.default_state,
        }
    }

    #[inline]
    pub fn from_meta(meta: Meta) -> Self {
        let type_id = (meta & 0xFFF) as Id;
        Self {
            type_id,
            block_type: BLOCK_TYPES.get(type_id),
            state: (meta >> 12) as State,
        }
    }

    #[inline]
    pub fn to_meta(&self) -> Meta {
        self.type_id as Meta + ((self.state as Meta) << 12)
    }

    #[inline]
    pub fn get_property<P: Property>(&self) -> P::Output {
        P::get_value_from_state(self.state)
    }

    #[inline]
    pub fn set_property<P: Property>(&mut self, value: P::Output) -> &mut Self {
        self.state = P::push_value_to_state(value, self.state);
        self
    }

    #[inline]
    pub fn with_property<P: Property>(&self, value: P::Output) -> Self {
        let mut state = *self;
        state.set_property::<P>(value);
        state
    }

    #[inline]
    pub fn model(&self) -> &BlockModel {
        let block_type = self.block_type;
        &block_type.models[(block_type.model_idx_of_state)(self.state)]
    }

    #[inline]
    pub fn bounds(&self, pos: BlockPos) -> Vec<AABB<Vec3>> {
        let block_type = self.block_type;
        block_type.bounds[(block_type.bounds_idx_of_state)(self.state)]
            .iter()
            .map(|b| b.translate(pos.as_vec3()))
            .collect()
    }
}
