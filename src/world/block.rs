use crate::util::collection::Registry;
use crate::util::Id;
use crate::world::meshing::BlockModel;
use crate::world::TemplatedMesh;
use glam::Vec3;
use std::sync::LazyLock;

pub static BLOCK_TYPES: LazyLock<Registry<BlockType>> = LazyLock::new(|| build_block_types());

fn build_block_types() -> Registry<BlockType> {
    let mut block_types = Registry::new();

    let air = BlockType {
        models: vec![BlockModel::empty()],
        model_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ZERO,
    };

    let bricks = BlockType {
        models: vec![BlockModel::new(vec![
            TemplatedMesh {
                template: 0,
                texture: 1,
            },
            TemplatedMesh {
                template: 1,
                texture: 1,
            },
            TemplatedMesh {
                template: 2,
                texture: 1,
            },
            TemplatedMesh {
                template: 3,
                texture: 1,
            },
            TemplatedMesh {
                template: 4,
                texture: 1,
            },
            TemplatedMesh {
                template: 5,
                texture: 1,
            },
        ])],
        model_idx_of_state: |_| -> usize { 0 },
        opacity: Vec3::ONE,
    };

    block_types.register(0, air);
    block_types.register(1, bricks);

    block_types
}

pub type Meta = u16;
pub type State = u8;

pub struct BlockType {
    pub models: Vec<BlockModel>,
    pub model_idx_of_state: fn(State) -> usize,
    pub opacity: Vec3,
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

impl Block {
    #[inline]
    pub fn air() -> Self {
        Self::from_meta(0)
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
}
