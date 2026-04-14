use glam::Vec3;
use crate::util::Id;
use crate::world::meshing::BlockModel;

pub mod registries {
	use crate::util::collection::Registry;
	use crate::world::block::BlockType;
	use crate::world::registries::block_mesh_templates;
    use log::error;
	use std::sync::OnceLock;
    use glam::Vec3;
    use crate::world::meshing::{BlockModel, TemplatedMesh};
    
    static BLOCK_TYPES: OnceLock<Registry<BlockType>> = OnceLock::new();

    pub fn init_block_types() {
        let mut block_types = Registry::new();
        let mesh_templates = block_mesh_templates();

        let air = BlockType {
            models: vec![BlockModel::empty()],
            model_idx_of_state: |_| -> usize { 0 },
            opacity: Vec3::ZERO,
        };

        let bricks = BlockType {
            models: vec![BlockModel::new(vec![
                TemplatedMesh {
                    template_id: mesh_templates.id_from_name("cube_w"),
                    texture_id: 1,
                },
                TemplatedMesh {
                    template_id: mesh_templates.id_from_name("cube_e"),
                    texture_id: 1,
                },
                TemplatedMesh {
                    template_id: mesh_templates.id_from_name("cube_d"),
                    texture_id: 1,
                },
                TemplatedMesh {
                    template_id: mesh_templates.id_from_name("cube_u"),
                    texture_id: 1,
                },
                TemplatedMesh {
                    template_id: mesh_templates.id_from_name("cube_n"),
                    texture_id: 1,
                },
                TemplatedMesh {
                    template_id: mesh_templates.id_from_name("cube_s"),
                    texture_id: 1,
                },
            ])],
            model_idx_of_state: |_| -> usize { 0 },
            opacity: Vec3::ONE,
        };

        block_types.register("air", air);
        block_types.register("bricks", bricks);

        if BLOCK_TYPES.set(block_types).is_err() {
            error!("Block Types already initialized");
        }
    }

    #[inline]
    pub fn block_types() -> &'static Registry<BlockType> {
        BLOCK_TYPES
            .get()
            .expect("Failed to get Block Types. Make sure to call registries::init first!")
    }
}

pub type Meta = u16;
pub type State = u8;

pub struct BlockType {
    models: Vec<BlockModel>,
    model_idx_of_state: fn(State) -> usize,
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
            block_type: registries::block_types().by_id(type_id),
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
