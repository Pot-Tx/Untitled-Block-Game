use crate::ecs::*;
use crate::render::*;
use crate::world::*;

resources! {
    pub struct BlockTextures(BindSet<TextureArraySampler>);
}
