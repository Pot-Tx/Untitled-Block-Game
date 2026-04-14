use crate::util::Id;
use crate::world::block::Meta;
use crate::world::region::*;
use crate::world::BlockPos;
use glam::{DVec3, IVec3, U8Vec3};
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;
use std::array;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

pub trait Generate: Sync + Send + Sized + 'static {
    fn generate(&self, pos: BlockPos) -> Meta;

    fn generate_chunk(&self, origin: BlockPos, size: u8) -> Chunk;

    fn perform<T: GenType>(&self, task: GenTask, context: &T::Context) -> GenResult<T> {
        let id = task.id;
        let pos = task.pos;
        let output = T::perform(self, &task, context);

        GenResult { id, pos, output }
    }

    fn perform_batch<T: GenType>(&self, batch: GenTaskBatch<T>) -> GenResultBatch<T> {
        let context = batch.context;
        let results = batch
            .tasks
            .into_par_iter()
            .map(|task| self.perform(task, &context))
            .collect();

        GenResultBatch {
            context: T::next(&context),
            results,
        }
    }
}

pub struct TestGen {
    perlin: Perlin,
}

impl Generate for TestGen {
    fn generate(&self, pos: BlockPos) -> Meta {
        let height = (self.perlin.get([pos.x as f64 / 32.0, pos.z as f64 / 32.0]) * 32.0) as i32;
        if pos.y > height { 0 } else { 1 }
    }

    fn generate_chunk(&self, origin: BlockPos, size: u8) -> Chunk {
        let mut chunk = Chunk::new(size);

        for x in 0..size {
            for z in 0..size {
                let abx = origin.x + x as i32;
                let abz = origin.z + z as i32;
                let aby = (self.perlin.get([abx as f64 / 32.0, abz as f64 / 32.0]) * 32.0) as i32;
                let y = (aby - origin.y).clamp(0, (size - 1) as i32) as u8;
                
                chunk.fill(RelBlockPos::new(x, 0, z), RelBlockPos::new(x + 1, y + 1, z + 1), 1);
            }
        }

        chunk
    }
}

impl TestGen {
    pub fn new(seed: u32) -> Self {
        Self {
            perlin: Perlin::new(seed),
        }
    }
}

pub trait GenType {
    type Output: Sync + Send;
    type Context: Sync + Send;

    fn perform(generator: &impl Generate, task: &GenTask, context: &Self::Context) -> Self::Output;

    fn next(context: &Self::Context) -> Self::Context;
}

pub trait TreeGenType: GenType<Context = u8> {
    fn applies_to(node: &BlockNode) -> bool;
}

pub struct Split;

pub struct Fold;

pub struct Area;

impl GenType for Split {
    type Output = [Meta; 8];
    type Context = u8;

    #[inline]
    fn perform(generator: &impl Generate, task: &GenTask, context: &Self::Context) -> Self::Output {
        let offset = BlockTree::block_size_on_layer(context + 1);
        let base = task.pos + (offset / 2) as i32;
        array::from_fn(|i| {
            let pos = base + IVec3::from(BlockTree::child_pos_of_index(i as u8) * offset);
            generator.generate(pos)
        })
    }

    #[inline]
    fn next(context: &Self::Context) -> Self::Context {
        context + 1
    }
}

impl TreeGenType for Split {
    #[inline]
    fn applies_to(node: &BlockNode) -> bool {
        node.is_leaf()
    }
}

impl Split {
    #[inline]
    pub fn further_on(result: &GenResult<Self>, child_id: Id, offset: u8) -> [GenTask; 8] {
        let base = result.pos;
        array::from_fn(|i| {
            let id = child_id + i as u32;
            let pos = base + IVec3::from(BlockTree::child_pos_of_index(i as u8) * offset);
            GenTask { id, pos }
        })
    }
}

impl GenType for Fold {
    type Output = Meta;
    type Context = u8;

    #[inline]
    fn perform(generator: &impl Generate, task: &GenTask, context: &Self::Context) -> Self::Output {
        let offset = BlockTree::block_size_on_layer(context + 1);
        let pos = task.pos + offset as i32;
        generator.generate(pos)
    }

    #[inline]
    fn next(context: &Self::Context) -> Self::Context {
        *context
    }
}

impl TreeGenType for Fold {
    #[inline]
    fn applies_to(node: &BlockNode) -> bool {
        node.is_branch()
    }
}

impl GenType for Area {
    type Output = Chunk;
    type Context = u8;

    #[inline]
    fn perform(generator: &impl Generate, task: &GenTask, context: &Self::Context) -> Self::Output {
        generator.generate_chunk(task.pos, *context)
    }

    #[inline]
    fn next(context: &Self::Context) -> Self::Context {
        *context
    }
}

pub struct GenTaskBatch<T: GenType> {
    pub context: T::Context,
    pub tasks: Vec<GenTask>,
    pub _marker: PhantomData<T>,
}

pub struct GenTask {
    pub id: Id,
    pub pos: BlockPos,
}

pub struct GenResult<T: GenType> {
    pub id: Id,
    pub pos: BlockPos,
    pub output: T::Output,
}

pub struct GenResultBatch<T: GenType> {
    pub context: T::Context,
    pub results: Vec<GenResult<T>>,
}

impl GenTaskBatch<Area> {
    pub fn new_near(origin: BlockPos) -> Self {
        let task = GenTask {
            id: 0,
            pos: origin - 1,
        };

        Self {
            context: REGION_SIZE + 2,
            tasks: vec![task],
            _marker: PhantomData,
        }
    }
}

impl<T: TreeGenType> GenTaskBatch<T> {
    pub fn new_far(origin: BlockPos, tree: Arc<RwLock<BlockTree>>, layer: u8) -> Self {
        let mut tasks = Vec::new();
        let tree = tree.read().unwrap();

        for item in tree.iter_layer(layer) {
            if T::applies_to(item.node) {
                let task = GenTask {
                    id: item.id,
                    pos: origin + IVec3::from(item.pos),
                };
                tasks.push(task);
            }
        }

        Self {
            context: layer,
            tasks,
            _marker: PhantomData,
        }
    }
}
