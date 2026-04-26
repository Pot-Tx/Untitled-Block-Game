use crate::util::collection::{CubicVec, DenseMap};
use crate::util::math::AllEq;
use crate::util::{Id, IdAllocator};
use crate::world::block::Meta;
use crate::world::generation::*;
use crate::world::meshing::MeshingTask;
use crate::world::{Block, BlockPos, RegionPos};
use anyhow::{Error, Result};
use crossbeam_channel::*;
use glam::U8Vec3;
use log::error;
use rayon::ThreadPool;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use std::{array, io};

pub type Chunk = CubicVec<Meta>;
pub type RelBlockPos = U8Vec3;
pub type ChunkPos = U8Vec3;
pub const REGION_SIZE: u8 = 64;
pub const MAX_LAYER_DEPTH: u8 = 6;

pub struct BlockTree {
    nodes: DenseMap<BlockNode>,
    allocator: IdAllocator,
}

#[derive(Eq, PartialEq, Debug)]
pub enum BlockNode {
    Leaf(Meta),
    Branch(Id),
}

pub struct BlockTreeIter<'a> {
    tree: &'a BlockTree,
    items: Vec<BlockNodeInfo<'a>>,
}

pub struct BlockLayerIter<'a> {
    tree: &'a BlockTree,
    layer: u8,
    items: Vec<BlockNodeInfo<'a>>,
}

#[derive(Clone)]
pub struct BlockNodeInfo<'a> {
    pub layer: u8,
    pub id: Id,
    pub pos: RelBlockPos,
    pub node: &'a BlockNode,
    count: u8,
}

impl Debug for BlockTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "BlockTree")?;
        for item in self.iter() {
            let prefix = "  ".repeat(item.layer as usize);
            let pos = item.pos;
            writeln!(
                f,
                "{}└─{} ({}, {}, {}) {:?}",
                prefix, item.id, pos.x, pos.y, pos.z, item.node
            )?;
        }
        Ok(())
    }
}

impl BlockTree {
    fn new(meta: Meta) -> Self {
        let mut nodes = DenseMap::new();
        let mut allocator = IdAllocator::new();
        nodes.insert(0, BlockNode::Leaf(meta));
        allocator.alloc(1);

        Self { nodes, allocator }
    }

    #[inline]
    pub const fn child_pos_of_index(idx: u8) -> U8Vec3 {
        U8Vec3::new(idx & 0b1, (idx >> 1) & 0b1, (idx >> 2) & 0b1)
    }

    #[inline]
    pub const fn child_index_of_pos(pos: U8Vec3) -> u8 {
        pos.x + (pos.y << 1) + (pos.z << 2)
    }

    #[inline]
    pub const fn block_size_on_layer(layer: u8) -> u8 {
        REGION_SIZE >> layer
    }

    #[inline]
    fn root(&self) -> &BlockNode {
        self.nodes.get(0).expect("Block Tree does not have root")
    }

    #[inline]
    fn get(&self, id: Id) -> Option<&BlockNode> {
        self.nodes.get(id)
    }

    #[inline]
    fn get_group(&self, id: Id) -> [Option<&BlockNode>; 8] {
        array::from_fn(|i| self.get(id + i as u32))
    }

    fn get_sized_meta(&self, mut pos: RelBlockPos) -> (u8, Meta) {
        let mut node = self.root();
        let mut size = REGION_SIZE;

        loop {
            match node {
                BlockNode::Leaf(meta) => {
                    return (size, *meta);
                }

                BlockNode::Branch(id) => {
                    size /= 2;
                    let idx = BlockTree::child_index_of_pos(pos / size);
                    node = self
                        .get(*id + idx as u32)
                        .expect("Failed to find child of Block Tree Branch");
                    pos %= size;
                }
            }
        }
    }

    fn replace(&mut self, id: Id, node: BlockNode) -> Option<BlockNode> {
        self.nodes.insert(id, node)
    }

    #[inline]
    fn insert(&mut self, node: BlockNode) -> Id {
        let id = self.allocator.alloc(1);
        self.nodes.insert(id, node);
        id
    }

    #[inline]
    fn insert_group(&mut self, nodes: [BlockNode; 8]) -> Id {
        let id = self.allocator.alloc(8);
        for (i, node) in nodes.into_iter().enumerate() {
            self.nodes.insert(id + i as u32, node);
        }
        id
    }

    #[inline]
    fn remove(&mut self, id: Id) {
        self.nodes.remove(id);
        self.allocator.free(id, 1);
    }

    #[inline]
    fn remove_group(&mut self, id: Id) {
        for i in 0..8 {
            self.nodes.remove(id + i);
        }
        self.allocator.free(id, 8);
    }

    fn split(&mut self, id: Id, children: [Meta; 8], force: bool) -> Option<Id> {
        if let Some(BlockNode::Leaf(_)) = self.get(id) {
            if force || !children.all_eq() {
                let children = children.map(|meta| BlockNode::Leaf(meta));
                let child_id = self.insert_group(children);
                self.replace(id, BlockNode::Branch(child_id));

                return Some(child_id);
            }
        }
        None
    }

    fn fold(&mut self, id: Id, meta: Meta, force: bool) -> Option<BlockNode> {
        if let Some(BlockNode::Branch(child_id)) = self.get(id) {
            let children = self.get_group(*child_id);
            if force || children.all_eq() {
                self.remove_group(*child_id);
                let node = self.replace(id, BlockNode::Leaf(meta));

                return node;
            }
        }
        None
    }

    pub fn iter(&'_ self) -> BlockTreeIter<'_> {
        BlockTreeIter::new(self)
    }

    pub fn iter_layer(&'_ self, layer: u8) -> BlockLayerIter<'_> {
        BlockLayerIter::new(self, layer)
    }

    pub fn chunk(&self, depth: u8) -> Chunk {
        let d = MAX_LAYER_DEPTH - depth;
        let mut vec = CubicVec::new(REGION_SIZE >> d);

        for item in self.iter() {
            if let BlockNode::Leaf(meta) = item.node {
                let pos = item.pos >> d;
                let size = BlockTree::block_size_on_layer(item.layer) >> d;
                vec.fill(pos, pos + size, *meta);
            }
        }

        vec
    }
}

impl BlockNode {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        match self {
            Self::Leaf(_) => true,
            Self::Branch(_) => false,
        }
    }

    #[inline]
    pub fn is_branch(&self) -> bool {
        match self {
            Self::Leaf(_) => false,
            Self::Branch(_) => true,
        }
    }
}

impl<'a> Iterator for BlockTreeIter<'a> {
    type Item = BlockNodeInfo<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(BlockNodeInfo {
                layer,
                id: _,
                node,
                pos,
                count,
            }) = self.items.last_mut()
            {
                let layer = layer.wrapping_add(1);

                if let BlockNode::Branch(id) = node {
                    let idx = *count;

                    if idx < 8 {
                        *count += 1;
                        let id = (*id).wrapping_add(idx as u32);

                        if let Some(node) = self.tree.get(id) {
                            let pos = (*pos).wrapping_add(
                                BlockTree::child_pos_of_index(idx)
                                    * BlockTree::block_size_on_layer(layer),
                            );
                            let last = BlockNodeInfo {
                                layer,
                                id,
                                node,
                                pos,
                                count: 0,
                            };

                            if node.is_branch() {
                                self.items.push(last.clone());
                            }
                            return Some(last);
                        }
                    } else {
                        self.items.pop();
                    }
                }
            } else {
                return None;
            }
        }
    }
}

impl<'a> BlockTreeIter<'a> {
    pub fn new(tree: &'a BlockTree) -> Self {
        Self {
            tree,
            items: vec![BlockNodeInfo::iter_root()],
        }
    }
}

impl<'a> Iterator for BlockLayerIter<'a> {
    type Item = BlockNodeInfo<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(BlockNodeInfo {
                layer,
                id: _,
                node,
                pos,
                count,
            }) = self.items.last_mut()
            {
                let layer = layer.wrapping_add(1);

                if let BlockNode::Branch(id) = node {
                    let idx = *count;

                    if idx < 8 {
                        *count += 1;
                        let id = (*id).wrapping_add(idx as u32);

                        if let Some(node) = self.tree.get(id) {
                            let pos = (*pos).wrapping_add(
                                BlockTree::child_pos_of_index(idx)
                                    * BlockTree::block_size_on_layer(layer),
                            );
                            let last = BlockNodeInfo {
                                layer,
                                id,
                                node,
                                pos,
                                count: 0,
                            };

                            if layer == self.layer {
                                return Some(last);
                            } else if node.is_branch() {
                                self.items.push(last);
                            }
                        }
                    } else {
                        self.items.pop();
                    }
                }
            } else {
                return None;
            }
        }
    }
}

impl<'a> BlockLayerIter<'a> {
    pub fn new(tree: &'a BlockTree, layer: u8) -> Self {
        Self {
            tree,
            layer,
            items: vec![BlockNodeInfo::iter_root()],
        }
    }
}

impl BlockNodeInfo<'_> {
    const ITER_ROOT_NODE: BlockNode = BlockNode::Branch((0 as Id).wrapping_sub(7));
    const ITER_ROOT_POS: RelBlockPos = RelBlockPos::ZERO.wrapping_sub(RelBlockPos::splat(64));

    fn iter_root() -> Self {
        Self {
            layer: u8::MAX,
            id: Id::MAX,
            node: &Self::ITER_ROOT_NODE,
            pos: Self::ITER_ROOT_POS,
            count: 7,
        }
    }
}

pub struct Region<G: Generate> {
    pos: RegionPos,
    mode: RegionMode<G>,
    meshing_tx: Sender<MeshingTask>,
}

pub enum RegionMode<G: Generate> {
    Vec(VecMode),
    Tree(TreeMode<G>),
}

pub struct RegionContext<G: Generate> {
    pub generator: Arc<G>,
    pub meshing_tx: Sender<MeshingTask>,
    pub depths: Option<(u8, u8)>,
}

pub struct VecMode {
    pub blocks: Arc<RwLock<CubicVec<Meta>>>,
    dirty_chunks: HashSet<ChunkPos>,
    touched: bool,
    changed: bool,

    gen_rx: Option<Receiver<GenResultBatch<Area>>>,
}

pub struct TreeMode<G: Generate> {
    pub origin: BlockPos,
    pub tree: Arc<RwLock<BlockTree>>,
    min_depth: u8,
    max_depth: u8,
    ongoing_splits: u8,
    ongoing_folds: u8,
    generator: Arc<G>,

    split_tx: Sender<GenResultBatch<Split>>,
    split_rx: Receiver<GenResultBatch<Split>>,
    fold_tx: Sender<GenResultBatch<Fold>>,
    fold_rx: Receiver<GenResultBatch<Fold>>,
}

impl<G: Generate> Region<G> {
    pub fn new(pos: RegionPos, context: &RegionContext<G>, threads: &ThreadPool) -> Self {
        let mode = RegionMode::new(pos, context, threads);

        Self {
            pos,
            mode,
            meshing_tx: context.meshing_tx.clone(),
        }
    }

    pub fn get_block(&self, pos: RelBlockPos) -> Block {
        match &self.mode {
            RegionMode::Vec(mode) => {
                let blocks = mode.blocks.read().unwrap();
                Block::from_meta(*blocks.get(pos + 1))
            }
            
            RegionMode::Tree(_) => Block::air(),
        }
    }

    pub fn set_block(&mut self, pos: U8Vec3, block: Block) -> bool {
        match &mut self.mode {
            RegionMode::Vec(mode) => {
                mode.set_block(pos, block);
                true
            }

            RegionMode::Tree(_) => false,
        }
    }

    pub fn update(&mut self, context: &RegionContext<G>, threads: &ThreadPool) -> bool {
        match &mut self.mode {
            RegionMode::Vec(mode) => {
                if context.depths.is_none() || mode.touched {
                    false
                } else {
                    self.mode = RegionMode::new(self.pos, context, threads);
                    true
                }
            }
            
            RegionMode::Tree(mode) => {
                if let Some((min_depth, max_depth)) = context.depths {
                    mode.set_min_depth(min_depth, threads) | mode.set_max_depth(max_depth, threads)
                } else {
                    self.mode = RegionMode::new(self.pos, context, threads);
                    true
                }
            }
        }
    }

    pub fn poll(&mut self, threads: &ThreadPool) -> bool {
        let finished = match &mut self.mode {
            RegionMode::Vec(mode) => mode.poll_generation(),
            
            RegionMode::Tree(mode) => mode.poll_splits(threads) & mode.poll_folds(threads),
        };

        if finished {
            let pos = self.pos;

            match &mut self.mode {
                RegionMode::Vec(mode) => {
                    for chunk_pos in mode.dirty_chunks.drain() {
                        let blocks = mode.blocks.clone();
                        let meshing_tx = self.meshing_tx.clone();

                        threads.spawn(move || {
                            let origin = chunk_pos * VecMode::CHUNK_SIZE;
                            let chunk = {
                                let blocks = blocks.read().unwrap();
                                blocks.part(origin, VecMode::CHUNK_SIZE + 2)
                            };

                            let task = MeshingTask {
                                pos,
                                chunk_pos: Some(chunk_pos),
                                chunk,
                            };
                            if meshing_tx.try_send(task).is_err() {
                                error!("Failed to send Meshing Task from Region at {}", pos);
                            }
                        });
                    }
                }
                
                RegionMode::Tree(mode) => {
                    let tree = mode.tree.clone();
                    let depth = mode.max_depth;
                    let meshing_tx = self.meshing_tx.clone();

                    threads.spawn(move || {
                        let blocks = {
                            let tree = tree.read().unwrap();
                            tree.chunk(depth)
                        };

                        let mut chunk = CubicVec::new(blocks.side + 2);
                        chunk.fit(U8Vec3::splat(1), &blocks);

                        let task = MeshingTask {
                            pos,
                            chunk_pos: None,
                            chunk,
                        };
                        if meshing_tx.try_send(task).is_err() {
                            error!("Failed to send Meshing Task from Region at {}", pos);
                        }
                    });
                }
            }
        }

        finished
    }
    
    pub fn save(&mut self) {
        match &mut self.mode {
            RegionMode::Vec(mode) => {
                if mode.changed {
                    if mode.save(self.pos).is_err() {
                        error!("Failed to save region at {}", self.pos);
                    }
                    
                    mode.changed = false;
                }
            }
            
            RegionMode::Tree(_) => (),
        }
    }
    
    pub fn is_vec(&self) -> bool {
        match self.mode {
            RegionMode::Vec(_) => true,
            RegionMode::Tree(_) => false,
        }
    }
    
    pub fn is_tree(&self) -> bool {
        match self.mode {
            RegionMode::Vec(_) => false,
            RegionMode::Tree(_) => true,
        }
    }
}

impl<G: Generate> RegionMode<G> {
    fn new(pos: RegionPos, context: &RegionContext<G>, threads: &ThreadPool) -> Self {
        if let Ok(mode) = VecMode::load(pos) {
            return Self::Vec(mode);
        }
        
        match context.depths {
            Some((min_depth, max_depth)) => Self::Tree(TreeMode::new(
                pos,
                &context.generator,
                min_depth,
                max_depth,
                threads,
            )),
            None => Self::Vec(VecMode::new(pos, &context.generator, threads)),
        }
    }
}

impl VecMode {
    pub const CHUNK_SIZE: u8 = 16;
    const CHUNK_POSES: [ChunkPos; 64] = Self::chunk_poses();
    
    const fn chunk_poses() -> [ChunkPos; 64] {
        let mut poses = [RelBlockPos::ZERO; 64];

        let mut x = 0;
        while x < 4 {
            let mut y = 0;
            while y < 4 {
                let mut z = 0;
                while z < 4 {
                    let idx = x as usize + ((y as usize) << 2) + ((z as usize) << 4);
                    poses[idx] = RelBlockPos::new(x, y, z);
                    z += 1;
                }
                y += 1;
            }
            x += 1;
        }

        poses
    }
    
    #[inline]
    fn pos_influence(pos: U8Vec3) -> SmallVec<[ChunkPos; 8]> {
        let mut influenced = SmallVec::new();
        
        fn range(b: u8) -> (u8, u8) {
            let b = b as i16;
            let min = ((b - 17) / 16).max(0);
            let max = (b / 16).min(3);
            (min as u8, max as u8)
        }
        
        let (minx, maxx) = range(pos.x);
        let (miny, maxy) = range(pos.y);
        let (minz, maxz) = range(pos.z);
        
        for x in minx..=maxx {
            for y in miny..=maxy {
                for z in minz..=maxz {
                    influenced.push(ChunkPos::new(x, y, z));
                }
            }
        }
        
        influenced
    }

    fn new<G: Generate>(pos: RegionPos, generator: &Arc<G>, threads: &ThreadPool) -> Self {
        let origin = pos * REGION_SIZE as i32;
        let blocks = CubicVec::new(REGION_SIZE + 2);
        let (gen_tx, gen_rx) = bounded(1);

        let region = Self {
            blocks: Arc::new(RwLock::new(blocks)),
            dirty_chunks: HashSet::from(Self::CHUNK_POSES),
            touched: false,
            changed: false,

            gen_rx: Some(gen_rx),
        };

        let generator = generator.clone();
        threads.spawn(move || {
            let tasks = GenTaskBatch::new_near(origin);
            let results = generator.perform_batch(tasks);
            if gen_tx.try_send(results).is_err() {
                error!("Failed to send Generation results to Region at {}", origin);
            }
        });

        region
    }
    
    fn load(pos: RegionPos) -> Result<Self> {
        let mut file = File::open(&format!("saves/{}.{}.{}.regn", pos.x, pos.y, pos.z))?;
        
        let mut magic_data = [0; 4];
        file.read_exact(&mut magic_data)?;
        let magic = str::from_utf8(&magic_data).unwrap_or("ERROR");
        
        let mut version_data = [0; 4];
        file.read_exact(&mut version_data)?;
        let version = u32::from_le_bytes(version_data);
        
        if magic != "REGN" || version != 0 {
            return Err(Error::from(io::Error::new(
                io::ErrorKind::Other,
                "Region file header is invalid",
            )));
        }
        
        let mut block_data = Vec::new();
        file.read_to_end(&mut block_data)?;
        
        if block_data.len() % 4 != 0 {
            return Err(Error::from(io::Error::new(
                io::ErrorKind::Other,
                "Region file is corrupted",
            )));
        }
        
        let mut blocks = Vec::new();
        
        for i in 0..block_data.len() / 4 {
            let j = i * 4;
            let meta = Meta::from_le_bytes(block_data[j..j + 2].try_into()?);
            let count = u16::from_le_bytes(block_data[j + 2..j + 4].try_into()?);
            blocks.resize(blocks.len() + count as usize, meta);
        }
        
        let blocks = Chunk {
            side: REGION_SIZE + 2,
            vec: blocks,
        };
        
        Ok(Self {
            blocks: Arc::new(RwLock::new(blocks)),
            dirty_chunks: HashSet::from(Self::CHUNK_POSES),
            touched: true,
            changed: false,
            
            gen_rx: None,
        })
    }
    
    fn save(&self, pos: RegionPos) -> Result<()> {
        let mut file = File::create(&format!("saves/{}.{}.{}.regn", pos.x, pos.y, pos.z))?;
        
        file.write_all(b"REGN")?;
        
        file.write_all(&0u32.to_le_bytes())?;
        
        let mut block_data = Vec::new();
        let mut meta = Meta::MAX;
        let mut count: u16 = 0;
        
        let blocks = self.blocks.read().unwrap();
        
        for &m in blocks.vec.iter() {
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
        Ok(())
    }
    
    fn get_block(&self, pos: RelBlockPos) -> Block {
        let blocks = self.blocks.read().unwrap();
        Block::from_meta(*blocks.get(pos + 1))
    }
    
    fn set_block(&mut self, pos: U8Vec3, block: Block) {
        let mut blocks = self.blocks.write().unwrap();
        blocks.set(pos, block.to_meta());
        self.dirty_chunks.extend(Self::pos_influence(pos));
        self.touched = true;
        self.changed = true;
    }

    fn poll_generation(&mut self) -> bool {
        match &self.gen_rx {
            Some(gen_rx) => {
                if let Ok(results) = gen_rx.try_recv() {
                    for result in results.results {
                        let mut chunk = self.blocks.write().unwrap();
                        *chunk = result.output;
                    }
                    
                    self.gen_rx = None;
                    true
                } else {
                    false
                }
            }
            
            None => true,
        }
    }
}

impl<G: Generate> TreeMode<G> {
    fn new(
        pos: RegionPos,
        generator: &Arc<G>,
        min_depth: u8,
        max_depth: u8,
        threads: &ThreadPool,
    ) -> Self {
        assert!(min_depth <= max_depth && max_depth <= MAX_LAYER_DEPTH);

        let origin = pos * REGION_SIZE as i32;
        let generator = generator.clone();
        let center = origin + BlockTree::block_size_on_layer(1) as i32;
        let meta = generator.generate(center);
        let tree = Arc::new(RwLock::new(BlockTree::new(meta)));
        let (split_tx, split_rx) = bounded(8);
        let (fold_tx, fold_rx) = bounded(8);

        let mut region = Self {
            origin,
            tree,
            min_depth,
            max_depth,
            ongoing_splits: 0,
            ongoing_folds: 0,
            generator,

            split_tx,
            split_rx,
            fold_tx,
            fold_rx,
        };

        if max_depth > 0 {
            region.split_layer(0, threads);
        }

        region
    }

    fn split_layer(&mut self, layer: u8, threads: &ThreadPool) {
        let generator = self.generator.clone();
        let origin = self.origin;
        let split_tx = self.split_tx.clone();
        let tree = self.tree.clone();

        threads.spawn(move || {
            let tasks = GenTaskBatch::new_far(origin, tree, layer);
            let results = generator.perform_batch(tasks);
            if split_tx.try_send(results).is_err() {
                error!("Failed to send Split results to Region at {}", origin);
            }
        });

        self.ongoing_splits += 1;
    }

    fn poll_splits(&mut self, threads: &ThreadPool) -> bool {
        if self.ongoing_splits > 0 {
            while let Ok(batch) = self.split_rx.try_recv() {
                self.ongoing_splits -= 1;
                let layer = batch.context;

                if layer > self.max_depth {
                    continue;
                }

                let force = layer <= self.min_depth;
                let mut next = Vec::new();

                {
                    let mut tree = self.tree.write().unwrap();
                    for result in batch.results {
                        if let Some(child_id) = tree.split(result.id, result.output, force) {
                            next.push((result, child_id));
                        }
                    }
                }

                if layer < self.max_depth && !next.is_empty() {
                    let generator = self.generator.clone();
                    let origin = self.origin;
                    let split_tx = self.split_tx.clone();

                    threads.spawn(move || {
                        let offset = BlockTree::block_size_on_layer(layer);
                        let mut tasks = Vec::new();
                        for (result, child_id) in next {
                            tasks.extend(Split::further_on(&result, child_id, offset));
                        }
                        let tasks = GenTaskBatch {
                            context: layer,
                            tasks,
                            _marker: PhantomData,
                        };
                        let results = generator.perform_batch(tasks);
                        if split_tx.try_send(results).is_err() {
                            error!("Failed to send Split results to Region at {}", origin);
                        }
                    });

                    self.ongoing_splits += 1;
                }
            }
        }

        self.ongoing_splits == 0
    }

    fn fold_layer(&mut self, layer: u8, threads: &ThreadPool) {
        let generator = self.generator.clone();
        let origin = self.origin;
        let fold_tx = self.fold_tx.clone();
        let tree = self.tree.clone();

        threads.spawn(move || {
            let tasks = GenTaskBatch::new_far(origin, tree, layer);
            let results = generator.perform_batch(tasks);
            if fold_tx.try_send(results).is_err() {
                error!("Failed to send Fold results to Region at {}", origin);
            }
        });

        self.ongoing_folds += 1;
    }

    fn poll_folds(&mut self, threads: &ThreadPool) -> bool {
        if self.ongoing_folds > 0 {
            while let Ok(batch) = self.fold_rx.try_recv() {
                self.ongoing_folds -= 1;
                let layer = batch.context;

                if layer < self.min_depth {
                    continue;
                }

                let force = layer >= self.max_depth;
                {
                    let mut tree = self.tree.write().unwrap();
                    for result in batch.results {
                        tree.fold(result.id, result.output, force);
                    }
                }

                if layer > self.max_depth {
                    self.fold_layer(layer - 1, threads);
                }
            }
        }

        self.ongoing_folds == 0
    }

    fn set_min_depth(&mut self, depth: u8, threads: &ThreadPool) -> bool {
        assert!(depth <= MAX_LAYER_DEPTH);
        let cur_depth = self.min_depth;
        self.min_depth = depth;

        if depth > cur_depth {
            self.split_layer(cur_depth, threads);
            return true;
        }

        if depth < cur_depth {
            self.fold_layer(cur_depth, threads);
            return true;
        }

        false
    }

    fn set_max_depth(&mut self, depth: u8, threads: &ThreadPool) -> bool {
        assert!(depth <= MAX_LAYER_DEPTH);
        let cur_depth = self.max_depth;
        self.max_depth = depth;

        if depth < cur_depth {
            self.fold_layer(cur_depth, threads);
            return true;
        }

        if depth > cur_depth {
            self.split_layer(cur_depth, threads);
            return true;
        }

        false
    }
}
