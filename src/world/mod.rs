mod block;
mod generation;
mod region;
mod render;
mod meshing;

use crate::actor::{PlayerControlled, Position};
use crate::ecs::*;
use crate::resources;
use crate::util::math::L1ShellIter;
use crossbeam_channel::Sender;
use glam::IVec3;
use rayon::ThreadPool;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;

pub use block::*;
pub use generation::*;
pub use meshing::*;
pub use region::*;
pub use render::*;

pub mod registries {
	use crate::render::Tex;
	use crate::util::collection::Registry;
	use crate::world::{block, meshing, BlockMeshTemplate, BlockModel, BlockType, TemplatedMesh};
	use glam::Vec3;
	use log::error;
	use std::sync::OnceLock;
	
	static BLOCK_TEXTURES: OnceLock<Registry<Tex>> = OnceLock::new();
	static BLOCK_MESH_TEMPLATES: OnceLock<Registry<BlockMeshTemplate>> = OnceLock::new();
	static BLOCK_TYPES: OnceLock<Registry<BlockType>> = OnceLock::new();
	
	pub fn init_block_textures() {
		let mut textures = Registry::new();
		let missing = Tex::from_png("missing").expect("Failed to load Textures");
		
		textures.register("missing", missing.clone());
		textures.register("bricks", Tex::from_png("bricks").unwrap_or(missing.clone()));
		
		if BLOCK_TEXTURES.set(textures).is_err() {
			error!("Block Textures already initialized");
		}
	}
	
	pub fn init_block_mesh_templates() {
		let mut templates = Registry::new();
		
		let [cube_w, cube_e, cube_d, cube_u, cube_n, cube_s] =
			BlockMeshTemplate::cuboid(Vec3::ZERO, Vec3::ONE, [false; 6]);
		
		templates.register("cube_w", cube_w);
		templates.register("cube_e", cube_e);
		templates.register("cube_d", cube_d);
		templates.register("cube_u", cube_u);
		templates.register("cube_n", cube_n);
		templates.register("cube_s", cube_s);
		
		if BLOCK_MESH_TEMPLATES.set(templates).is_err() {
			error!("Block Mesh Templates already initialized");
		}
	}
	
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
	pub fn block_textures() -> &'static Registry<Tex> {
		BLOCK_TEXTURES
			.get()
			.expect("Failed to get Block Textures. Make sure to call registries::init_block_textures first!")
	}
	
	#[inline]
	pub fn block_mesh_templates() -> &'static Registry<BlockMeshTemplate> {
		BLOCK_MESH_TEMPLATES
			.get()
			.expect("Failed to get Block Mesh Templates. Make sure to call registries::init_block_mesh_templates first!")
	}
	
	#[inline]
	pub fn block_types() -> &'static Registry<BlockType> {
		BLOCK_TYPES
			.get()
			.expect("Failed to get Block Types. Make sure to call registries::init first!")
	}
	
	pub fn init() {
		init_block_textures();
		init_block_mesh_templates();
		init_block_types();
	}
}

pub type RegionPos = IVec3;
pub type BlockPos = IVec3;
const DETAIL_LEVELS: usize = 8;

pub struct World<G: Generate> {
	center: RegionPos,
	lod_radii: SmallVec<[u32; DETAIL_LEVELS]>,
	update_level: usize,
	update_iters: SmallVec<[L1ShellIter<RegionPos>; 10]>,
	regions: HashMap<RegionPos, Region<G>>,
	active_regions: HashSet<RegionPos>,
	generator: Arc<G>,
	
	meshing_tx: Sender<MeshingTask>,
}

impl<G: Generate> Resource for World<G> {}

impl<G: Generate> World<G> {
	const LOD_DEPTHS: [Option<(u8, u8)>; DETAIL_LEVELS] = [
		None,
		Some((4, 5)),
		Some((3, 5)),
		Some((2, 4)),
		Some((1, 3)),
		Some((0, 2)),
		Some((0, 1)),
		Some((0, 0)),
	];
	
	const MAX_UPDATE_COST: usize = 32;
	
	pub fn new(lod_radii: Vec<u32>, generator: G, meshing_tx: Sender<MeshingTask>) -> Self {
		assert!(!lod_radii.is_empty(), "World's lod radii must not be empty");
		assert!(
			lod_radii.len() <= DETAIL_LEVELS,
			"World's lod radii must not be longer than detail level count"
		);
		
		let mut update_iters = SmallVec::new();
		update_iters.push(L1ShellIter::new(RegionPos::ZERO, 0));
		for &r in lod_radii.iter() {
			update_iters.push(L1ShellIter::new(RegionPos::ZERO, r as i32));
		}
		
		Self {
			center: RegionPos::ZERO,
			lod_radii: SmallVec::from_vec(lod_radii),
			update_level: 0,
			update_iters,
			regions: HashMap::new(),
			active_regions: HashSet::new(),
			generator: Arc::new(generator),
			
			meshing_tx,
		}
	}
	
	#[inline]
	pub fn traverse_block_pos(pos: BlockPos) -> (RegionPos, RelBlockPos) {
		let region_size = IVec3::splat(REGION_SIZE as i32);
		let region_pos = pos.div_euclid(region_size);
		let rel_block_pos = pos.rem_euclid(region_size).as_u8vec3();
		(region_pos, rel_block_pos)
	}
	
	pub fn get_block(&self, pos: BlockPos) -> Block {
		let (region_pos, rel_block_pos) = Self::traverse_block_pos(pos);
		if let Some(region) = self.regions.get(&region_pos) {
			region.get_block(rel_block_pos)
		} else {
			Block::air()
		}
	}
	
	pub fn set_block(&mut self, pos: BlockPos, block: Block) {
		let (region_pos, rel_block_pos) = Self::traverse_block_pos(pos);
		if let Some(region) = self.regions.get_mut(&region_pos) {
			region.set_block(rel_block_pos, block);
		}
	}
	
	#[inline]
	fn near_radius(&self) -> u32 {
		*self
			.lod_radii
			.first()
			.expect("World's lod radii is empty. What happened?")
	}
	
	#[inline]
	fn far_radius(&self) -> u32 {
		*self
			.lod_radii
			.last()
			.expect("World's lod radii is empty. What happened?")
	}
	
	#[inline]
	fn min_update_radius(&self, level: usize) -> u32 {
		if level == 0 {
			0
		} else {
			self.lod_radii[level - 1]
		}
	}
	
	#[inline]
	fn max_update_radius(&self, level: usize) -> u32 {
		if level == self.lod_radii.len() {
			self.far_radius() * 2 - 1
		} else {
			self.lod_radii[level]
		}
	}
	
	pub fn update(&mut self, center: RegionPos, threads: &WorldThreads) {
		let center_displacement = (center - self.center).abs().element_sum() as u32;
		self.center = center;
		let WorldThreads(near_threads, far_threads) = threads;
		
		if center_displacement > 0 {
			for i in 0..=self.lod_radii.len() {
				let min_radius = self.min_update_radius(i);
				let iter = &mut self.update_iters[i];
				let radius = (iter.radius as u32)
					.saturating_sub(center_displacement)
					.max(min_radius);
				
				*iter = L1ShellIter::new(center, radius as i32);
			}
			
			if center_displacement >= self.max_update_radius(self.lod_radii.len()) {
				self.regions.clear();
			}
			
			self.update_level = 0;
		}
		
		let mut cost = 0;
		
		while self.update_level <= self.lod_radii.len() {
			let level = self.update_level;
			let complex = (DETAIL_LEVELS - level).pow(2);
			let max_radius = self.max_update_radius(level);
			
			let iter = &mut self.update_iters[level];
			let context = if level < self.lod_radii.len() {
				Some(RegionContext {
					generator: self.generator.clone(),
					meshing_tx: self.meshing_tx.clone(),
					depths: Self::LOD_DEPTHS[level],
				})
			} else {
				None
			};
			let threads = if level == 0 {
				near_threads
			} else {
				far_threads
			};
			
			while cost < Self::MAX_UPDATE_COST {
				match iter.next() {
					Some(pos) => {
						if let Some(ctx) = &context {
							if let Some(region) = self.regions.get_mut(&pos) {
								if let Some(ctx) = &context {
									if region.update(ctx, threads) {
										self.active_regions.insert(pos);
										cost += complex;
									} else {
										cost += 1;
									}
								}
							} else {
								let region = Region::new(pos, ctx, threads);
								self.regions.insert(pos, region);
								self.active_regions.insert(pos);
								cost += complex;
							}
						} else {
							self.regions.remove(&pos);
							cost += 1;
						}
					}
					
					None => break,
				}
			}
			
			if cost < Self::MAX_UPDATE_COST {
				let next_radius = iter.radius as u32 + 1;
				if next_radius < max_radius {
					*iter = L1ShellIter::new(center, next_radius as i32);
				} else {
					self.update_level += 1;
				}
			} else {
				break;
			}
		}
		
		self.active_regions.retain(|pos| {
			if let Some(region) = self.regions.get_mut(pos) {
				let threads = if region.is_near() {
					near_threads
				} else {
					far_threads
				};
				
				!region.poll(threads)
			} else {
				false
			}
		});
	}
}

resources! {
    pub struct WorldThreads(ThreadPool, ThreadPool);
}

pub struct WorldUpdater<G: Generate>(pub PhantomData<G>);

impl<G: Generate> System for WorldUpdater<G> {
	type CompQuery = (CompRead<PlayerControlled>, CompRead<Position>);
	type ResQuery = (
		ResWrite<World<G>>,
		ResWrite<RenderedWorld>,
		ResRead<WorldThreads>,
	);
	
	fn operate(
		&mut self,
		entry: <Self::CompQuery as CompQuery>::Item<'_>,
		res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
	) -> Option<Vec<Command>> {
		let pos = entry.2.0.floor().as_ivec3();
		let center = pos.div_euclid(IVec3::splat(REGION_SIZE as i32));
		
		res.0.update(center, res.2);
		res.1.update(center);
		
		None
	}
}
