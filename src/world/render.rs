use crate::actor::{PlayerControlled, Position, Rotation};
use crate::ecs::*;
use crate::render::*;
use crate::util::bounding::{PlaneGroup, AABB};
use crate::util::collection::{CubicVec, Registry};
use crate::util::math::L1ShellIter;
use crate::util::SwapPair;
use crate::world::*;
use crossbeam_channel::Receiver;
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use wgpu::{Color, LoadOp, PrimitiveTopology};

pub fn create_block_textures() -> Registry<Tex> {
    let mut textures = Registry::new();

    let missing = Tex::from_png("missing").expect("Failed to load Textures");
    let bricks = Tex::from_png("bricks").unwrap_or(missing.clone());

    textures.register(0, missing);
    textures.register(1, bricks);

    textures
}

pub struct RenderedChunk {
    block_geometry: SwapPair<Geometry<NormTexVertex>>,
    occlusion_geometry: SwapPair<Geometry<AlphaVertex>>,
    instance: Instances<IntTransInst>,
}

pub struct RenderedRegion {
    origin: BlockPos,
    near: Option<CubicVec<RenderedChunk>>,
    far: Option<RenderedChunk>,
    is_far: bool,
    on_far: bool,
}

pub struct RenderedWorld {
    center: RegionPos,
    radius: u32,
    remove_iter: L1ShellIter<RegionPos>,
    regions: HashMap<RegionPos, RenderedRegion>,
    active_regions: HashSet<RegionPos>,

    meshing_rx: Receiver<MeshingResult>,
}

impl Resource for RenderedWorld {}

impl RenderedWorld {
    const MAX_REMOVE_COST: usize = 256;

    pub fn new(radius: u32, meshing_rx: Receiver<MeshingResult>) -> Self {
        Self {
            center: RegionPos::ZERO,
            radius,
            remove_iter: L1ShellIter::new(RegionPos::ZERO, radius as i32),
            regions: HashMap::new(),
            active_regions: HashSet::new(),

            meshing_rx,
        }
    }
	
	pub fn update(&mut self, canvas: &Canvas, center: RegionPos) {
		while let Ok(result) = self.meshing_rx.try_recv() {
			let pos = result.pos;
			
			if ((pos - self.center).abs().element_sum() as u32) >= self.radius {
				continue;
			}
			
			if let Some(region) = self.regions.get_mut(&pos) {
				region.update(&canvas, result);
			} else {
				let is_far = result.chunk_pos.is_none();
				let mut region = RenderedRegion::new(pos, is_far);
				region.update(&canvas, result);
				self.regions.insert(pos, region);
			}
			
			self.active_regions.insert(pos);
        }

        self.active_regions.retain(|pos| {
            if let Some(region) = self.regions.get_mut(pos) {
                !region.poll()
            } else {
                false
            }
        });

        let center_displacement = (center - self.center).abs().element_sum() as u32;
        self.center = center;
        let remove_radius = (self.remove_iter.radius as u32).saturating_add(center_displacement);
        let max_radius = self.radius * 2;

        if remove_radius < max_radius {
            if center_displacement > 0 {
                self.remove_iter = L1ShellIter::new(center, remove_radius as i32);
            }

            if remove_radius >= self.radius {
                let mut cost = 0;

                while cost < Self::MAX_REMOVE_COST {
                    let iter = &mut self.remove_iter;

                    if let Some(pos) = iter.next() {
                        self.regions.remove(&pos);
                        cost += 1;
                    } else {
                        let next_radius = iter.radius as u32 - 1;
                        if next_radius < self.radius {
                            break;
                        } else {
                            self.remove_iter = L1ShellIter::new(center, next_radius as i32);
                        }
                    }
                }
            }
        } else {
            self.regions.clear();
            self.remove_iter = L1ShellIter::new(center, self.radius as i32 - 1);
        }
    }
}

impl Render<NormTexVertex, IntTransInst> for RenderedRegion {
    fn rendered(&self) -> Vec<RenderItem<'_, NormTexVertex, IntTransInst>> {
        let mut items = Vec::new();

        match self.on_far {
            false => {
                if let Some(chunks) = &self.near {
                    for chunk in chunks.vec.iter() {
                        if let Some(geometry) = chunk.block_geometry.current() {
                            items.push(RenderItem {
                                geometry,
                                instances: &chunk.instance,
                            });
                        }
                    }
                }
            }

            true => {
                if let Some(chunk) = &self.far {
                    if let Some(geometry) = chunk.block_geometry.current() {
                        items.push(RenderItem {
                            geometry,
                            instances: &chunk.instance,
                        });
                    }
                }
            }
        }

        items
    }
}

impl Render<AlphaVertex, IntTransInst> for RenderedRegion {
    fn rendered(&self) -> Vec<RenderItem<'_, AlphaVertex, IntTransInst>> {
        let mut items = Vec::new();

        match self.on_far {
            false => {
                if let Some(chunks) = &self.near {
                    for chunk in chunks.vec.iter() {
                        if let Some(geometry) = chunk.occlusion_geometry.current() {
                            items.push(RenderItem {
                                geometry,
                                instances: &chunk.instance,
                            });
                        }
                    }
                }
            }

            true => {
                if let Some(chunk) = &self.far {
                    if let Some(geometry) = chunk.occlusion_geometry.current() {
                        items.push(RenderItem {
                            geometry,
                            instances: &chunk.instance,
                        });
                    }
                }
            }
        }

        items
    }
}

impl RenderedRegion {
    fn new(pos: RegionPos, is_far: bool) -> Self {
        Self {
            origin: BlockPos::from(pos) * REGION_SIZE as i32,
            near: None,
            far: None,
            is_far,
            on_far: is_far,
        }
    }

    fn update(&mut self, canvas: &Canvas, result: MeshingResult) {
        let chunk = match result.chunk_pos {
            Some(chunk_pos) => {
                if self.near.is_none() {
                    let chunks = CubicVec::from_fn(4, |p| {
                        let pos = self.origin + BlockPos::from(p) * VecMode::CHUNK_SIZE as i32;
                        RenderedChunk::new(canvas, pos)
                    });
                    self.near = Some(chunks);
                }

                self.is_far = false;
                self.near.as_mut().unwrap().get_mut(chunk_pos)
            }

            None => {
                if self.far.is_none() {
                    let chunk = RenderedChunk::new(canvas, self.origin);
                    self.far = Some(chunk);
                }

                self.is_far = true;
                self.far.as_mut().unwrap()
            }
        };

        let block_geometry = if result.block_mesh.is_empty() {
            None
        } else {
            Some(result.block_mesh.geometry(canvas, "block"))
        };

        let occlusion_geometry = if result.occlusion_mesh.is_empty() {
            None
        } else {
            Some(result.occlusion_mesh.geometry(canvas, "occlusion"))
        };

        chunk.block_geometry.set(block_geometry, 4);
        chunk.occlusion_geometry.set(occlusion_geometry, 4);
    }

    fn poll(&mut self) -> bool {
        let finished = match self.is_far {
            false => {
                if let Some(chunks) = &mut self.near {
                    let mut finished = true;
                    for chunk in chunks.vec.iter_mut() {
                        finished &= chunk.update();
                    }
                    finished
                } else {
                    true
                }
            }

            true => {
                if let Some(chunk) = &mut self.far {
                    chunk.update()
                } else {
                    true
                }
            }
        };

        if self.on_far != self.is_far && finished {
            self.on_far = self.is_far;
            if self.on_far {
                self.near = None;
            } else {
                self.far = None;
            }
        }

        finished
    }

    pub fn bound(&self) -> AABB<Vec3> {
        AABB {
            min: self.origin.as_vec3(),
            max: (self.origin + REGION_SIZE as i32).as_vec3(),
        }
    }
}

impl RenderedChunk {
    fn new(canvas: &Canvas, pos: BlockPos) -> Self {
        Self {
            block_geometry: SwapPair::new(),
            occlusion_geometry: SwapPair::new(),
            instance: [IntTransInst { pos }].instances(canvas, "chunk"),
        }
    }

    fn update(&mut self) -> bool {
        self.block_geometry.update() & self.occlusion_geometry.update()
    }
}

resources! {
    pub struct BlockTextures(BindSet<TextureSampler>);
}

pub struct WorldRenderer {
    block_desc: RenderDescriptor<'static>,
    block_batch: RenderBatch<(TextureSampler, Transformation), NormTexVertex, IntTransInst>,

    occlusion_desc: RenderDescriptor<'static>,
    occlusion_batch: RenderBatch<Transformation, AlphaVertex, IntTransInst>,
}

impl System for WorldRenderer {
    type CompQuery = (
        CompRead<PlayerControlled>,
        CompRead<Position>,
        CompRead<Rotation>,
    );
    type ResQuery = (
        ResWrite<Option<Frame>>,
        ResRead<BlockTextures>,
        ResRead<Camera>,
        ResRead<RenderedWorld>,
    );

    fn operate(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let mut regions = res.3.regions.values().collect::<Vec<_>>();
        regions.retain(|&region| res.2.frustum.is_aabb_inside(region.bound()));

        if let Some(frame) = res.0 {
            frame.render(&self.block_desc, |mut pass| {
                self.block_batch.begin(&mut pass);
                self.block_batch
                    .push(&mut pass, (&res.1.0, &res.2.transform));

                regions.iter().for_each(|&region| {
                    self.block_batch.draw(&mut pass, region);
                });
            });

            frame.render(&self.occlusion_desc, |mut pass| {
                self.occlusion_batch.begin(&mut pass);
                self.occlusion_batch.push(&mut pass, &res.2.transform);

                regions.iter().for_each(|&region| {
                    self.occlusion_batch.draw(&mut pass, region);
                });
            })
        }

        None
    }
}

impl WorldRenderer {
	pub fn new(canvas: &Canvas) -> Self {
        Self {
            block_desc: RenderDescriptor {
                name: "block",
                color_load: LoadOp::Clear(Color {
                    r: 0.375,
                    g: 0.625,
                    b: 1.0,
                    a: 1.0,
                }),
                depth_load: LoadOp::Clear(0.0),
            },
            block_batch: RenderBatch::new(
                &canvas,
                &RenderBatchConfig {
                    name: "block",
                    shader: "block",
                    translucent: false,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),

            occlusion_desc: RenderDescriptor {
                name: "occlusion",
                color_load: LoadOp::Load,
                depth_load: LoadOp::Load,
            },
            occlusion_batch: RenderBatch::new(
                &canvas,
                &RenderBatchConfig {
                    name: "occlusion",
                    shader: "occlusion",
                    translucent: true,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: false,
                },
            ),
        }
    }
}
