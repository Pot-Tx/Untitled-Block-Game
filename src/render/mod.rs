mod canvas;
mod batch;
mod config;
mod mesh;
mod vertex;

use crate::actor::{PlayerControlled, Position, Rotation, Velocity};
use crate::ecs::*;
use crate::util::bounding::Plane;
use crate::util::collection::Registry;
use crate::util::transform::*;
use crate::{resources, world};
use bytemuck::{Pod, Zeroable};
use glam::*;
use std::f32::consts::FRAC_PI_2;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::Read;
use wgpu::*;

pub use batch::*;
pub use canvas::*;
pub use config::*;
pub use mesh::*;
pub use vertex::*;

pub mod registries {
	use crate::render::{Binding, Canvas};
	use crate::util::collection::Registry;
	use log::error;
	use std::sync::{OnceLock, RwLock};
	
	static CANVAS: OnceLock<RwLock<Canvas>> = OnceLock::new();
	static BINDINGS: OnceLock<Registry<Binding>> = OnceLock::new();
	
	pub fn set_canvas(canvas: Canvas) {
		if CANVAS.set(RwLock::new(canvas)).is_err() {
			error!("Canvas already initialized");
		}
	}
	
	pub fn set_bindings(bindings: Registry<Binding>) {
		if BINDINGS.set(bindings).is_err() {
			error!("Bindings already initialized");
		}
	}
	
	#[inline]
	pub fn canvas() -> &'static RwLock<Canvas> {
		CANVAS
			.get()
			.expect("Failed to get Canvas. Make sure to call registries::set_canvas first!")
	}
	
	#[inline]
	pub fn bindings() -> &'static Registry<Binding> {
		BINDINGS
			.get()
			.expect("Failed to get Bindings. Make sure to call registries::set_bindings first!")
	}
}

resources! {
    pub struct PartialTick(f32);
}

pub struct Camera {
	near: f32,
	far: f32,
	fov: f32,
	
	pub buffer: Buffer,
	pub frustum: [Plane<Vec3>; 5],
}

#[derive(Clone)]
pub struct Tex {
	pub width: u32,
	pub height: u32,
	pub data: Vec<u8>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MipParams {
	pub width: u32,
	pub height: u32,
	pub level: u32,
	_pad: u32,
}

impl Resource for Camera {}

impl Camera {
	pub fn new(canvas: &Canvas) -> Self {
		Self {
			near: 0.125,
			far: 4096.0,
			fov: FRAC_PI_2,
			
			buffer: Buffer::new(canvas, &BufferConfig {
				name: "camera",
				content: &[Mat4::default()],
				usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			}),
			
			frustum: [Plane::default(); 5],
		}
	}
	
	pub fn transform(
		&mut self,
		pos: &Position,
		rot: &Rotation,
		vel: &Velocity,
		partial_tick: &PartialTick,
	) {
		let pos = pos.0 - vel.0 * (1.0 - partial_tick.0);
		let rot = rot.0;
		let aspect;
		
		let trans = Mat4::translation(-pos[0], -pos[1], -pos[2]);
		let rot = Mat4::rotation(-rot[0], -rot[1], -rot[2]);
		
		{
			let canvas = registries::canvas().read().unwrap();
			aspect = canvas.surface_config.width as f32 / canvas.surface_config.height as f32;
			let proj = Mat4::projection(self.near, self.far, self.fov, aspect);
			let mat = proj * rot * trans;
			
			let queue = &canvas.queue;
			queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[mat]));
		}
		
		let dy = self.far * (self.fov / 2.0).tan();
		let dx = dy * aspect;
		let rot = Mat3::from_mat4(rot).transpose();
		
		let bl = pos + rot * Vec3::new(dx, -dy, -self.far);
		let br = pos + rot * Vec3::new(-dx, -dy, -self.far);
		let tl = pos + rot * Vec3::new(dx, dy, -self.far);
		let tr = pos + rot * Vec3::new(-dx, dy, -self.far);
		let back = pos;
		let orient = pos + rot * Vec3::new(0.0, 0.0, -self.near);
		
		self.frustum = [
			Plane::from_points(back, bl, tl, orient),
			Plane::from_points(back, tr, br, orient),
			Plane::from_points(back, tl, tr, orient),
			Plane::from_points(back, br, bl, orient),
			Plane::from_points(bl, br, tl, orient),
		];
	}
}

impl Debug for Tex {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		f.debug_struct("Tex")
			.field("width", &self.width)
			.field("height", &self.height)
			.finish()
	}
}

impl Tex {
	pub fn from_png(name: &str) -> anyhow::Result<Self> {
		let mut file = File::open(&format!("assets/textures/{}.png", name))?;
		let mut bytes = Vec::new();
		file.read_to_end(&mut bytes)?;
		let image = image::load_from_memory(&bytes)?.into_rgba8();
		let (width, height) = image.dimensions();
		let data = image.into_raw();
		Ok(Self {
			width,
			height,
			data,
		})
	}
}

impl Registry<Tex> {
	pub fn create_texture_sampler<'a>(
		&self,
		canvas: &Canvas,
		name: &'a str,
	) -> (Texture, Sampler) {
		let width = self.by_id(0).width;
		let height = self.by_id(0).height;
		let len = self.items.len() as u32;
		let mip_level_count = width.min(height).ilog2() + 1;
		
		let mip_name = &format!("{}_mip", name);
		
		let storage_texture = Texture::new(
			canvas,
			&TextureConfig {
				name: mip_name,
				texs: &world::registries::block_textures().items,
				width,
				height,
				mip_level_count,
				storage: true,
			},
		);
		
		let texture = Texture::new(
			canvas,
			&TextureConfig {
				name,
				texs: &world::registries::block_textures().items,
				width,
				height,
				mip_level_count,
				storage: false,
			},
		);
		
		let src_binding = Binding::new(
			canvas,
			&BindingDescriptor {
				index: 0,
				name: &format!("{}_src", mip_name),
				entries: vec![BindingEntryDescriptor {
					visibility: ShaderStages::COMPUTE,
					item: BindingItem::TextureView(
						&TextureView::new(
							&texture,
							&TextureViewConfig {
								name: &format!("{}_src", mip_name),
								dimension: TextureViewDimension::D2Array,
								mip_level: None,
							},
						),
						TextureViewDimension::D2Array,
						None,
					),
				}],
			},
		);
		
		let mut frame = canvas.begin();
		
		for level in 1..mip_level_count {
			let width = width >> level;
			let height = height >> level;
			
			let dst_binding = Binding::new(
				canvas,
				&BindingDescriptor {
					index: 1,
					name: &format!("{}_dst", mip_name),
					entries: vec![
						BindingEntryDescriptor {
							visibility: ShaderStages::COMPUTE,
							item: BindingItem::TextureView(
								&TextureView::new(
									&storage_texture,
									&TextureViewConfig {
										name: &format!("{}_dst", mip_name),
										dimension: TextureViewDimension::D2Array,
										mip_level: Some(level),
									}
								),
								TextureViewDimension::D2Array,
								Some(StorageTextureAccess::WriteOnly),
							),
						},
						BindingEntryDescriptor {
							visibility: ShaderStages::COMPUTE,
							item: BindingItem::Buffer(
								&Buffer::new(
									&canvas,
									&BufferConfig {
										name: &format!("{}_params", mip_name),
										content: &[MipParams {
											width,
											height,
											level,
											_pad: 0,
										}],
										usage: BufferUsages::UNIFORM,
									},
								),
								BufferBindingType::Uniform,
							),
						},
					],
				},
			);
			
			let mut batch = ComputeBatch::new(
				canvas,
				ComputeBatchDescriptor {
					name: mip_name,
					bindings: vec![Some(&src_binding), Some(&dst_binding)],
					shader_name: "mipmap",
				},
			);
			
			frame.compute(&ComputeDescriptor { name: mip_name }, |mut pass| {
				batch.begin(&mut pass);
				batch.dispatch(&mut pass, width.div_ceil(8), height.div_ceil(8), len);
			});
			
			frame.encoder.copy_texture_to_texture(
				TexelCopyTextureInfo {
					texture: &storage_texture,
					mip_level: level,
					origin: Origin3d::default(),
					aspect: TextureAspect::default(),
				},
				TexelCopyTextureInfo {
					texture: &texture,
					mip_level: level,
					origin: Origin3d::default(),
					aspect: TextureAspect::default(),
				},
				Extent3d {
					width,
					height,
					depth_or_array_layers: len,
				},
			);
		}
		
		canvas.end(frame);
		
		let sampler = Sampler::new(&canvas, &SamplerConfig {
			name,
			address_mode: AddressMode::Repeat,
			mipmap_filter: MipmapFilterMode::Linear,
			mip_level_count,
		});
		
		(texture, sampler)
	}
}

pub struct CameraTransformer;

impl System for CameraTransformer {
	type CompQuery = (
		CompRead<PlayerControlled>,
		CompRead<Position>,
		CompRead<Rotation>,
		CompRead<Velocity>,
	);
	type ResQuery = (ResWrite<Camera>, ResRead<PartialTick>);
	
	fn operate(
		&mut self,
		entry: <Self::CompQuery as CompQuery>::Item<'_>,
		res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
	) -> Option<Vec<Command>> {
		res.0.transform(entry.2, entry.3, entry.4, res.1);
		
		None
	}
}