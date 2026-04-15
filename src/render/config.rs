use crate::render::*;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub trait FromConfig<C> {
	type Base;
	
	fn new(base: &Self::Base, config: &C) -> Self;
}

pub struct SamplerConfig<'a> {
	pub name: &'a str,
	pub address_mode: AddressMode,
	pub mipmap_filter: MipmapFilterMode,
	pub mip_level_count: u32,
}

pub struct TextureConfig<'a> {
	pub name: &'a str,
	pub texs: &'a Vec<Tex>,
	pub width: u32,
	pub height: u32,
	pub mip_level_count: u32,
	pub storage: bool,
}

pub struct TextureViewConfig<'a> {
	pub name: &'a str,
	pub dimension: TextureViewDimension,
	pub mip_level: Option<u32>,
}

pub struct BufferConfig<'a, T: Pod + Zeroable> {
	pub name: &'a str,
	pub content: &'a [T],
	pub usage: BufferUsages,
}

impl FromConfig<SamplerConfig<'_>> for Sampler {
	type Base = Canvas;
	
	#[inline]
	fn new(base: &Self::Base, config: &SamplerConfig) -> Self {
		base.device.create_sampler(&SamplerDescriptor {
			label: Label::from(format!("{}_sampler", config.name).as_str()),
			address_mode_u: config.address_mode,
			address_mode_v: config.address_mode,
			address_mode_w: config.address_mode,
			mag_filter: FilterMode::default(),
			min_filter: FilterMode::default(),
			mipmap_filter: config.mipmap_filter,
			lod_min_clamp: 0.0,
			lod_max_clamp: (config.mip_level_count - 1) as f32,
			compare: None,
			anisotropy_clamp: 1,
			border_color: None,
		})
	}
}

impl FromConfig<TextureConfig<'_>> for Texture {
	type Base = Canvas;
	
	#[inline]
	fn new(base: &Self::Base, config: &TextureConfig) -> Self {
		let device = &base.device;
		let queue = &base.queue;
		let textures = config.texs;
		
		let size = Extent3d {
			width: config.width,
			height: config.height,
			depth_or_array_layers: textures.len() as u32,
		};
		let single_size = Extent3d {
			width: config.width,
			height: config.height,
			depth_or_array_layers: 1,
		};
		
		let (format, usage) = match config.storage {
			false => (
				TextureFormat::Rgba8UnormSrgb,
				TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
			),
			true => (
				TextureFormat::Rgba8Unorm,
				TextureUsages::TEXTURE_BINDING
					| TextureUsages::STORAGE_BINDING
					| TextureUsages::COPY_DST
					| TextureUsages::COPY_SRC,
			),
		};
		
		let texture = device.create_texture(&TextureDescriptor {
			label: Label::from(format!("{}_texture", config.name).as_str()),
			size,
			mip_level_count: config.mip_level_count,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format,
			usage,
			view_formats: &[],
		});
		
		for i in 0..textures.len() {
			queue.write_texture(
				TexelCopyTextureInfoBase {
					texture: &texture,
					mip_level: 0,
					origin: Origin3d {
						x: 0,
						y: 0,
						z: i as u32,
					},
					aspect: TextureAspect::All,
				},
				&textures[i].data,
				TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(4 * config.width),
					rows_per_image: Some(config.height),
				},
				single_size,
			);
		}
		
		texture
	}
}

impl FromConfig<TextureViewConfig<'_>> for TextureView {
	type Base = Texture;
	
	#[inline]
	fn new(base: &Self::Base, config: &TextureViewConfig<'_>) -> Self {
		let (base_mip_level, mip_level_count) = match config.mip_level {
			Some(mip_level) => (mip_level, Some(1)),
			None => (0, None),
		};
		
		base.create_view(&TextureViewDescriptor {
			label: Label::from(format!("{}_texture_view", config.name).as_str()),
			format: None,
			dimension: Some(config.dimension),
			usage: None,
			aspect: TextureAspect::default(),
			base_mip_level,
			mip_level_count,
			base_array_layer: 0,
			array_layer_count: None,
		})
	}
}

impl<T: Pod + Zeroable> FromConfig<BufferConfig<'_, T>> for Buffer {
	type Base = Canvas;
	
	#[inline]
	fn new(base: &Self::Base, config: &BufferConfig<T>) -> Self {
		base.device.create_buffer_init(&util::BufferInitDescriptor {
			label: Label::from(format!("{}_buffer", config.name).as_str()),
			contents: bytemuck::cast_slice(config.content),
			usage: config.usage,
		})
	}
}
