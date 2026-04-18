use crate::render::*;
use crate::util::collection::Registry;
use bytemuck::{Pod, Zeroable};
use glam::{bool, f32, u32, u8};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::Read;
use wgpu::wgt::TexelCopyTextureInfo as TexelCopyTextureInfoBase;
use wgpu::*;

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
    ) -> BindSet<TextureSampler> {
        let width = self.get(0).width;
        let height = self.get(0).height;
        let len = self.items.len() as u32;
        let mip_level_count = width.min(height).ilog2() + 1;

        let mip_name = &format!("{}_mip", name);

        let storage_texture = Texture::new(
            canvas,
            &TextureConfig {
                name: mip_name,
                texs: &self.items,
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
                texs: &self.items,
                width,
                height,
                mip_level_count,
                storage: false,
            },
        );

        let src_bind = BindSet::<MipSource>::new(
            canvas,
            &BindSetConfig {
                name,
                content: &TextureView::new(
                    &texture,
                    &TextureViewConfig {
                        name: mip_name,
                        dimension: TextureViewDimension::D2Array,
                        mip_level: None,
                    },
                ),
            },
        );

        let batch = ComputeBatch::<(MipSource, MipDestination)>::new(
            canvas,
            &ComputeBatchConfig {
                name: mip_name,
                shader: "mipmap",
            },
        );

        let mut frame = canvas.begin();

        for level in 1..mip_level_count {
            let width = width >> level;
            let height = height >> level;

            let dst_bind = BindSet::<MipDestination>::new(
                canvas,
                &BindSetConfig {
                    name,
                    content: (
                        &TextureView::new(
                            &storage_texture,
                            &TextureViewConfig {
                                name: mip_name,
                                dimension: TextureViewDimension::D2Array,
                                mip_level: Some(level),
                            },
                        ),
                        &Buffer::new(
                            canvas,
                            &BufferConfig {
                                name: mip_name,
                                init: BufferInit::Content(&[MipParams {
                                    width,
                                    height,
                                    level,
                                    _pad: 0,
                                }]),
                                usage: BufferUsages::UNIFORM,
                            },
                        ),
                    ),
                },
            );

            frame.compute(&ComputeDescriptor { name: mip_name }, |mut pass| {
                batch.begin(&mut pass);
                batch.push(&mut pass, (&src_bind, &dst_bind));
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

        let texture_view = TextureView::new(
            &texture,
            &TextureViewConfig {
                name,
                dimension: TextureViewDimension::D2Array,
                mip_level: None,
            },
        );

        let sampler = Sampler::new(
            canvas,
            &SamplerConfig {
                name,
                address_mode: AddressMode::Repeat,
                mipmap_filter: MipmapFilterMode::Linear,
                mip_level_count,
            },
        );

        BindSet::new(
            canvas,
            &BindSetConfig {
                name,
                content: (&texture_view, &sampler),
            },
        )
    }
}

pub struct TextureSampler;

struct MipSource;

struct MipDestination;

impl BindSignature for TextureSampler {
    const NAME: &'static str = "texture_sampler";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
    ];
    type Content<'a> = (&'a TextureView, &'a Sampler);
}

impl BindSignature for MipSource {
    const NAME: &'static str = "mip_src";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }];
    type Content<'a> = &'a TextureView;
}

impl BindSignature for MipDestination {
    const NAME: &'static str = "mip_dst";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2Array,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    type Content<'a> = (&'a TextureView, &'a Buffer);
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
