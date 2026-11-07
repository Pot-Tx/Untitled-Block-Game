use crate::game::ModRegistry;
use crate::render::*;
use crate::util::collection::Registry;
use bytemuck::{Pod, Zeroable};
use glam::{bool, f32, u8, u32};
use log::warn;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use wgpu::wgt::TexelCopyTextureInfo as TexelCopyTextureInfoBase;
use wgpu::*;

/// A texture as it is loaded from an image file, in rgba8.
#[derive(Clone)]
pub struct Tex {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// The uniform buffer of the mipmap compute shader.
///
/// `level` selects the mip level to write, and `_pad` keeps the struct aligned
/// to the 16 bytes the shader expects.
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
    /// Decodes a png file into rgba8 pixels.
    pub fn from_png(bytes: &[u8]) -> anyhow::Result<Self> {
        let image = image::load_from_memory(bytes)?.into_rgba8();
        let (width, height) = image.dimensions();
        let data = image.into_raw();

        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Creates a single layer texture and the sampler that reads it.
    pub fn create_texture_sampler(&self, canvas: &Canvas, name: &str) -> BindSet<TextureSampler> {
        let texture = Texture::new(
            canvas,
            &TextureConfig {
                name,
                texs: vec![Some(self)],
                width: self.width,
                height: self.height,
                mip_level_count: 1,
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            },
        );

        let texture_view = TextureView::new(
            &texture,
            &TextureViewConfig {
                name,
                dimension: TextureViewDimension::D2,
                mip_level: None,
            },
        );

        let sampler = Sampler::new(
            canvas,
            &SamplerConfig {
                name,
                filter: FilterMode::Nearest,
                address_mode: AddressMode::Repeat,
                mipmap_filter: MipmapFilterMode::Linear,
                mip_level_count: 1,
                compare: None,
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

pub fn load_texture_array<T>(
    canvas: &Canvas,
    mods: &ModRegistry,
    dir: &str,
    name: &str,
    format: TextureFormat,
    mipmap: bool,
    f: impl Fn(&Tex) -> T,
) -> anyhow::Result<(Registry<T>, Texture, TextureView)> {
    let mut registry = Registry::new();
    let mut texs = Vec::new();

    let mut size = None;
    for (id, asset) in mods.files(Path::new(dir))?.into_iter().enumerate() {
        let name = asset.name()?;
        let tex = Tex::from_png(&mods.read(&asset)?)?;

        if size.unwrap_or((tex.width, tex.height)) == (tex.width, tex.height) {
            size = Some((tex.width, tex.height));
            registry.register(id as u32, name, f(&tex));
            texs.push(Some(tex));
        } else {
            registry.register(id as u32, name.clone(), f(&tex));
            texs.push(None);
            warn!("texture {name} of {} has an incompatible size", asset.display());
        }
    }

    let (texture, texture_view) = create_texture_array(canvas, name, format, mipmap, texs)?;

    Ok((registry, texture, texture_view))
}

/// Uploads `texs` as the layers of the texture array `name`, in the order of their ids,
/// and answers the bind group of the array and its sampler.
///
/// A layer whose size is not the size of the array is read as a `None` and left as it is:
/// a layer is named by its id, which is also its position in the array, so the layers
/// around it keep their place.
pub fn create_texture_array(
    canvas: &Canvas,
    name: &str,
    format: TextureFormat,
    mipmap: bool,
    texs: Vec<Option<Tex>>,
) -> anyhow::Result<(Texture, TextureView)> {
    let (width, height) = texs
        .iter()
        .flatten()
        .next()
        .map(|tex| (tex.width, tex.height))
        .ok_or_else(|| anyhow!("the texture array {} has no layer", name))?;
    let len = texs.len() as u32;
    let limit = canvas.device.limits().max_texture_array_layers;

    if len > limit {
        return Err(anyhow!(
            "the texture array {name} has {len} layers, but the device allows {limit}"
        ));
    }

    let mip_level_count = match mipmap {
        true => width.min(height).ilog2() + 1,
        false => 1,
    };

    let texture = Texture::new(
        canvas,
        &TextureConfig {
            name,
            texs: texs.iter().map(Option::as_ref).collect(),
            width,
            height,
            mip_level_count,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        },
    );

    if mipmap {
        let mip_name = &format!("{}_mip", name);
        let storage_texture = Texture::new(
            canvas,
            &TextureConfig {
                name: mip_name,
                texs: texs.iter().map(Option::as_ref).collect(),
                width,
                height,
                mip_level_count,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC,
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

            frame.compute(mip_name, |mut pass| {
                batch.begin(&mut pass);
                batch.push(&mut pass, (&src_bind, &dst_bind));
                batch.dispatch(&mut pass, width.div_ceil(8), height.div_ceil(8), len);
            });

            frame.0.copy_texture_to_texture(
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
    }

    let texture_view = TextureView::new(
        &texture,
        &TextureViewConfig {
            name,
            dimension: TextureViewDimension::D2Array,
            mip_level: None,
        },
    );

    /*let sampler = Sampler::new(
        canvas,
        &SamplerConfig {
            name,
            address_mode: AddressMode::Repeat,
            mipmap_filter: MipmapFilterMode::Linear,
            mip_level_count,
            compare: None,
        },
    );

    let binding = BindSet::new(
        canvas,
        &BindSetConfig {
            name,
            content: (&texture_view, &sampler),
        },
    );*/

    Ok((texture, texture_view))
}

/// The bind group of a single layer texture and its sampler.
pub struct TextureSampler;

/// The bind group of a texture array and its sampler.
pub struct TextureArraySampler;

/// The bind group of the texture a mipmap pass reads.
struct MipSource;

/// The bind group of the storage texture and parameters a mipmap pass writes.
struct MipDestination;

impl BindSignature for TextureSampler {
    const NAME: &'static str = "texture";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
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

impl BindSignature for TextureArraySampler {
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

/// The name, address mode and mip settings of a sampler.
pub struct SamplerConfig<'a> {
    pub name: &'a str,
    pub filter: FilterMode,
    pub address_mode: AddressMode,
    pub mipmap_filter: MipmapFilterMode,
    pub mip_level_count: u32,
    pub compare: Option<CompareFunction>,
}

/// The name, content, format and usages of a texture array to create.
pub struct TextureConfig<'a> {
    pub name: &'a str,
    pub texs: Vec<Option<&'a Tex>>,
    pub width: u32,
    pub height: u32,
    pub mip_level_count: u32,
    pub format: TextureFormat,
    pub usage: TextureUsages,
}

/// The name, dimension and mip range of a texture view.
pub struct TextureViewConfig<'a> {
    pub name: &'a str,
    pub dimension: TextureViewDimension,
    pub mip_level: Option<u32>,
}

/// The name and the layer of a view of a single layer of an array texture.
///
/// A pass that draws into one cascade of the shadow map needs a view of one
/// layer, which is not what [`TextureViewConfig`] covers.
pub struct LayerViewConfig<'a> {
    pub name: &'a str,
    pub layer: u32,
}

impl FromConfig<LayerViewConfig<'_>> for TextureView {
    type Base = Texture;

    #[inline]
    fn new(base: &Self::Base, config: &LayerViewConfig) -> Self {
        base.create_view(&TextureViewDescriptor {
            label: Label::from(format!("{}_{}_texture_view", config.name, config.layer).as_str()),
            format: None,
            dimension: Some(TextureViewDimension::D2),
            usage: None,
            aspect: TextureAspect::default(),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: config.layer,
            array_layer_count: Some(1),
        })
    }
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
            mag_filter: config.filter,
            min_filter: config.filter,
            mipmap_filter: config.mipmap_filter,
            lod_min_clamp: 0.0,
            lod_max_clamp: (config.mip_level_count - 1) as f32,
            compare: config.compare,
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
        let textures = &config.texs;

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

        let texture = device.create_texture(&TextureDescriptor {
            label: Label::from(format!("{}_texture", config.name).as_str()),
            size,
            mip_level_count: config.mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.format,
            usage: config.usage,
            view_formats: &[],
        });

        for (i, tex) in textures.iter().enumerate() {
            if let Some(tex) = tex {
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
                    &tex.data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * config.width),
                        rows_per_image: Some(config.height),
                    },
                    single_size,
                );
            }
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
