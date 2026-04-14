use crate::render::*;
use bytemuck::{Pod, Zeroable};
use std::fs;
use wgpu::util::DeviceExt;

pub struct RenderPipelineConfig<'a> {
    pub name: &'a str,
    pub bind_group_layouts: Vec<&'a BindGroupLayout>,
    pub shader_name: &'a str,
    pub translucent: bool,
    pub vertex_buffer_layouts: &'a [VertexBufferLayout<'a>],
    pub topology: PrimitiveTopology,
    pub depth_write: bool,
}

pub struct ComputePipelineConfig<'a> {
    pub name: &'a str,
    pub bind_group_layouts: Vec<&'a BindGroupLayout>,
    pub shader_name: &'a str,
}

pub enum BindGroupEntryConfig<'a> {
    TextureView(
        &'a TextureView,
        TextureViewDimension,
        Option<StorageTextureAccess>,
    ),
    Sampler(&'a Sampler),
    Buffer(&'a Buffer, BufferBindingType),
}

pub struct BindGroupConfig<'a> {
    pub name: &'a str,
    pub items: Vec<(ShaderStages, BindGroupEntryConfig<'a>)>,
}

pub struct RenderPassConfig<'a> {
    pub name: &'a str,
    pub color_load: LoadOp<Color>,
    pub depth_load: LoadOp<f32>,
}

pub struct ComputePassConfig<'a> {
    pub name: &'a str,
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
    pub texture: &'a Texture,
    pub dimension: TextureViewDimension,
    pub mip_level: Option<u32>,
}

pub struct BufferConfig<'a, T: Pod + Zeroable> {
    pub name: &'a str,
    pub content: &'a [T],
    pub usage: BufferUsages,
}

impl RenderPipelineConfig<'_> {
    pub fn layout(&self, canvas: &Canvas) -> PipelineLayout {
        let device = &canvas.device;

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from(format!("{}_pipeline_layout", self.name).as_str()),
            bind_group_layouts: &self
                .bind_group_layouts
                .iter()
                .map(|&l| Some(l))
                .collect::<Vec<_>>(),
            immediate_size: 0,
        });

        layout
    }

    pub fn create(&self, canvas: &Canvas) -> RenderPipeline {
        let device = &canvas.device;
        let surface_config = &canvas.surface_config;

        let layout = self.layout(canvas);

        let shader_src =
            fs::read_to_string(format!("assets/shaders/{}.wgsl", self.shader_name)).expect(
                &format!("Failed to read shader file {}.wgsl", self.shader_name),
            );
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Label::from(format!("{}_shader", self.shader_name).as_str()),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let targets = if self.translucent {
            [Some(ColorTargetState {
                format: surface_config.format,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::SrcAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                }),
                write_mask: ColorWrites::ALL,
            })]
        } else {
            [Some(surface_config.format.into())]
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Label::from(format!("{}_pipeline", self.name).as_str()),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: self.vertex_buffer_layouts,
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            primitive: PrimitiveState {
                topology: self.topology,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::default(),
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(self.depth_write),
                depth_compare: Some(CompareFunction::Greater),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        pipeline
    }
}

impl ComputePipelineConfig<'_> {
    pub fn layout(&self, canvas: &Canvas) -> PipelineLayout {
        let device = &canvas.device;

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from(format!("{}_pipeline_layout", self.name).as_str()),
            bind_group_layouts: &self
                .bind_group_layouts
                .iter()
                .map(|&l| Some(l))
                .collect::<Vec<_>>(),
            immediate_size: 0,
        });

        layout
    }

    pub fn create(&self, canvas: &Canvas) -> ComputePipeline {
        let device = &canvas.device;

        let layout = self.layout(canvas);

        let shader_src =
            fs::read_to_string(format!("assets/shaders/{}.wgsl", self.shader_name)).expect(
                &format!("Failed to read shader file {}.wgsl", self.shader_name),
            );
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Label::from(format!("{}_shader", self.shader_name).as_str()),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from(format!("{}_pipeline", self.name).as_str()),
            layout: Some(&layout),
            module: &shader_module,
            entry_point: Some("cs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        pipeline
    }
}

impl BindGroupConfig<'_> {
    pub fn layout(&self, canvas: &Canvas) -> BindGroupLayout {
        let device = &canvas.device;

        let mut layout_entries = Vec::new();
        for i in 0..self.items.len() {
            let (visibility, item) = &self.items[i];
            layout_entries.push(item.layout(i as u32, *visibility));
        }
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Label::from(format!("{}_bind_group_layout", self.name).as_str()),
            entries: &layout_entries,
        });

        layout
    }

    pub fn create(&self, canvas: &Canvas) -> BindGroup {
        let device = &canvas.device;

        let layout = self.layout(canvas);
        let mut entries = Vec::new();
        self.items.iter().enumerate().for_each(|(i, (_, item))| {
            entries.push(item.create(i as u32));
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Label::from(format!("{}_bind_group", self.name).as_str()),
            layout: &layout,
            entries: &entries,
        });

        bind_group
    }
}

impl<'a> BindGroupEntryConfig<'a> {
    pub fn layout(&self, idx: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        let ty = match self {
            &Self::Sampler(_) => BindingType::Sampler(SamplerBindingType::Filtering),

            &Self::TextureView(_, view_dimension, storage_access) => match storage_access {
                Some(access) => BindingType::StorageTexture {
                    access,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension,
                },

                None => BindingType::Texture {
                    multisampled: false,
                    view_dimension,
                    sample_type: TextureSampleType::Float { filterable: true },
                },
            },

            &Self::Buffer(buffer, buffer_binding_type) => BindingType::Buffer {
                ty: buffer_binding_type,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(buffer.size()),
            },
        };

        BindGroupLayoutEntry {
            binding: idx,
            visibility,
            ty,
            count: None,
        }
    }

    pub fn create(&self, idx: u32) -> BindGroupEntry<'a> {
        let resource = match *self {
            Self::Sampler(sampler) => BindingResource::Sampler(sampler),
            Self::TextureView(texture_view, ..) => BindingResource::TextureView(texture_view),
            Self::Buffer(buffer, _) => buffer.as_entire_binding(),
        };

        BindGroupEntry {
            binding: idx,
            resource,
        }
    }
}

impl RenderPassConfig<'_> {
    pub fn create<'a>(&self, canvas: &'a mut Canvas) -> RenderPass<'a> {
        if let Some(texture_view) = &canvas.texture_view
            && let Some(command_encoder) = &mut canvas.command_encoder
        {
            let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Label::from(format!("{}_render_pass", self.name).as_str()),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: self.color_load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &canvas.depth_texture_view,
                    depth_ops: Some(Operations {
                        load: self.depth_load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass
        } else {
            panic!("Failed to create Render Pass. Make sure to call Canvas::begin first!");
        }
    }
}

impl ComputePassConfig<'_> {
    pub fn create<'a>(&self, canvas: &'a mut Canvas) -> ComputePass<'a> {
        if let Some(command_encoder) = &mut canvas.command_encoder {
            let compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Label::from(format!("{}_compute_pass", self.name).as_str()),
                timestamp_writes: None,
            });

            compute_pass
        } else {
            panic!("Failed to create Compute Pass. Make sure to call Canvas::begin first!");
        }
    }
}

impl SamplerConfig<'_> {
    pub fn create(&self, canvas: &Canvas) -> Sampler {
        let device = &canvas.device;

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Label::from(format!("{}_sampler", self.name).as_str()),
            address_mode_u: self.address_mode,
            address_mode_v: self.address_mode,
            address_mode_w: self.address_mode,
            mag_filter: FilterMode::default(),
            min_filter: FilterMode::default(),
            mipmap_filter: self.mipmap_filter,
            lod_min_clamp: 0.0,
            lod_max_clamp: (self.mip_level_count - 1) as f32,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        sampler
    }
}

impl TextureConfig<'_> {
    pub fn create(&self, canvas: &Canvas) -> Texture {
        let device = &canvas.device;
        let queue = &canvas.queue;
        let textures = self.texs;

        let size = Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: textures.len() as u32,
        };
        let single_size = Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        let (format, usage) = match self.storage {
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
            label: Label::from(format!("{}_texture", self.name).as_str()),
            size,
            mip_level_count: self.mip_level_count,
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
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
                single_size,
            );
        }

        texture
    }
}

impl TextureViewConfig<'_> {
    pub fn create(&self) -> TextureView {
        let (base_mip_level, mip_level_count) = match self.mip_level {
            Some(mip_level) => (mip_level, Some(1)),
            None => (0, None),
        };

        let texture_view = self.texture.create_view(&TextureViewDescriptor {
            label: Label::from(format!("{}_texture_array_view", self.name).as_str()),
            format: None,
            dimension: Some(self.dimension),
            usage: None,
            aspect: TextureAspect::default(),
            base_mip_level,
            mip_level_count,
            base_array_layer: 0,
            array_layer_count: None,
        });

        texture_view
    }
}

impl<T: Pod + Zeroable> BufferConfig<'_, T> {
    pub fn create(&self, canvas: &Canvas) -> Buffer {
        let device = &canvas.device;

        let buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Label::from(format!("{}_buffer", self.name).as_str()),
            contents: bytemuck::cast_slice(self.content),
            usage: self.usage,
        });

        buffer
    }
}
