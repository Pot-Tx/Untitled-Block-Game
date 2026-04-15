use crate::render::*;
use glam::u32;
use std::fs;
use std::marker::PhantomData;

pub struct RenderBatch<'a, V: Vertex, I: Inst> {
	pipeline: RenderPipeline,
	bind_groups: Vec<Option<&'a BindGroup>>,
	_v_marker: PhantomData<V>,
	_i_marker: PhantomData<I>,
}

pub struct RenderBatchDescriptor<'a> {
	pub name: &'a str,
	pub bindings: Vec<Option<&'a Binding>>,
	pub shader_name: &'a str,
	pub translucent: bool,
	pub topology: PrimitiveTopology,
	pub depth_write: bool,
}

pub struct ComputeBatch<'a> {
	pipeline: ComputePipeline,
	bind_groups: Vec<Option<&'a BindGroup>>,
}

pub struct ComputeBatchDescriptor<'a> {
	pub name: &'a str,
	pub bindings: Vec<Option<&'a Binding>>,
	pub shader_name: &'a str,
}

pub struct Binding {
	pub index: u32,
	pub layout: BindGroupLayout,
	pub bind_group: BindGroup,
}

pub struct BindingDescriptor<'a> {
	pub index: u32,
	pub name: &'a str,
	pub entries: Vec<BindingEntryDescriptor<'a>>,
}

pub struct BindingEntryDescriptor<'a> {
	pub visibility: ShaderStages,
	pub item: BindingItem<'a>,
}

pub enum BindingItem<'a> {
	TextureView(
		&'a TextureView,
		TextureViewDimension,
		Option<StorageTextureAccess>,
	),
	Sampler(&'a Sampler),
	Buffer(&'a Buffer, BufferBindingType),
}

impl Binding {
	pub fn new(canvas: &Canvas, desc: &BindingDescriptor) -> Self {
		let device = &canvas.device;
		
		let mut layouts = Vec::new();
		let mut entries = Vec::new();
		
		desc.entries.iter().enumerate().for_each(|(i, entry)| {
			let (ty, resource) = match entry.item {
				BindingItem::Sampler(sampler) => (
					BindingType::Sampler(SamplerBindingType::Filtering),
					BindingResource::Sampler(sampler),
				),
				
				BindingItem::TextureView(view, dim, access) => (
					match access {
						Some(access) => BindingType::StorageTexture {
							access,
							format: TextureFormat::Rgba8Unorm,
							view_dimension: dim,
						},
						
						None => BindingType::Texture {
							multisampled: false,
							view_dimension: dim,
							sample_type: TextureSampleType::Float { filterable: true },
						},
					},
					BindingResource::TextureView(view),
				),
				
				BindingItem::Buffer(buffer, ty) => (
					BindingType::Buffer {
						ty,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(buffer.size()),
					},
					buffer.as_entire_binding()
				)
			};
			
			layouts.push(BindGroupLayoutEntry {
				binding: i as u32,
				visibility: entry.visibility,
				ty,
				count: None,
			});
			
			entries.push(BindGroupEntry {
				binding: i as u32,
				resource,
			});
		});
		
		let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Label::from(format!("{}_bind_group_layout", desc.name).as_str()),
			entries: &layouts,
		});
		
		let bind_group = device.create_bind_group(&BindGroupDescriptor {
			label: Label::from(format!("{}_bind_group", desc.name).as_str()),
			layout: &layout,
			entries: &entries,
		});
		
		Binding {
			index: desc.index,
			layout,
			bind_group,
		}
	}
}

impl<'a, V: Vertex, I: Inst> RenderBatch<'a, V, I> {
	pub fn new(canvas: &Canvas, desc: RenderBatchDescriptor<'a>) -> Self {
		let device = &canvas.device;
		let surface_config = &canvas.surface_config;
		
		let mut layouts = Vec::new();
		let mut bind_groups = Vec::new();
		
		desc
			.bindings
			.iter()
			.enumerate()
			.for_each(|(i, &binding)| {
				let (layout, bind_group) = match binding {
					Some(b) => {
						assert_eq!(i as u32, b.index);
						(Some(&b.layout), Some(&b.bind_group))
					}
					None => (None, None)
				};
				
				layouts.push(layout);
				bind_groups.push(bind_group);
			});
		
		let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Label::from(format!("{}_pipeline_layout", desc.name).as_str()),
			bind_group_layouts: &layouts,
			immediate_size: 0,
		});
		
		let shader_src =
			fs::read_to_string(format!("assets/shaders/{}.wgsl", desc.shader_name)).expect(
				&format!("Failed to read shader file {}.wgsl", desc.shader_name),
			);
		let shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Label::from(format!("{}_shader", desc.shader_name).as_str()),
			source: ShaderSource::Wgsl(shader_src.into()),
		});
		
		let targets = match desc.translucent {
			false => [Some(surface_config.format.into())],
			
			true => [Some(ColorTargetState {
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
			})],
		};
		
		let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
			label: Label::from(format!("{}_pipeline", desc.name).as_str()),
			layout: Some(&layout),
			vertex: VertexState {
				module: &shader_module,
				entry_point: Some("vs_main"),
				buffers: &[V::LAYOUT, I::layout::<V>()],
				compilation_options: PipelineCompilationOptions::default(),
			},
			fragment: Some(FragmentState {
				module: &shader_module,
				entry_point: Some("fs_main"),
				compilation_options: PipelineCompilationOptions::default(),
				targets: &targets,
			}),
			primitive: PrimitiveState {
				topology: desc.topology,
				strip_index_format: None,
				front_face: FrontFace::Ccw,
				cull_mode: Some(Face::Back),
				unclipped_depth: false,
				polygon_mode: PolygonMode::default(),
				conservative: false,
			},
			depth_stencil: Some(DepthStencilState {
				format: TextureFormat::Depth32Float,
				depth_write_enabled: Some(desc.depth_write),
				depth_compare: Some(CompareFunction::Greater),
				stencil: StencilState::default(),
				bias: DepthBiasState::default(),
			}),
			multisample: MultisampleState::default(),
			multiview_mask: None,
			cache: None,
		});
		
		Self {
			pipeline,
			bind_groups,
			_v_marker: PhantomData,
			_i_marker: PhantomData,
		}
	}
	
	pub fn begin(&self, render_pass: &mut RenderPass) {
		render_pass.set_pipeline(&self.pipeline);
		
		for (idx, &bind_group) in self.bind_groups.iter().enumerate() {
			render_pass.set_bind_group(idx as u32, bind_group, &[]);
		}
	}
	
	pub fn draw(&self, render_pass: &mut RenderPass, item: &impl Render<V, I>) {
		let items = item.rendered();
		
		for item in items {
			render_pass.set_vertex_buffer(0, item.geometry.vertex_buffer.slice(..));
			render_pass.set_index_buffer(item.geometry.index_buffer.slice(..), IndexFormat::Uint16);
			render_pass.set_vertex_buffer(1, item.instances.instance_buffer.slice(..));
			
			render_pass.draw_indexed(
				0..item.geometry.index_count,
				0,
				0..item.instances.instance_count,
			);
		}
	}
}

impl<'a> ComputeBatch<'a> {
	pub fn new(canvas: &Canvas, desc: ComputeBatchDescriptor<'a>) -> Self {
		let device = &canvas.device;
		
		let mut layouts = Vec::new();
		let mut bind_groups = Vec::new();
		
		desc
			.bindings
			.iter()
			.enumerate()
			.for_each(|(i, &binding)| {
				let (layout, bind_group) = match binding {
					Some(b) => {
						assert_eq!(i as u32, b.index);
						(Some(&b.layout), Some(&b.bind_group))
					}
					None => (None, None)
				};
				
				layouts.push(layout);
				bind_groups.push(bind_group);
			});
		
		let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Label::from(format!("{}_pipeline_layout", desc.name).as_str()),
			bind_group_layouts: &layouts,
			immediate_size: 0,
		});
		
		let shader_src =
			fs::read_to_string(format!("assets/shaders/{}.wgsl", desc.shader_name)).expect(
				&format!("Failed to read shader file {}.wgsl", desc.shader_name),
			);
		let shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Label::from(format!("{}_shader", desc.shader_name).as_str()),
			source: ShaderSource::Wgsl(shader_src.into()),
		});
		
		let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: Label::from(format!("{}_pipeline", desc.name).as_str()),
			layout: Some(&layout),
			module: &shader_module,
			entry_point: Some("cs_main"),
			compilation_options: PipelineCompilationOptions::default(),
			cache: None,
		});
		
		Self {
			pipeline,
			bind_groups,
		}
	}
	
	pub fn begin(&self, compute_pass: &mut ComputePass) {
		compute_pass.set_pipeline(&self.pipeline);
		
		for (idx, &bind_group) in self.bind_groups.iter().enumerate() {
			compute_pass.set_bind_group(idx as u32, bind_group, &[]);
		}
	}
	
	pub fn dispatch(&self, compute_pass: &mut ComputePass, x: u32, y: u32, z: u32) {
		compute_pass.dispatch_workgroups(x, y, z);
	}
}
