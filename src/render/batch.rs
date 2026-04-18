use crate::render::*;
use glam::u32;
use std::fs;
use std::marker::PhantomData;

pub trait BatchParam {
    fn bind_groups(&self) -> Vec<&BindGroup>;
}

pub trait BatchSignature: 'static {
    type Param<'a>: BatchParam;

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout>;
}

pub struct RenderBatch<S: BatchSignature, V: Vertex, I: Inst> {
    pub pipeline: RenderPipeline,
    _s_marker: PhantomData<S>,
    _v_marker: PhantomData<V>,
    _i_marker: PhantomData<I>,
}

pub struct RenderBatchConfig<'a> {
    pub name: &'a str,
    pub shader: &'a str,
    pub translucent: bool,
    pub topology: PrimitiveTopology,
    pub depth_write: bool,
}

pub struct ComputeBatch<S: BatchSignature> {
    pub pipeline: ComputePipeline,
    _marker: PhantomData<S>,
}

pub struct ComputeBatchConfig<'a> {
    pub name: &'a str,
    pub shader: &'a str,
}

impl<S: BatchSignature, V: Vertex, I: Inst> FromConfig<RenderBatchConfig<'_>>
    for RenderBatch<S, V, I>
{
    type Base = Canvas;

    fn new(base: &Self::Base, config: &RenderBatchConfig) -> Self {
        let device = &base.device;
        let surface_config = &base.surface_config;

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from(format!("{}_pipeline_layout", config.name).as_str()),
            bind_group_layouts: &S::layouts(base).iter().map(|l| Some(l)).collect::<Vec<_>>(),
            immediate_size: 0,
        });

        let shader_src = fs::read_to_string(format!("assets/shaders/{}.wgsl", config.shader))
            .expect(&format!(
                "Failed to read shader file {}.wgsl",
                config.shader
            ));
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Label::from(format!("{}_shader", config.shader).as_str()),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let targets = match config.translucent {
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
            label: Label::from(format!("{}_pipeline", config.name).as_str()),
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
                topology: config.topology,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::default(),
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(config.depth_write),
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
            _s_marker: PhantomData,
            _v_marker: PhantomData,
            _i_marker: PhantomData,
        }
    }
}

impl<S: BatchSignature, V: Vertex, I: Inst> RenderBatch<S, V, I> {
    pub fn begin(&self, pass: &mut RenderPass) {
        pass.set_pipeline(&self.pipeline);
    }

    pub fn push(&self, pass: &mut RenderPass, args: S::Param<'_>) {
        for (idx, &bind_group) in args.bind_groups().iter().enumerate() {
            pass.set_bind_group(idx as u32, bind_group, &[]);
        }
    }

    pub fn draw(&self, pass: &mut RenderPass, item: &impl Render<V, I>) {
        let items = item.rendered();

        for item in items {
            pass.set_vertex_buffer(0, item.geometry.vertex_buffer.slice(..));
            pass.set_index_buffer(item.geometry.index_buffer.slice(..), IndexFormat::Uint16);
            pass.set_vertex_buffer(1, item.instances.instance_buffer.slice(..));

            pass.draw_indexed(
                0..item.geometry.index_count,
                0,
                0..item.instances.instance_count,
            );
        }
    }
}

impl<S: BatchSignature> FromConfig<ComputeBatchConfig<'_>> for ComputeBatch<S> {
    type Base = Canvas;

    fn new(base: &Self::Base, config: &ComputeBatchConfig) -> Self {
        let device = &base.device;

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from(format!("{}_pipeline_layout", config.name).as_str()),
            bind_group_layouts: &S::layouts(base).iter().map(|l| Some(l)).collect::<Vec<_>>(),
            immediate_size: 0,
        });

        let shader_src = fs::read_to_string(format!("assets/shaders/{}.wgsl", config.shader))
            .expect(&format!(
                "Failed to read shader file {}.wgsl",
                config.shader
            ));
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Label::from(format!("{}_shader", config.shader).as_str()),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from(format!("{}_pipeline", config.name).as_str()),
            layout: Some(&layout),
            module: &shader_module,
            entry_point: Some("cs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            _marker: PhantomData,
        }
    }
}

impl<S: BatchSignature> ComputeBatch<S> {
    pub fn begin(&self, pass: &mut ComputePass) {
        pass.set_pipeline(&self.pipeline);
    }

    pub fn push(&self, pass: &mut ComputePass, args: S::Param<'_>) {
        for (idx, &bind_group) in args.bind_groups().iter().enumerate() {
            pass.set_bind_group(idx as u32, bind_group, &[]);
        }
    }

    pub fn dispatch(&self, pass: &mut ComputePass, x: u32, y: u32, z: u32) {
        pass.dispatch_workgroups(x, y, z);
    }
}

impl<S: BindSignature> BatchParam for &BindSet<S> {
    fn bind_groups(&self) -> Vec<&BindGroup> {
        vec![&self.bind_group]
    }
}

impl<S: BindSignature, T: BindSignature> BatchParam for (&BindSet<S>, &BindSet<T>) {
    fn bind_groups(&self) -> Vec<&BindGroup> {
        vec![&self.0.bind_group, &self.1.bind_group]
    }
}

impl<S: BindSignature, T: BindSignature, U: BindSignature> BatchParam
    for (&BindSet<S>, &BindSet<T>, &BindSet<U>)
{
    fn bind_groups(&self) -> Vec<&BindGroup> {
        vec![&self.0.bind_group, &self.1.bind_group, &self.2.bind_group]
    }
}

impl<S: BindSignature, T: BindSignature, U: BindSignature, V: BindSignature> BatchParam
    for (&BindSet<S>, &BindSet<T>, &BindSet<U>, &BindSet<V>)
{
    fn bind_groups(&self) -> Vec<&BindGroup> {
        vec![
            &self.0.bind_group,
            &self.1.bind_group,
            &self.2.bind_group,
            &self.3.bind_group,
        ]
    }
}

impl<S: BindSignature> BatchSignature for S {
    type Param<'a> = &'a BindSet<S>;

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout> {
        vec![S::layout(canvas)]
    }
}

impl<S: BindSignature, T: BindSignature> BatchSignature for (S, T) {
    type Param<'a> = (&'a BindSet<S>, &'a BindSet<T>);

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout> {
        vec![S::layout(canvas), T::layout(canvas)]
    }
}

impl<S: BindSignature, T: BindSignature, U: BindSignature> BatchSignature for (S, T, U) {
    type Param<'a> = (&'a BindSet<S>, &'a BindSet<T>, &'a BindSet<U>);

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout> {
        vec![S::layout(canvas), T::layout(canvas), U::layout(canvas)]
    }
}

impl<S: BindSignature, T: BindSignature, U: BindSignature, V: BindSignature> BatchSignature
    for (S, T, U, V)
{
    type Param<'a> = (
        &'a BindSet<S>,
        &'a BindSet<T>,
        &'a BindSet<U>,
        &'a BindSet<V>,
    );

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout> {
        vec![
            S::layout(canvas),
            T::layout(canvas),
            U::layout(canvas),
            V::layout(canvas),
        ]
    }
}
