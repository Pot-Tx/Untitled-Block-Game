use crate::render::*;
use glam::u32;
use std::fs;
use std::marker::PhantomData;

/// The bind groups a batch is drawn with, in binding order.
pub trait BatchParam {
    fn bind_groups(&self) -> Vec<&BindGroup>;
}

/// The bind group layouts of a batch, and the parameters it is pushed with.
///
/// Implemented for a single [`BindSignature`] and for tuples of them, so that a
/// batch can take any number of bind groups.
pub trait BatchSignature: 'static {
    type Param<'a>: BatchParam;

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout>;
}

/// A render pipeline together with the bind groups it draws with.
pub struct RenderBatch<S: BatchSignature, V: Vertex, I: Inst> {
    pub pipeline: RenderPipeline,
    _s_marker: PhantomData<S>,
    _v_marker: PhantomData<V>,
    _i_marker: PhantomData<I>,
}

/// The name, shader, blend and depth settings of a [`RenderBatch`].
pub struct RenderBatchConfig<'a> {
    pub name: &'a str,
    /// Name of the shader file to load from `assets/shaders`.
    pub shader: &'a str,
    /// Whether the batch is alpha blended instead of drawing opaque geometry.
    pub translucent: bool,
    pub topology: PrimitiveTopology,
    pub depth_write: bool,
}

/// A compute pipeline together with the bind groups it dispatches with.
pub struct ComputeBatch<S: BatchSignature> {
    pub pipeline: ComputePipeline,
    _marker: PhantomData<S>,
}

/// The name and shader of a [`ComputeBatch`].
pub struct ComputeBatchConfig<'a> {
    pub name: &'a str,
    /// Name of the shader file to load from `assets/shaders`.
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
            bind_group_layouts: &S::layouts(base).iter().map(Some).collect::<Vec<_>>(),
            immediate_size: 0,
        });

        let shader_src = fs::read_to_string(format!("assets/shaders/{}.wgsl", config.shader))
            .unwrap_or_else(|_| panic!("failed to read shader file {}.wgsl", config.shader));
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
                // Depth is reversed, matching the projection built by
                // `Trans4::projection`.
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
    /// Binds the pipeline of this batch.
    pub fn begin(&self, pass: &mut RenderPass) {
        pass.set_pipeline(&self.pipeline);
    }

    /// Binds the bind groups of this batch, in order.
    pub fn push(&self, pass: &mut RenderPass, args: S::Param<'_>) {
        for (idx, &bind_group) in args.bind_groups().iter().enumerate() {
            pass.set_bind_group(idx as u32, bind_group, &[]);
        }
    }

    /// Draws every item the geometry reports.
    pub fn draw(&self, pass: &mut RenderPass, item: &impl Render<V, I>) {
        let items = item.rendered();

        for item in items {
            if let Some(slice) = item.indices.into() {
                if V::LAYOUT.is_some() {
                    pass.set_vertex_buffer(0, item.vertices);
                }
                pass.set_index_buffer(slice, IndexFormat::Uint16);
                if I::layout::<V>().is_some() {
                    pass.set_vertex_buffer(1, item.instances);
                }
            }

            pass.draw_indexed(0..item.indices.length, 0, 0..item.instances.length);
        }
    }
}

impl<S: BatchSignature> FromConfig<ComputeBatchConfig<'_>> for ComputeBatch<S> {
    type Base = Canvas;

    fn new(base: &Self::Base, config: &ComputeBatchConfig) -> Self {
        let device = &base.device;

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from(format!("{}_pipeline_layout", config.name).as_str()),
            bind_group_layouts: &S::layouts(base).iter().map(Some).collect::<Vec<_>>(),
            immediate_size: 0,
        });

        let shader_src = fs::read_to_string(format!("assets/shaders/{}.wgsl", config.shader))
            .unwrap_or_else(|_| panic!("failed to read shader file {}.wgsl", config.shader));
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
    /// Binds the pipeline of this batch.
    pub fn begin(&self, pass: &mut ComputePass) {
        pass.set_pipeline(&self.pipeline);
    }

    /// Binds the bind groups of this batch, in order.
    pub fn push(&self, pass: &mut ComputePass, args: S::Param<'_>) {
        for (idx, &bind_group) in args.bind_groups().iter().enumerate() {
            pass.set_bind_group(idx as u32, bind_group, &[]);
        }
    }

    /// Dispatches the given number of workgroups.
    pub fn dispatch(&self, pass: &mut ComputePass, x: u32, y: u32, z: u32) {
        pass.dispatch_workgroups(x, y, z);
    }
}

/// A single bind set is pushed as the only bind group of its layout.
impl<S: BindSignature> BatchParam for &BindSet<S> {
    fn bind_groups(&self) -> Vec<&BindGroup> {
        vec![&self.bind_group]
    }
}

/// Tuples of bind sets are pushed in the order they are written, so that the
/// group indices match the binding order of the signature.
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

/// A batch whose signature declares a single bind group layout.
impl<S: BindSignature> BatchSignature for S {
    type Param<'a> = &'a BindSet<S>;

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout> {
        vec![S::layout(canvas)]
    }
}

/// A batch that takes one bind group layout per element of the tuple.
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
