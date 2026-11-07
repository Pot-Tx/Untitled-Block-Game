use crate::game::{AssetId, ModRegistry};
use crate::render::canvas::Color;
use crate::render::*;
use crate::util::OnceInit;
use glam::u32;
use log::error;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

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

/// A render pipeline together with the bind groups it draws with, built for the
/// shape of the pass it is begun in.
pub struct RenderBatch<S: BatchSignature, V: Vertex, I: Inst, C: Color, D: Depth> {
    pub pipeline: RenderPipeline,
    _s_marker: PhantomData<S>,
    _v_marker: PhantomData<V>,
    _i_marker: PhantomData<I>,
    _c_marker: PhantomData<C>,
    _d_marker: PhantomData<D>,
}

/// The name, shader, pipeline targets and draw state of a [`RenderBatch`].
pub struct RenderBatchConfig<'a, C: Color, D: Depth> {
    pub name: &'a str,
    /// Name of the shader file to load from the merged `assets/shaders`
    /// directory.
    pub shader: &'a str,
    /// The formats of the color attachments, in attachment order.
    pub color_formats: C::Formats,
    /// The format of the depth attachment, or `None` when the pass has none.
    pub depth_format: D::Format,
    /// The face culling mode of the pipeline.
    pub cull: Option<Face>,
    /// The blend state of the first color attachment.
    pub blend: Option<BlendState>,
    /// The blend state of the second color attachment, when there is one.
    pub secondary_blend: Option<BlendState>,
    pub topology: PrimitiveTopology,
    /// Whether the pipeline writes depth. Only read when the pass has a depth
    /// attachment.
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
    /// Name of the shader file to load from the merged `assets/shaders`
    /// directory.
    pub shader: &'a str,
}

impl<S: BatchSignature, V: Vertex, I: Inst, C: Color, D: Depth>
    FromConfig<RenderBatchConfig<'_, C, D>> for RenderBatch<S, V, I, C, D>
{
    type Base = Canvas;

    fn new(base: &Self::Base, config: &RenderBatchConfig<C, D>) -> Self {
        let device = &base.device;

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from(format!("{}_pipeline_layout", config.name).as_str()),
            bind_group_layouts: &S::layouts(base).iter().map(Some).collect::<Vec<_>>(),
            immediate_size: 0,
        });

        let shader_src = shader_source(Path::new(&format!("{}.wgsl", config.shader)))
            .unwrap_or_else(|e| panic!("failed to read shader file {}.wgsl: {e}", config.shader));
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Label::from(format!("{}_shader", config.shader).as_str()),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        // Color pipelines always need a fragment stage. A depth pipeline needs
        // one only when its depth mode performs a fragment-level test.
        let color_formats = C::formats(config.color_formats);
        let targets = (!color_formats.is_empty()).then(|| {
            color_formats
                .iter()
                .enumerate()
                .map(|(index, &format)| {
                    Some(ColorTargetState {
                        format,
                        blend: match index {
                            0 => config.blend,
                            1 => config.secondary_blend,
                            _ => None,
                        },
                        write_mask: ColorWrites::ALL,
                    })
                })
                .collect::<Vec<_>>()
        });

        let fragment = (targets.is_some() || D::FRAGMENT).then(|| FragmentState {
            module: &shader_module,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: targets.as_deref().unwrap_or(&[]),
        });

        let depth_format = D::format(config.depth_format);
        let depth_stencil = depth_format.map(|format| DepthStencilState {
            format,
            depth_write_enabled: Some(config.depth_write),
            // Depth is reversed, matching the projection built by
            // `Trans4::projection`.
            depth_compare: Some(CompareFunction::Greater),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Label::from(format!("{}_pipeline", config.name).as_str()),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[V::LAYOUT, I::layout::<V>()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment,
            primitive: PrimitiveState {
                topology: config.topology,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: config.cull,
                unclipped_depth: false,
                polygon_mode: PolygonMode::default(),
                conservative: false,
            },
            depth_stencil,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            _s_marker: PhantomData,
            _v_marker: PhantomData,
            _i_marker: PhantomData,
            _c_marker: PhantomData,
            _d_marker: PhantomData,
        }
    }
}

impl<S: BatchSignature, V: Vertex, I: Inst, C: Color, D: Depth> RenderBatch<S, V, I, C, D> {
    /// Binds the pipeline of this batch, which a pass only accepts when it has
    /// the shape the batch was built for.
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

        let shader_src = shader_source(Path::new(&format!("{}.wgsl", config.shader)))
            .unwrap_or_else(|e| panic!("failed to read shader file {}.wgsl: {e}", config.shader));
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

/// The sources of the shaders, expanded when the renderer starts and read by
/// every pipeline that is built after it.
static SHADER_SOURCES: OnceInit<HashMap<PathBuf, String>> = OnceInit::new();

/// Reads the shaders of the mods, so that the pipelines have their sources.
///
/// The shaders are content, so a mod may override or add any of them; they are
/// read once, before the first pipeline is built, because a pipeline needs the
/// source of its shader.
pub(crate) fn init_shaders(mods: &ModRegistry) {
    match load_shaders(mods) {
        Ok(sources) => SHADER_SOURCES.init(sources),
        Err(e) => {
            error!("failed to load the shaders: {e}");
            SHADER_SOURCES.init(HashMap::new());
        }
    }
}

fn load_shaders(mods: &ModRegistry) -> anyhow::Result<HashMap<PathBuf, String>> {
    let mut sources = HashMap::new();

    for asset in mods.files(Path::new(SHADER_DIR))? {
        let mut stack = Vec::new();
        sources.insert(asset.path.clone(), read_shader(mods, &asset, &mut stack)?);
    }

    Ok(sources)
}

/// The source of the shader `path`, with the includes already expanded.
pub(crate) fn shader_source(path: &Path) -> anyhow::Result<String> {
    let path = Path::new(SHADER_DIR).join(path);

    SHADER_SOURCES
        .get(&path)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no loaded mod provides the shader {}", path.display()))
}

/// The directory of the shaders inside a mod.
const SHADER_DIR: &str = "assets/shaders";

/// Reads the shader of `asset` and expands its includes in place.
///
/// Includes are deliberately small: a line of the form
/// `//#include "name.wgsl"` is replaced by the file of that name next to the
/// including shader. The assets of the mods form one namespace, so a mod
/// overrides an included file as it overrides any other.
fn read_shader(
    mods: &ModRegistry,
    asset: &AssetId,
    stack: &mut Vec<AssetId>,
) -> anyhow::Result<String> {
    if stack.contains(asset) {
        anyhow::bail!("shader include cycle through {}", asset.display());
    }

    stack.push(asset.clone());
    let source = mods.text(asset)?;
    let dir = asset.path.parent().unwrap_or(Path::new("."));
    let mut expanded = String::with_capacity(source.len());

    for line in source.lines() {
        if let Some(include) = line.trim().strip_prefix("//#include ") {
            let name = include
                .strip_prefix('"')
                .and_then(|name| name.strip_suffix('"'))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "malformed include in {}: expected //#include \"file.wgsl\"",
                        asset.display()
                    )
                })?;
            let included = mods.asset(&dir.join(name))?;
            expanded.push_str(&read_shader(mods, &included, stack)?);
        } else {
            expanded.push_str(line);
        }

        expanded.push('\n');
    }

    stack.pop();
    Ok(expanded)
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

/// A batch whose signature declares a single bind group layout.
impl<S: BindSignature> BatchSignature for S {
    type Param<'a> = &'a BindSet<S>;

    fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout> {
        vec![S::layout(canvas)]
    }
}

/// Implements [`BatchSignature`] for a tuple of signatures, together with the
/// [`BatchParam`] the matching tuple of bind sets is pushed as.
///
/// Tuples of bind sets are pushed in the order they are written, so that the
/// group indices match the binding order of the signature; every signature is
/// written as the pair of its type and the name its bind set is bound to.
macro_rules! impl_batch_signature {
    ($($sig:ident($var:ident)),+ $(,)?) => {
        impl<$($sig: BindSignature),+> BatchParam for ($(&BindSet<$sig>,)+) {
            fn bind_groups(&self) -> Vec<&BindGroup> {
                let ($($var,)+) = *self;

                vec![$(&$var.bind_group,)+]
            }
        }

        impl<$($sig: BindSignature),+> BatchSignature for ($($sig,)+) {
            type Param<'a> = ($(&'a BindSet<$sig>,)+);

            fn layouts(canvas: &Canvas) -> Vec<BindGroupLayout> {
                vec![$($sig::layout(canvas),)+]
            }
        }
    };
}

impl_batch_signature!(S(s), T(t));
impl_batch_signature!(S(s), T(t), U(u));
impl_batch_signature!(S(s), T(t), U(u), V(v));
