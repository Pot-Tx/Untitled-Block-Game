use crate::ecs::*;
use crate::render::*;
use crate::util::OnceInit;
use glam::{f32, u32};
use log::error;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub const WBOIT_REVEAL_BLEND: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::Zero,
        dst_factor: BlendFactor::OneMinusSrc,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::Zero,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    },
};

static RESOLVE_BATCH: OnceInit<RenderBatch<TextureSampler, (), (), ColorDraw<1>, ()>> =
    OnceInit::new();
static WBOIT_RESOLVE_BATCH: OnceInit<RenderBatch<WboitSource, (), (), ColorDraw<1>, ()>> =
    OnceInit::new();
pub static COLOR_SAMPLER: OnceInit<Sampler> = OnceInit::new();

pub(super) fn init_resolve(canvas: &Canvas) {
    let batch: RenderBatch<TextureSampler, (), (), ColorDraw<1>, ()> = RenderBatch::new(
        canvas,
        &RenderBatchConfig {
            name: "resolve",
            shader: "resolve",
            color_formats: [canvas.surface_config.format],
            depth_format: (),
            cull: Some(Face::Back),
            blend: None,
            secondary_blend: None,
            topology: PrimitiveTopology::TriangleList,
            depth_write: false,
        },
    );
    let wboit_batch: RenderBatch<WboitSource, (), (), ColorDraw<1>, ()> = RenderBatch::new(
        canvas,
        &RenderBatchConfig {
            name: "wboit_resolve",
            shader: "wboit_resolve",
            color_formats: [Canvas::SCENE_COLOR_FORMAT],
            depth_format: (),
            cull: Some(Face::Back),
            blend: Some(BlendState::ALPHA_BLENDING),
            secondary_blend: None,
            topology: PrimitiveTopology::TriangleList,
            depth_write: false,
        },
    );

    let sampler = Sampler::new(
        canvas,
        &SamplerConfig {
            name: "resolve",
            filter: FilterMode::Nearest,
            address_mode: AddressMode::ClampToEdge,
            mipmap_filter: MipmapFilterMode::Nearest,
            mip_level_count: 1,
            compare: None,
        },
    );

    RESOLVE_BATCH.init(batch);
    WBOIT_RESOLVE_BATCH.init(wboit_batch);
    COLOR_SAMPLER.init(sampler);
}

/// The command encoder of the frame that is being recorded.
pub struct Frame(pub CommandEncoder);

/// The window surface, its configuration and the device the game draws with.
pub struct Canvas {
    surface: Surface<'static>,
    pub surface_config: SurfaceConfiguration,
    pub device: Device,
    pub queue: Queue,
    color: Texture,
    depth: Texture,
    wboit_accum: Texture,
    wboit_reveal: Texture,
    /// The view of the color texture the scene is drawn into.
    pub color_view: TextureView,
    /// The view of the depth texture of the scene.
    pub depth_view: TextureView,
    pub wboit_accum_view: TextureView,
    pub wboit_reveal_view: TextureView,
}

impl Resource for Canvas {}

impl Canvas {
    /// The format of the color texture the scene is drawn into.
    pub const SCENE_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
    /// The format of the depth texture of the scene.
    pub const SCENE_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

    /// Creates the device and the swap chain of `window`.
    pub async fn new(window: &'static Window) -> Self {
        let wgpu_instance = Instance::default();
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let surface = wgpu_instance
            .create_surface(window)
            .expect("failed to create surface");
        let adapter = wgpu_instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .expect("failed to find adapter");
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            format: TextureFormat::Rgba8UnormSrgb,
            color_space: SurfaceColorSpace::default(),
            width,
            height,
            present_mode: PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Label::from("device"),
                required_features: Features::empty(),
                required_limits: Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .expect("failed to create device");
        surface.configure(&device, &surface_config);

        let (color, depth, wboit_accum, wboit_reveal) =
            Self::create_textures(&device, width, height);
        let color_view = Self::view(&color, "color");
        let depth_view = Self::view(&depth, "depth");
        let wboit_accum_view = Self::view(&wboit_accum, "wboit_accum");
        let wboit_reveal_view = Self::view(&wboit_reveal, "wboit_reveal");

        Self {
            surface,
            surface_config,
            device,
            queue,
            color,
            depth,
            wboit_accum,
            wboit_reveal,
            color_view,
            depth_view,
            wboit_accum_view,
            wboit_reveal_view,
        }
    }

    /// Reconfigures the surface and recreates the depth texture for a window of
    /// `size`.
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        let width = size.width.max(1);
        let height = size.height.max(1);

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);

        let (color, depth, wboit_accum, wboit_reveal) =
            Self::create_textures(&self.device, width, height);
        self.color_view = Self::view(&color, "color");
        self.depth_view = Self::view(&depth, "depth");
        self.wboit_accum_view = Self::view(&wboit_accum, "wboit_accum");
        self.wboit_reveal_view = Self::view(&wboit_reveal, "wboit_reveal");
        self.color = color;
        self.depth = depth;
        self.wboit_accum = wboit_accum;
        self.wboit_reveal = wboit_reveal;
    }

    /// Starts recording a frame of the scene.
    pub fn begin(&self) -> Frame {
        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Label::from("command_encoder"),
            });

        Frame(encoder)
    }

    /// Submits the recorded commands and presents the frame.
    pub fn end(&self, mut frame: Frame) {
        if let CurrentSurfaceTexture::Success(surface_texture) = self.surface.get_current_texture()
        {
            let view = TextureView::new(
                &surface_texture.texture,
                &TextureViewConfig {
                    name: "color",
                    dimension: TextureViewDimension::D2,
                    mip_level: None,
                },
            );

            let binding = BindSet::<TextureSampler>::new(
                self,
                &BindSetConfig {
                    name: "resolve",
                    content: (&self.color_view, &COLOR_SAMPLER),
                },
            );

            let target = ColorTarget {
                view: &view,
                load: LoadOp::Clear(wgpu::Color::BLACK),
            };

            frame.render::<ColorDraw<1>, ()>("resolve", [target], (), |mut pass| {
                RESOLVE_BATCH.begin(&mut pass);
                RESOLVE_BATCH.push(&mut pass, &binding);
                pass.draw(0..3, 0..1);
            });

            self.queue.submit(Some(frame.0.finish()));
            self.queue.present(surface_texture);
        } else {
            error!("failed to get next surface texture");
        }
    }

    /// Composites the camera-visible WBOIT targets onto the scene color.
    pub(crate) fn resolve_wboit(&self, frame: &mut Frame) {
        let binding = BindSet::<WboitSource>::new(
            self,
            &BindSetConfig {
                name: "wboit_resolve",
                content: (
                    &self.wboit_accum_view,
                    &self.wboit_reveal_view,
                    &COLOR_SAMPLER,
                ),
            },
        );
        let target = ColorTarget {
            view: &self.color_view,
            load: LoadOp::Load,
        };

        frame.render::<ColorDraw<1>, ()>("wboit_resolve", [target], (), |mut pass| {
            WBOIT_RESOLVE_BATCH.begin(&mut pass);
            WBOIT_RESOLVE_BATCH.push(&mut pass, &binding);
            pass.draw(0..3, 0..1);
        });
    }

    /// Creates the view a texture of the scene is drawn into.
    fn view(texture: &Texture, name: &str) -> TextureView {
        TextureView::new(
            texture,
            &TextureViewConfig {
                name,
                dimension: TextureViewDimension::D2,
                mip_level: None,
            },
        )
    }

    /// Creates the color and depth textures of the scene of a frame of `width`
    /// by `height`, which both have to follow the size of the window.
    fn create_textures(
        device: &Device,
        width: u32,
        height: u32,
    ) -> (Texture, Texture, Texture, Texture) {
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let color = device.create_texture(&TextureDescriptor {
            label: Label::from("scene_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: Self::SCENE_COLOR_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth = device.create_texture(&TextureDescriptor {
            label: Label::from("depth_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: Self::SCENE_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let wboit_accum = device.create_texture(&TextureDescriptor {
            label: Label::from("wboit_accum_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let wboit_reveal = device.create_texture(&TextureDescriptor {
            label: Label::from("wboit_reveal_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        (color, depth, wboit_accum, wboit_reveal)
    }
}

/// The WBOIT accumulation and revealage textures.
pub struct WboitSource;

impl BindSignature for WboitSource {
    const NAME: &'static str = "wboit";
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
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
    ];
    type Content<'a> = (&'a TextureView, &'a TextureView, &'a Sampler);
}

/// The frame that is currently being recorded, shared with the systems that
/// draw into it and empty outside of rendering.
#[derive(Default)]
pub struct FrameSlot(Option<Frame>);

impl Resource for FrameSlot {}

impl FrameSlot {
    /// Puts the frame of this tick into the slot.
    #[inline]
    pub fn put(&mut self, frame: Frame) {
        self.0 = Some(frame);
    }

    /// Takes the frame out of the slot once every system has drawn into it.
    #[inline]
    pub fn take(&mut self) -> Option<Frame> {
        self.0.take()
    }

    /// The frame that is being recorded, if there is one.
    #[inline]
    pub fn as_mut(&mut self) -> Option<&mut Frame> {
        self.0.as_mut()
    }
}

impl Frame {
    /// Records a render pass of shape `P` begun with `value`.
    ///
    /// The shape is the one the batches drawn in the pass were built for, so
    /// only those batches can be begun in it.
    pub fn render<'a, C: Color, D: Depth>(
        &mut self,
        name: &str,
        color_targets: C::Targets<'a>,
        depth_target: D::Target<'a>,
        f: impl FnOnce(RenderPass),
    ) {
        let pass = self.0.begin_render_pass(&RenderPassDescriptor {
            label: Label::from(format!("{name}_render_pass").as_str()),
            color_attachments: &C::attachments(color_targets),
            depth_stencil_attachment: D::attachment(depth_target),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        f(pass);
    }

    /// Records a compute pass.
    pub fn compute<F: FnOnce(ComputePass)>(&mut self, name: &str, f: F) {
        let pass = self.0.begin_compute_pass(&ComputePassDescriptor {
            label: Label::from(format!("{}_compute_pass", name).as_str()),
            timestamp_writes: None,
        });
        f(pass);
    }
}

pub trait Color {
    type Formats: Clone + Copy;
    type Targets<'a>;

    fn formats(value: Self::Formats) -> Vec<TextureFormat>;

    fn attachments(value: Self::Targets<'_>) -> Vec<Option<RenderPassColorAttachment<'_>>>;
}

pub trait Depth {
    type Format: Clone + Copy;
    type Target<'a>;

    /// Whether the depth pipeline needs a fragment stage despite having no
    /// color attachment.
    const FRAGMENT: bool = false;

    fn format(value: Self::Format) -> Option<TextureFormat>;

    fn attachment(value: Self::Target<'_>) -> Option<RenderPassDepthStencilAttachment<'_>>;
}

pub struct ColorDraw<const N: usize>;
pub struct DepthDraw;

impl Color for () {
    type Formats = ();
    type Targets<'a> = ();

    fn formats(_: Self::Formats) -> Vec<TextureFormat> {
        Vec::new()
    }

    fn attachments(_: Self::Targets<'_>) -> Vec<Option<RenderPassColorAttachment<'_>>> {
        Vec::new()
    }
}

macro_rules! impl_color_for_n {
    ($N:literal) => {
        impl Color for ColorDraw<$N> {
            type Formats = [TextureFormat; $N];
            type Targets<'a> = [ColorTarget<'a>; $N];

            fn formats(value: Self::Formats) -> Vec<TextureFormat> {
                value.to_vec()
            }

            fn attachments(value: Self::Targets<'_>) -> Vec<Option<RenderPassColorAttachment<'_>>> {
                value.into_iter().map(|t| Some(t.attachment())).collect()
            }
        }
    };
}

macro_rules! impl_color_for {
    ($($N:literal),* $(,)?) => {
        $( impl_color_for_n!($N); )*
    };
}

impl_color_for!(1, 2);

impl Depth for () {
    type Format = ();
    type Target<'a> = ();

    fn format(_: Self::Format) -> Option<TextureFormat> {
        None
    }

    fn attachment(_: Self::Target<'_>) -> Option<RenderPassDepthStencilAttachment<'_>> {
        None
    }
}

impl Depth for DepthDraw {
    type Format = TextureFormat;
    type Target<'a> = DepthTarget<'a>;
    const FRAGMENT: bool = true;

    fn format(value: Self::Format) -> Option<TextureFormat> {
        Some(value)
    }

    fn attachment(value: Self::Target<'_>) -> Option<RenderPassDepthStencilAttachment<'_>> {
        Some(value.attachment())
    }
}

/// The color attachment of a pass: the view it draws into, and how that view is
/// loaded.
pub struct ColorTarget<'a> {
    pub view: &'a TextureView,
    pub load: LoadOp<wgpu::Color>,
}

/// The depth attachment of a pass: the view it tests and writes depth through,
/// and how that view is loaded.
pub struct DepthTarget<'a> {
    pub view: &'a TextureView,
    pub load: LoadOp<f32>,
}

impl<'a> ColorTarget<'a> {
    /// The color attachment `wgpu` begins the pass with.
    fn attachment(self) -> RenderPassColorAttachment<'a> {
        RenderPassColorAttachment {
            view: self.view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: self.load,
                store: StoreOp::Store,
            },
        }
    }
}

impl<'a> DepthTarget<'a> {
    /// The depth attachment `wgpu` begins the pass with.
    fn attachment(self) -> RenderPassDepthStencilAttachment<'a> {
        RenderPassDepthStencilAttachment {
            view: self.view,
            depth_ops: Some(Operations {
                load: self.load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }
    }
}
