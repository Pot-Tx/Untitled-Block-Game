use crate::ecs::*;
use crate::render::{FromConfig, TextureViewConfig};
use glam::{f32, u32};
use wgpu::CurrentSurfaceTexture::Success;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// The attachments and command encoder of the frame that is being recorded.
pub struct Frame {
    surface_texture: SurfaceTexture,
    pub color: TextureView,
    pub depth: TextureView,
    pub encoder: CommandEncoder,
}

/// The name and the attachment load operations of a render pass.
pub struct RenderDescriptor<'a> {
    pub name: &'a str,
    pub color_load: LoadOp<Color>,
    pub depth_load: LoadOp<f32>,
}

/// The name of a compute pass.
pub struct ComputeDescriptor<'a> {
    pub name: &'a str,
}

/// The window surface, its configuration and the device the game draws with.
pub struct Canvas {
    surface: Surface<'static>,
    pub surface_config: SurfaceConfiguration,
    pub device: Device,
    pub queue: Queue,
    depth: Texture,
}

impl Resource for Canvas {}

impl Canvas {
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

        let depth = device.create_texture(&TextureDescriptor {
            label: Label::from("depth_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        Self {
            surface,
            surface_config,
            device,
            queue,
            depth,
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

        self.depth = self.device.create_texture(&TextureDescriptor {
            label: Label::from("depth_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
    }

    /// Acquires the next surface texture and starts recording a frame.
    pub fn begin(&self) -> Frame {
        if let Success(surface_texture) = self.surface.get_current_texture() {
            let color = TextureView::new(
                &surface_texture.texture,
                &TextureViewConfig {
                    name: "color",
                    dimension: TextureViewDimension::D2,
                    mip_level: None,
                },
            );
            let depth = TextureView::new(
                &self.depth,
                &TextureViewConfig {
                    name: "depth",
                    dimension: TextureViewDimension::D2,
                    mip_level: None,
                },
            );
            let encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Label::from("command_encoder"),
                });

            Frame {
                surface_texture,
                color,
                depth,
                encoder,
            }
        } else {
            panic!("failed to get next surface texture");
        }
    }

    /// Submits the recorded commands and presents the frame.
    pub fn end(&self, frame: Frame) {
        self.queue.submit(Some(frame.encoder.finish()));
        self.queue.present(frame.surface_texture);
    }
}

/// The frame that is currently being recorded, shared with the systems that
/// draw into it and empty outside of rendering.
impl Resource for Option<Frame> {}

impl Frame {
    /// Records a render pass with the attachments of this frame.
    pub fn render<F: FnOnce(RenderPass)>(&mut self, desc: &RenderDescriptor, f: F) {
        let pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Label::from(format!("{}_render_pass", desc.name).as_str()),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.color,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: desc.color_load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(Operations {
                    load: desc.depth_load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        f(pass);
    }

    /// Records a compute pass.
    pub fn compute<F: FnOnce(ComputePass)>(&mut self, desc: &ComputeDescriptor, f: F) {
        let pass = self.encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Label::from(format!("{}_compute_pass", desc.name).as_str()),
            timestamp_writes: None,
        });
        f(pass);
    }
}
