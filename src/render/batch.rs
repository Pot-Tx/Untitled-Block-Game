use crate::render::config::RenderPipelineConfig;
use crate::render::*;
use glam::u32;
use std::marker::PhantomData;
use wgpu::CurrentSurfaceTexture::Success;

pub mod registries {
    use crate::render::*;
    use crate::util::collection::Registry;
    use crate::world;
    use log::error;
    use std::sync::OnceLock;
    use wgpu::{AddressMode, MipmapFilterMode, ShaderStages};
    
    static CANVAS: OnceLock<RwLock<Canvas>> = OnceLock::new();
    static BINDINGS: OnceLock<Registry<Binding>> = OnceLock::new();

    pub fn set_canvas(canvas: Canvas) {
        if CANVAS.set(RwLock::new(canvas)).is_err() {
            error!("Canvas already initialized");
        }
    }

    pub fn set_bindings(bindings: Registry<Binding>) {
        if BINDINGS.set(bindings).is_err() {
            error!("Bindings already initialized");
        }
    }

    #[inline]
    pub fn canvas() -> &'static RwLock<Canvas> {
        CANVAS
            .get()
            .expect("Failed to get Canvas. Make sure to call registries::set_canvas first!")
    }

    #[inline]
    pub fn bindings() -> &'static Registry<Binding> {
        BINDINGS
            .get()
            .expect("Failed to get Bindings. Make sure to call registries::set_bindings first!")
    }
}

pub struct RenderBatch<'a, V: Vertex, I: Inst> {
    render_pipeline: RenderPipeline,
    bindings: Vec<&'a Binding>,
    _v_marker: PhantomData<V>,
    _i_marker: PhantomData<I>,
}

pub struct RenderBatchDescriptor<'a> {
    pub name: &'a str,
    pub bindings: Vec<&'a Binding>,
    pub shader_name: &'a str,
    pub translucent: bool,
    pub topology: PrimitiveTopology,
    pub depth_write: bool,
}

pub struct ComputeBatch<'a> {
    compute_pipeline: ComputePipeline,
    bindings: Vec<&'a Binding>,
}

pub struct ComputeBatchDescriptor<'a> {
    pub name: &'a str,
    pub bindings: Vec<&'a Binding>,
    pub shader_name: &'a str,
}

#[derive(Debug)]
pub struct Binding {
    pub index: u32,
    pub bind_group: BindGroup,
    pub layout: BindGroupLayout,
}

pub struct BindingDescriptor<'a> {
    pub index: u32,
    pub name: &'a str,
    pub items: Vec<(ShaderStages, BindGroupEntryConfig<'a>)>,
}

pub struct Canvas {
    surface: Surface<'static>,
    pub surface_config: SurfaceConfiguration,
    pub device: Device,
    pub queue: Queue,
    depth_texture: Texture,
    pub depth_texture_view: TextureView,
    surface_texture: Option<SurfaceTexture>,
    pub texture_view: Option<TextureView>,
    pub command_encoder: Option<CommandEncoder>,
}

unsafe impl Sync for Canvas {}
unsafe impl Send for Canvas {}

impl Canvas {
    pub async fn new(window: &Arc<Window>) -> Self {
        let wgpu_instance = Instance::default();
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let surface = wgpu_instance.create_surface(window.clone()).unwrap();
        let adapter = wgpu_instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find adapter");
        let surface_config = surface.get_default_config(&adapter, width, height).unwrap();
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Label::from("device"),
                required_features: Features::empty(),
                required_limits: Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .expect("Failed to create device");
        surface.configure(&device, &surface_config);
        let depth_texture = device.create_texture(&TextureDescriptor {
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
        let depth_texture_view = depth_texture.create_view(&TextureViewDescriptor::default());

        Self {
            surface,
            surface_config,
            device,
            queue,
            depth_texture,
            depth_texture_view,
            surface_texture: None,
            texture_view: None,
            command_encoder: None,
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        let width = size.width.max(1);
        let height = size.height.max(1);

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);

        self.depth_texture = self.device.create_texture(&TextureDescriptor {
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
        self.depth_texture_view = self
            .depth_texture
            .create_view(&TextureViewDescriptor::default());
    }

    pub fn begin(&mut self) {
        if let Success(surface_texture) = self.surface.get_current_texture() {
            self.texture_view = Some(
                surface_texture
                    .texture
                    .create_view(&TextureViewDescriptor::default()),
            );
            self.surface_texture = Some(surface_texture);
            self.command_encoder = Some(self.device.create_command_encoder(
                &CommandEncoderDescriptor {
                    label: Label::from("command_encoder"),
                },
            ));
        } else {
            panic!("Failed to get next surface texture");
        }
    }

    pub fn end(&mut self) {
        if let Some(surface_texture) = self.surface_texture.take()
            && let Some(_) = self.texture_view.take()
            && let Some(command_encoder) = self.command_encoder.take()
        {
            self.queue.submit(Some(command_encoder.finish()));
            surface_texture.present();
        } else {
            panic!("Failed to render on Canvas. Make sure to call Canvas::begin first!");
        }
    }
}

impl Resource for Registry<Binding> {}

impl Binding {
    pub fn new(canvas: &Canvas, desc: BindingDescriptor) -> Self {
        let index = desc.index;
        let config = BindGroupConfig {
            name: desc.name,
            items: desc.items,
        };
        let bind_group = config.create(canvas);
        let layout = config.layout(canvas);

        Binding {
            index,
            bind_group,
            layout,
        }
    }
}

impl<'a, V: Vertex, I: Inst> RenderBatch<'a, V, I> {
    pub fn new(canvas: &Canvas, desc: RenderBatchDescriptor<'a>) -> Self {
        let layouts = desc
            .bindings
            .iter()
            .enumerate()
            .map(|(idx, &binding)| {
                assert_eq!(idx as u32, binding.index);
                &binding.layout
            })
            .collect();

        let config = RenderPipelineConfig {
            name: desc.name,
            bind_group_layouts: layouts,
            shader_name: desc.shader_name,
            translucent: desc.translucent,
            vertex_buffer_layouts: &[V::layout(), I::layout::<V>()],
            topology: desc.topology,
            depth_write: desc.depth_write,
        };
        let render_pipeline = config.create(canvas);

        Self {
            render_pipeline,
            bindings: desc.bindings,
            _v_marker: PhantomData,
            _i_marker: PhantomData,
        }
    }

    pub fn begin(&mut self, render_pass: &mut RenderPass) {
        render_pass.set_pipeline(&self.render_pipeline);

        for &binding in self.bindings.iter() {
            render_pass.set_bind_group(binding.index, &binding.bind_group, &[]);
        }
    }

    pub fn draw(&mut self, render_pass: &mut RenderPass, item: &impl Render<V, I>) {
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
        let layouts = desc
            .bindings
            .iter()
            .enumerate()
            .map(|(idx, &binding)| {
                assert_eq!(idx as u32, binding.index);
                &binding.layout
            })
            .collect();

        let config = ComputePipelineConfig {
            name: desc.name,
            bind_group_layouts: layouts,
            shader_name: desc.shader_name,
        };
        let compute_pipeline = config.create(canvas);

        Self {
            compute_pipeline,
            bindings: desc.bindings,
        }
    }

    pub fn begin(&mut self, compute_pass: &mut ComputePass) {
        compute_pass.set_pipeline(&self.compute_pipeline);

        for &binding in self.bindings.iter() {
            compute_pass.set_bind_group(binding.index, &binding.bind_group, &[]);
        }
    }

    pub fn dispatch(&self, compute_pass: &mut ComputePass, x: u32, y: u32, z: u32) {
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
