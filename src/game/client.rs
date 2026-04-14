use crate::actor::registries::actor_types;
use crate::actor::*;
use crate::ecs::*;
use crate::game::constants;
use crate::game::input::{InputFlusher, InputState, MouseMotionFlusher};
use crate::render::*;
use crate::util::collection::Registry;
use crate::world::*;
use crate::{render, world};
use crossbeam_channel::unbounded;
use rayon::ThreadPoolBuilder;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use wgpu::{BufferBindingType, ShaderStages, TextureViewDimension};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

pub struct GameClient {
    window: Option<Arc<Window>>,
    frame_timer: Instant,
    tick_timer: Instant,

    simulation: Simulation,
    render_systems: SystemManager,
}

impl ApplicationHandler for GameClient {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.init(event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                let canvas = render::registries::canvas();
                canvas.write().unwrap().resize(size);
            }

            WindowEvent::RedrawRequested => {
                let components = &mut self.simulation.components;
                let resources = &mut self.simulation.resources;
                self.render_systems.update(components, resources);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.simulation
                    .resources
                    .get_mut::<InputState>()
                    .cast_mut::<InputState>()
                    .push_key_event(event);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.simulation
                    .resources
                    .get_mut::<InputState>()
                    .cast_mut::<InputState>()
                    .push_cursor_pos(position);
            }

            WindowEvent::MouseInput { button, state, .. } => {
                self.simulation
                    .resources
                    .get_mut::<InputState>()
                    .cast_mut::<InputState>()
                    .push_button_event(button, state);
            }

            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.simulation
                    .resources
                    .get_mut::<InputState>()
                    .cast_mut::<InputState>()
                    .push_mouse_motion(delta);
            }

            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if self.tick_timer.elapsed() >= constants::tick_duration() {
            self.tick_timer = Instant::now();

            self.simulation.update();
        }

        if self.frame_timer.elapsed() >= constants::frame_duration() {
            self.frame_timer = Instant::now();

            let partial_tick =
                self.tick_timer.elapsed().as_secs_f32() / constants::tick_duration().as_secs_f32();
            self.simulation
                .resources
                .get_mut::<PartialTick>()
                .cast_mut::<PartialTick>()
                .0 = partial_tick;

            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

impl GameClient {
    pub fn new() -> Self {
        let (meshing_tx, meshing_rx) = unbounded();
        let (mesh_tx, mesh_rx) = unbounded();

        let mut simulation = Simulation::new();

        simulation.components.register::<Position>();
        simulation.components.register::<Rotation>();
        simulation.components.register::<Velocity>();
        simulation.components.register::<Speed>();
        simulation.components.register::<PlayerControlled>();

        simulation.systems.register(0, PlayerController);
        simulation.systems.register(1, Translator);
        simulation.systems.register(2, Friction);
        simulation
            .systems
            .register(3, WorldUpdater(PhantomData::<TestGen>));
        simulation.systems.register(4, ChunkMeshing);
        simulation.systems.register(5, InputFlusher);

        simulation.resources.register(InputState::new());
        let near_threads = ThreadPoolBuilder::new()
            .stack_size(4 * 1024 * 1024)
            .thread_name(|i| format!("world_near_{}", i))
            .build()
            .expect("Failed to build Thread Pool for near Regions");
        let far_threads = ThreadPoolBuilder::new()
            .stack_size(4 * 1024 * 1024)
            .thread_name(|i| format!("world_far_{}", i))
            .build()
            .expect("Failed to build Thread Pool for far Regions");
        simulation
            .resources
            .register(WorldThreads(near_threads, far_threads));
        simulation.resources.register(World::new(
            vec![2, 4, 6, 8, 10, 12, 14, 16],
            TestGen::new(0),
            meshing_tx,
        ));
        simulation
            .resources
            .register(ChunkMesher::new(meshing_rx, mesh_tx));
        simulation
            .resources
            .register(RenderedWorld::new(16, mesh_rx));
        simulation.resources.register(PartialTick(0.0));

        let mut render_systems = SystemManager::new();

        render_systems.register(0, PlayerRotator);
        render_systems.register(1, CameraTransformer);
        render_systems.register(2, RenderStarter);
        render_systems.register(4, RenderFinisher);
        render_systems.register(5, MouseMotionFlusher);

        Self {
            window: None,
            frame_timer: Instant::now(),
            tick_timer: Instant::now(),

            simulation,
            render_systems,
        }
    }

    pub fn init(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("To be Titled"))
                .expect("Failed to create window"),
        );

        let mut canvas = pollster::block_on(Canvas::new(&window));
        self.window = Some(window);

        let mut bindings = Registry::new();

        let (block_texture, block_sampler) =
            world::registries::block_textures().create_texture_sampler(&mut canvas, "block");

        let camera = Camera::new(&canvas);

        bindings.register(
            "block_texture_sampler",
            Binding::new(
                &canvas,
                BindingDescriptor {
                    index: 0,
                    name: "block_texture_sampler",
                    items: vec![
                        (
                            ShaderStages::FRAGMENT,
                            BindGroupEntryConfig::TextureView(
                                &TextureViewConfig {
                                    name: "block",
                                    texture: &block_texture,
                                    dimension: TextureViewDimension::D2Array,
                                    mip_level: None,
                                }
                                .create(),
                                TextureViewDimension::D2Array,
                                None,
                            ),
                        ),
                        (
                            ShaderStages::FRAGMENT,
                            BindGroupEntryConfig::Sampler(&block_sampler),
                        ),
                    ],
                },
            ),
        );

        bindings.register(
            "camera",
            Binding::new(
                &canvas,
                BindingDescriptor {
                    index: 1,
                    name: "camera",
                    items: vec![(
                        ShaderStages::VERTEX,
                        BindGroupEntryConfig::Buffer(&camera.buffer, BufferBindingType::Uniform),
                    )],
                },
            ),
        );

        render::registries::set_canvas(canvas);
        render::registries::set_bindings(bindings);

        let world_renderer = WorldRenderer::new();

        self.render_systems.register(3, world_renderer);
        self.simulation.resources.register(camera);
        self.simulation
            .spawn(actor_types().by_name("spectator").create());
    }
}
