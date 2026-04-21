use crate::actor::*;
use crate::ecs::*;
use crate::game::{Escaper, InputFlusher, InputState, FRAME_DURATION, TICK_DURATION};
use crate::render::*;
use crate::util::OnceInit;
use crate::world::*;
use crossbeam_channel::unbounded;
use rayon::ThreadPoolBuilder;
use std::marker::PhantomData;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

pub static WINDOW: OnceInit<Window> = OnceInit::new();

pub struct GameClient {
    frame_timer: Instant,
    tick_timer: Instant,

    simulation: Simulation,
    render_systems: SystemManager,
}

impl ApplicationHandler for GameClient {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !WINDOW.ready() {
            WINDOW.init(
                event_loop
                    .create_window(Window::default_attributes().with_title("To be Titled"))
                    .expect("Failed to create window"),
            );
            
            self.init();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                self.simulation.resources.get_mut::<Canvas>().resize(size);
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
                    .push_key_event(event);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.simulation
                    .resources
                    .get_mut::<InputState>()
                    .push_cursor_pos(position);
            }

            WindowEvent::MouseInput { button, state, .. } => {
                self.simulation
                    .resources
                    .get_mut::<InputState>()
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
                    .push_mouse_motion(delta);
            }

            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if self.tick_timer.elapsed() >= *TICK_DURATION {
            self.tick_timer = Instant::now();
            
            self.simulation.update();
        }

        if self.frame_timer.elapsed() >= *FRAME_DURATION {
            self.frame_timer = Instant::now();

            let partial_tick =
                self.tick_timer.elapsed().as_secs_f32() / TICK_DURATION.as_secs_f32();
            self.simulation.resources.get_mut::<PartialTick>().0 = partial_tick;
            
            WINDOW.request_redraw();
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
        simulation.components.register::<Bound>();
        simulation.components.register::<Option<Selection>>();

        simulation.systems.register(0, PlayerController);
        simulation.systems.register(1, Translator);
        simulation.systems.register(2, Friction);
        simulation
            .systems
            .register(3, Selector::<TestGen>(PhantomData));
        simulation
            .systems
            .register(4, WorldUpdater(PhantomData::<TestGen>));
        simulation.systems.register(5, ChunkMeshing);

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
        simulation.resources.register::<Option<Frame>>(None);
        simulation.resources.register(PartialTick(0.0));

        let mut render_systems = SystemManager::new();
        
        render_systems.register(0, Escaper);
        render_systems.register(1, PlayerRotator);
        render_systems.register(2, Interactor(PhantomData::<TestGen>));
        render_systems.register(2, CameraTransformer);
        render_systems.register(3, InputFlusher);
        render_systems.register(3, RenderStarter);
        render_systems.register(5, RenderFinisher);

        Self {
            frame_timer: Instant::now(),
            tick_timer: Instant::now(),

            simulation,
            render_systems,
        }
    }
    
    pub fn init(&mut self) {
        let canvas = pollster::block_on(Canvas::new(&WINDOW));

        let block_textures =
            BlockTextures(create_block_textures().create_texture_sampler(&canvas, "block"));
        let camera = Camera::new(&canvas);
        let world_renderer = WorldRenderer::new(&canvas);
        let selection_renderer = SelectionRenderer::new(&canvas);

        self.render_systems.register(3, world_renderer);
        self.render_systems.register(3, selection_renderer);
        self.simulation.resources.register(canvas);
        self.simulation.resources.register(block_textures);
        self.simulation.resources.register(camera);
        self.simulation.spawn(ACTOR_TYPES.get(1).create());
        
        self.simulation.systems.init();
        self.render_systems.init();
    }
}
