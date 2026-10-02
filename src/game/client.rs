use crate::ecs::*;
use crate::game::*;
use crate::render::*;
use crate::util::coord::{Direction, ICoord3};
use crate::util::OnceInit;
use crate::world::*;
use crossbeam_channel::unbounded;
use glam::{Vec3, Vec3Swizzles};
use log::error;
use noise_functions::{CellDistanceSq, Noise, Perlin};
use rayon::ThreadPoolBuilder;
use smallvec::smallvec;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

/// The window of the game, which only exists once the event loop has resumed.
pub(crate) static WINDOW: OnceInit<Window> = OnceInit::new();

/// The game client: it reacts to the window events and updates the worlds of the
/// game and of the user interface.
pub(crate) struct GameClient {
    frame_timer: Instant,
    tick_timer: Instant,

    entities: EntityManager,
    resources: ResourceManager,
    tick_systems: SystemManager,
    frame_systems: SystemManager,

    ui_entities: EntityManager,
    ui_systems: SystemManager,
}

impl ApplicationHandler for GameClient {
    /// Creates the window and initialises the client, once.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !WINDOW.ready() {
            WINDOW.init(
                event_loop
                    .create_window(Window::default_attributes().with_title("To be Titled"))
                    .expect("failed to create window"),
            );

            self.init();
        }
    }

    /// Forwards the window events to the canvas and to the input state.
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                self.resources
                    .get::<Canvas>()
                    .unwrap()
                    .get_mut::<Canvas>()
                    .resize(size);
                self.resources
                    .get::<Viewports>()
                    .unwrap()
                    .get_mut::<Viewports>()
                    .transform(self.resources.get::<Canvas>().unwrap().get::<Canvas>());
            }

            WindowEvent::RedrawRequested => {
                self.frame();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.resources
                    .get::<InputState>()
                    .unwrap()
                    .get_mut::<InputState>()
                    .push_key_event(event);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.resources
                    .get::<InputState>()
                    .unwrap()
                    .get_mut::<InputState>()
                    .push_cursor_pos(position);
            }

            WindowEvent::MouseInput { button, state, .. } => {
                self.resources
                    .get::<InputState>()
                    .unwrap()
                    .get_mut::<InputState>()
                    .push_button_event(button, state);
            }
            
            WindowEvent::MouseWheel { delta, .. } => {
                self.resources
                    .get::<InputState>()
                    .unwrap()
                    .get_mut::<InputState>()
                    .push_mouse_scroll(delta);
            }

            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.resources
                    .get::<InputState>()
                    .unwrap()
                    .get_mut::<InputState>()
                    .push_mouse_motion(delta);
            }

            _ => (),
        }
    }

    /// Ticks the simulation, starts a frame and asks the window to redraw.
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if !self.paused() && self.tick_timer.elapsed() >= *TICK_DURATION {
            self.tick_timer = Instant::now();

            self.tick();
        }

        if self.frame_timer.elapsed() >= *FRAME_DURATION {
            self.frame_timer = Instant::now();

            if !self.paused() {
                let partial_tick =
                    self.tick_timer.elapsed().as_secs_f32() / TICK_DURATION.as_secs_f32();
                self.resources
                    .get::<PartialTick>()
                    .unwrap()
                    .get_mut::<PartialTick>()
                    .0 = partial_tick;
            }

            WINDOW.request_redraw();
        }
    }

    /// Saves the world before the window closes.
    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.resources
            .get::<World>()
            .unwrap()
            .get_mut::<World>()
            .save();
    }
}

impl Default for GameClient {
    fn default() -> Self {
        Self::new()
    }
}

impl GameClient {
    /// Creates the client; everything it needs is registered by [`Self::setup`].
    pub(crate) fn new() -> Self {
        Self {
            frame_timer: Instant::now(),
            tick_timer: Instant::now(),

            entities: EntityManager::new(),
            resources: ResourceManager::new(),
            tick_systems: SystemManager::new(),
            frame_systems: SystemManager::new(),

            ui_entities: EntityManager::new(),
            ui_systems: SystemManager::new(),
        }
    }

    /// Registers the components, the systems, the resources and the data of
    /// the game, which the window is not needed for.
    ///
    /// The systems are registered by stage: systems of the same stage run in
    /// parallel unless their accesses overlap, and a lower stage finishes
    /// before the next one starts.
    pub(crate) fn setup(&mut self) {
        crate::actor::register(
            &mut self.entities,
            &mut self.tick_systems,
            &mut self.frame_systems,
        );
        crate::game::register(&mut self.frame_systems, &mut self.resources);
        crate::render::register(&mut self.frame_systems, &mut self.resources);
        crate::ui::register(
            &mut self.ui_entities,
            &mut self.ui_systems,
            &mut self.resources,
        );
        crate::world::register(&mut self.tick_systems, &mut self.resources);

        self.register_world();
    }

    /// Creates the channels, the threads and the resources of the voxel world.
    ///
    /// Temporary: the world and its generator are still described here, until
    /// the mod scripts drive the world data.
    fn register_world(&mut self) {
        let (gen_tx, gen_rx) = unbounded();
        let (meshing_tx, meshing_rx) = unbounded();

        let near_threads = ThreadPoolBuilder::new()
            .stack_size(4 * 1024 * 1024)
            .thread_name(|i| format!("world_near_{}", i))
            .build()
            .expect("failed to build thread pool for near regions");
        let far_threads = ThreadPoolBuilder::new()
            .stack_size(4 * 1024 * 1024)
            .thread_name(|i| format!("world_far_{}", i))
            .build()
            .expect("failed to build thread pool for far regions");
        let world = World::new(
            RegionPos::ZERO,
            // The radius of each level of detail, from the finest level to the
            // coarsest one; further levels such as 13 and 21 are possible, but
            // every one of them costs another ring of regions.
            smallvec![3, 7],
            gen_tx,
            meshing_tx,
        );
        let generator = Generator::new(
            RegionPos::ZERO,
            8,
            Field {
                climate: |_| -> Vec3 { Vec3::ZERO },
                // A continental noise blob that is flatter at the top of a hill
                // than at the bottom of a valley, plus a finer noise for the
                // details of the surface.
                density: |pos| -> f32 {
                    let pos = pos.as_vec3();

                    let h = Perlin::default().frequency(0.015625).sample2(pos.xz()) * 32.0;
                    let a = 0.125;
                    let b = 0.0625;
                    let d1 = (if h > 0.0 {
                        ((h * a).exp() - 1.0) / a
                    } else {
                        (1.0 - (h * -b).exp()) / b
                    } - pos.y)
                        * 0.03125;

                    let d2 = Perlin::default().frequency(0.0625).sample3(pos) * 0.75;

                    (d1.tanh() + d2).tanh()
                },
                erosion: |pos| -> f32 {
                    let mut pos = pos.as_vec3();
                    pos.y *= 2.0;

                    let v1 = CellDistanceSq::default()
                        .jitter(0.75)
                        .frequency(0.03125)
                        .sample3(pos);

                    let v2 = CellDistanceSq::default()
                        .jitter(1.25)
                        .frequency(0.03125)
                        .sample3(pos);

                    let v3 = -(pos.y + 128.0) * 0.00390625;

                    (v1 + v2 + v3).tanh()
                },
            },
            // Turns a sample into a block: solid where the field is dense and not
            // eroded, dirt just below the surface, and air everywhere else.
            |sample| -> Meta {
                if sample.density > 0.0 && sample.erosion < 0.0 {
                    if sample.density < 0.125
                        && sample.gradient.y < 0.0
                        && sample.gradient.xz().length_squared() < 0.00390625
                    {
                        2
                    } else {
                        1
                    }
                } else {
                    0
                }
            },
            vec![Structure::tree()],
            // Grass on top of every dirt block that is exposed to the air.
            |meta, chunk, pos| -> Meta {
                if meta == 2 && *chunk.get(pos.step(Direction::Up)) == 0 {
                    3
                } else {
                    meta
                }
            },
            gen_rx,
        );

        self.resources
            .register("world_threads", WorldThreads(near_threads, far_threads));
        self.resources.register("world", world);
        self.resources.register("generator", generator);
        self.resources
            .register("chunk_mesher", ChunkMesher::new(meshing_rx));
    }

    /// Returns whether an open screen pauses the game.
    fn paused(&self) -> bool {
        self.resources
            .get::<Paused>()
            .is_ok_and(|paused| paused.get::<Paused>().0)
    }

    /// Applies the queued commands and runs one step of the simulation.
    fn tick(&mut self) {
        if let Err(e) = self.entities.flush() {
            error!("failed to flush commands: {}", e);
        }

        match self.tick_systems.update(&self.entities, &self.resources) {
            Ok(commands) => self.entities.submit(commands),

            Err(e) => error!("failed to tick: {}", e),
        }
    }

    /// Records one frame: the world, the selection, the user interface and the
    /// presentation of the result.
    fn frame(&mut self) {
        if let Err(e) = self.ui_entities.flush() {
            error!("failed to flush ui commands: {}", e);
        }

        let frame = self
            .resources
            .get::<Canvas>()
            .unwrap()
            .get::<Canvas>()
            .begin();
        self.resources
            .get::<Option<Frame>>()
            .unwrap()
            .get_mut::<Option<Frame>>()
            .replace(frame);

        match self.frame_systems.update(&self.entities, &self.resources) {
            Ok(commands) => self.entities.submit(commands),

            Err(e) => error!("failed to render: {}", e),
        }

        match self.ui_systems.update(&self.ui_entities, &self.resources) {
            Ok(commands) => self.ui_entities.submit(commands),

            Err(e) => error!("failed to update ui: {}", e),
        }

        self.resources
            .get::<InputState>()
            .unwrap()
            .get_mut::<InputState>()
            .clear();

        if let Some(frame) = self
            .resources
            .get::<Option<Frame>>()
            .unwrap()
            .get_mut::<Option<Frame>>()
            .take()
        {
            self.resources
                .get::<Canvas>()
                .unwrap()
                .get::<Canvas>()
                .end(frame);
        } else {
            error!("failed to render frame");
        };
    }

    /// Creates the resources that need a window, registers the renderers and
    /// starts the first screens.
    pub(crate) fn init(&mut self) {
        let canvas = pollster::block_on(Canvas::new(&WINDOW));

        crate::render::init(&mut self.resources, &canvas);
        crate::actor::register_canvas(&mut self.frame_systems, &canvas);
        crate::world::register_canvas(&mut self.resources, &mut self.frame_systems, &canvas);
        crate::ui::register_canvas(&mut self.ui_systems, &mut self.resources, &canvas);

        self.resources.register("canvas", canvas);
        crate::actor::spawn_player(&mut self.entities);
        crate::ui::open_initial(&mut self.ui_entities, &self.resources);

        self.tick_systems.init();
        self.ui_systems.init();
        self.frame_systems.init();
    }
}
