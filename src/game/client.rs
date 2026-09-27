use crate::actor::*;
use crate::ecs::*;
use crate::game::*;
use crate::render::*;
use crate::ui::*;
use crate::util::coord::{Direction, ICoord3};
use crate::util::OnceInit;
use crate::world::*;
use crossbeam_channel::unbounded;
use glam::{Vec3, Vec3Swizzles};
use log::error;
use noise_functions::{CellDistanceSq, Noise, Perlin};
use rayon::ThreadPoolBuilder;
use smallvec::smallvec;
use std::collections::HashMap;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

/// The window of the game, which only exists once the event loop has resumed.
pub static WINDOW: OnceInit<Window> = OnceInit::new();

/// The game client: it reacts to the window events and updates the worlds of the
/// game and of the user interface.
pub struct GameClient {
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
                self.resources.get_mut::<Canvas>().resize(size);
                self.resources
                    .get_mut::<Viewports>()
                    .transform(self.resources.get::<Canvas>());
            }

            WindowEvent::RedrawRequested => {
                self.frame();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.resources.get_mut::<InputState>().push_key_event(event);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.resources
                    .get_mut::<InputState>()
                    .push_cursor_pos(position);
            }

            WindowEvent::MouseInput { button, state, .. } => {
                self.resources
                    .get_mut::<InputState>()
                    .push_button_event(button, state);
            }

            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.resources
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
                self.resources.get_mut::<PartialTick>().0 = partial_tick;
            }

            WINDOW.request_redraw();
        }
    }

    /// Saves the world before the window closes.
    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.resources.get_mut::<World>().save();
    }
}

impl Default for GameClient {
    fn default() -> Self {
        Self::new()
    }
}

impl GameClient {
    /// Registers the components, the systems and the resources of the game.
    ///
    /// The systems are registered by stage: systems of the same stage run in
    /// parallel unless their accesses overlap, and a lower stage finishes before
    /// the next one starts.
    pub fn new() -> Self {
        let (gen_tx, gen_rx) = unbounded();
        let (meshing_tx, meshing_rx) = unbounded();

        let mut entities = EntityManager::new();
        let mut tick_systems = SystemManager::new();
        let mut frame_systems = SystemManager::new();

        let mut ui_entities = EntityManager::new();
        let mut ui_systems = SystemManager::new();

        let mut resources = ResourceManager::new();

        entities.components.register::<Position>();
        entities.components.register::<PrevPos>();
        entities.components.register::<Rotation>();
        entities.components.register::<Velocity>();
        entities.components.register::<Speed>();
        entities.components.register::<PlayerControlled>();
        entities.components.register::<Bound>();
        entities.components.register::<Contact>();
        entities.components.register::<Flight>();
        entities.components.register::<Selection>();
        ui_entities.components.register::<UiRect>();
        ui_entities.components.register::<UiSprite>();
        ui_entities.components.register::<UiText>();
        ui_entities.components.register::<ScreenTag>();
        ui_entities.components.register::<OnClick>();

        tick_systems.register(0, PlayerController);
        tick_systems.register(1, Stalker);
        tick_systems.register(1, Gravitator);
        tick_systems.register(2, Translator);
        tick_systems.register(3, Collider);
        tick_systems.register(4, Friction);
        tick_systems.register(5, Selector);
        tick_systems.register(6, WorldUpdater);
        tick_systems.register(7, ChunkMeshing);
        frame_systems.register(0, CursorApplier::new());
        frame_systems.register(1, PlayerRotator);
        frame_systems.register(2, Interactor);
        frame_systems.register(2, CameraTransformer);
        ui_systems.register(0, CursorTracker);
        ui_systems.register(1, ScreenController);
        ui_systems.register(1, ScreenCollector);
        ui_systems.register(2, UiPointer::new());

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

        resources.register(InputState::new());
        resources.register(Paused(false));
        resources.register(ActiveScreens {
            screens: HashMap::new(),
            order: Vec::new(),
        });
        resources.register(WorldThreads(near_threads, far_threads));
        resources.register(world);
        resources.register(generator);
        resources.register(ChunkMesher::new(meshing_rx));
        resources.register::<Option<Frame>>(None);
        resources.register(PartialTick(0.0));
        resources.register(Gravity(0.125));

        Self {
            frame_timer: Instant::now(),
            tick_timer: Instant::now(),

            entities,
            resources,
            tick_systems,
            frame_systems,

            ui_entities,
            ui_systems,
        }
    }

    /// Returns whether an open screen pauses the game.
    fn paused(&self) -> bool {
        self.resources
            .try_get::<Paused>()
            .is_ok_and(|paused| paused.0)
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

        let frame = self.resources.get::<Canvas>().begin();
        self.resources.get_mut::<Option<Frame>>().replace(frame);

        match self.frame_systems.update(&self.entities, &self.resources) {
            Ok(commands) => self.entities.submit(commands),

            Err(e) => error!("failed to render: {}", e),
        }

        match self.ui_systems.update(&self.ui_entities, &self.resources) {
            Ok(commands) => self.ui_entities.submit(commands),

            Err(e) => error!("failed to update ui: {}", e),
        }

        self.resources.get_mut::<InputState>().clear();

        if let Some(frame) = self.resources.get_mut::<Option<Frame>>().take() {
            self.resources.get::<Canvas>().end(frame);
        } else {
            error!("failed to render frame");
        };
    }

    /// Creates the resources that need a window, such as the canvas and the
    /// textures, and spawns the player.
    pub fn init(&mut self) {
        let canvas = pollster::block_on(Canvas::new(&WINDOW));
        EMPTY_BUFFER_VEC.init(create_empty_buffer_vec(&canvas));
        QUAD_INDEX_BUFFER_VEC.init(create_quad_index_buffer_vec(&canvas));

        let block_textures =
            BlockTextures(BLOCK_TEXTURES.create_texture_sampler(&canvas, "block", true));
        let camera = Camera::new(&canvas);
        let mut viewports = Viewports::new(&canvas, 256.0, 256.0);
        viewports.transform(&canvas);
        let player = ACTOR_TYPES.get(1).clone();

        self.frame_systems.register(3, WorldRenderer::new(&canvas));
        self.frame_systems
            .register(3, SelectionRenderer::new(&canvas));
        self.ui_systems.register(3, UiRenderer::new(&canvas));
        self.resources.register(canvas);
        self.resources.register(block_textures);
        self.resources.register(camera);
        self.resources.register(viewports);
        self.entities.spawn(player);

        // The screens describe their elements as commands, which the UI entity
        // manager applies on the next frame.
        let mut commands = Vec::new();
        {
            let screens = self.resources.get_mut::<ActiveScreens>();
            commands.extend(screens.open(HUD));
            commands.extend(screens.open(HOTBAR));
        }
        self.ui_entities.submit(commands);

        self.tick_systems.init();
        self.ui_systems.init();
        self.frame_systems.init();
    }
}
