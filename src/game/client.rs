use crate::actor::EntityStore;
use crate::actor::Session;
use crate::ecs::*;
use crate::game::*;
use crate::generation::*;
use crate::render::*;
use crate::util::OnceInit;
use crate::world::*;
use anyhow::anyhow;
use crossbeam_channel::unbounded;
use glam::Vec3;
use log::{error, info};
use rayon::ThreadPoolBuilder;
use smallvec::smallvec;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
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

    /// The parameters the game was started with, until the world is created.
    startup: Option<Startup>,

    world_entities: EntityManager,
    ui_entities: EntityManager,

    /// Keeps the identity of the world entities and writes them to their save
    /// files.
    store: EntityStore,
    /// The session the world was loaded from, until its entities are spawned.
    session: Option<Session>,

    resources: ResourceManager,

    tick_systems: SystemManager<1>,
    frame_systems: SystemManager<2>,
}

impl ApplicationHandler for GameClient {
    /// Creates the window and initializes the client, once.
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

    /// Ticks the simulation, saves while paused, starts a frame and asks the
    /// window to redraw.
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        let paused = self.paused();

        if !paused && self.tick_timer.elapsed() >= *TICK_DURATION {
            self.tick_timer = Instant::now();

            self.tick();
        }

        if self.frame_timer.elapsed() >= *FRAME_DURATION {
            self.frame_timer = Instant::now();

            if !paused {
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

    /// Saves the world and waits for the pending writes before the window
    /// closes.
    fn exiting(&mut self, _: &ActiveEventLoop) {
        let world = self.resources.get::<World>().unwrap().get_mut::<World>();
        world.flush();

        self.store.flush(&self.world_entities);

        let seed = self
            .resources
            .get::<Generator>()
            .map(|generator| generator.get::<Generator>().seed())
            .unwrap_or(0);

        self.store
            .write_session(&self.resources, &self.world_entities, seed);
    }
}

impl GameClient {
    /// Creates the client, which starts the world as `startup` asks.
    pub(crate) fn new(startup: Startup) -> Self {
        Self {
            frame_timer: Instant::now(),
            tick_timer: Instant::now(),

            startup: Some(startup),

            world_entities: EntityManager::new(),
            ui_entities: EntityManager::new(),

            store: EntityStore::new(),
            session: None,

            resources: ResourceManager::new(),

            tick_systems: SystemManager::new(),
            frame_systems: SystemManager::new(),
        }
    }

    /// Registers the components, the systems and the resources of the game,
    /// which the window is not needed for.
    ///
    /// Within a stage the systems run in parallel unless their accesses
    /// overlap, and a lower stage finishes before the next one starts. The data
    /// of the modules is loaded by [`Self::init`], once the window exists.
    pub(crate) fn setup(&mut self) {
        crate::game::register(&mut self.frame_systems, &mut self.resources);
        // The ticks of the world come before every system that reads them.
        self.tick_systems.register(0, "ticker", super::Ticker);
        crate::render::register(&mut self.frame_systems, &mut self.resources);
        crate::world::register(&mut self.resources);
        crate::actor::register(
            &mut self.world_entities,
            &mut self.tick_systems,
            &mut self.frame_systems,
        );
        crate::ui::register(
            &mut self.ui_entities,
            &mut self.frame_systems,
            &mut self.resources,
        );
    }

    /// Creates the channels, the threads and the resources of the voxel world.
    fn register_world(
        &mut self,
        canvas: &Canvas,
        mods: &ModRegistry,
        center: RegionPos,
        seed: u64,
    ) -> anyhow::Result<()> {
        // The color the sky has overhead and the one it holds around the horizon,
        // which the lights that do not cast a shadow carry. Both follow the sun:
        // they fade to the color of the night as it sets, and the horizon warms
        // up where the light of the sun takes a longer way through the
        // atmosphere.
        let overhead: fn(u64) -> Vec3 = |tick| {
            let day = ((tick % 2400) as f32 / 2400.0 * TAU).sin().clamp(0.0, 1.0) * 4.0;
            let day = day.clamp(0.0, 1.0);
            let day = day * day * (3.0 - 2.0 * day);

            Vec3::new(0.02, 0.03, 0.06).lerp(Vec3::new(0.35, 0.55, 1.0), day)
        };
        let horizon: fn(u64) -> Vec3 = |tick| {
            let height = ((tick % 2400) as f32 / 2400.0 * TAU).sin().clamp(0.0, 1.0);
            let day = (height * 4.0).clamp(0.0, 1.0);
            let day = day * day * (3.0 - 2.0 * day);
            let sunset = 1.0 - height;

            Vec3::new(0.02, 0.03, 0.06).lerp(Vec3::new(0.75, 0.85, 1.0), day)
                + Vec3::new(1.0, 0.55, 0.25) * sunset * sunset * 0.25
        };

        // The sky of the world. The rotation of a light is the `yaw`, `pitch` and
        // `roll` of the camera of the world turned inside out, and its `z` axis is
        // the direction it shines from: the sun and the moon turn about the axis of
        // the day alone, so that they rise in the east and set in the west, and the
        // moon stands opposite the sun. Both cast the shadow of the world, one
        // layer of the map each. A day lasts 2400 ticks, which is twenty minutes.
        //
        // The strength of the sun and of the moon is the height they stand at,
        // which is how much of their light reaches level ground: it rises from
        // zero with a finite slope, so the light of the world changes smoothly
        // through sunrise and sunset without being crushed for the rest of the
        // day, and the sun warms up towards the horizon.
        //
        // The lights that do not cast a shadow hold the color of the sky: one
        // overhead and one towards every horizon, which the shader weighs by how
        // much each surface faces them. Their rotations stand still, and are the
        // ones that turn the `z` axis of a light onto the axis they light along.
        let lighting_settings =
            LightingSettings::load().expect("failed to load the lighting settings");

        let sky = Sky::new(
            vec![
                Sun::new(
                    |tick| Vec3::new(0.0, -((tick % 2400) as f32 / 2400.0 * TAU), -FRAC_PI_2),
                    |tick| {
                        let height = ((tick % 2400) as f32 / 2400.0 * TAU).sin().clamp(0.0, 1.0);
                        let sunset = 1.0 - height;

                        Vec3::new(1.0, 0.98, 0.94).lerp(Vec3::new(1.0, 0.55, 0.25), sunset * sunset)
                            * height
                    },
                    0,
                ),
                Sun::new(
                    |tick| Vec3::new(0.0, PI - ((tick % 2400) as f32 / 2400.0 * TAU), -FRAC_PI_2),
                    |tick| {
                        let height = ((tick % 2400) as f32 / 2400.0 * TAU + PI)
                            .sin()
                            .clamp(0.0, 1.0);

                        Vec3::new(0.25, 0.35, 0.55) * 0.125 * height
                    },
                    0,
                ),
            ],
            vec![
                Sun::new(|_| Vec3::new(0.0, -FRAC_PI_2, -FRAC_PI_2), overhead, 0),
                Sun::new(|_| Vec3::new(0.0, FRAC_PI_2, -FRAC_PI_2), horizon, 0),
                Sun::new(|_| Vec3::new(0.0, 0.0, -FRAC_PI_2), horizon, 0),
                Sun::new(|_| Vec3::new(0.0, PI, -FRAC_PI_2), horizon, 0),
                Sun::new(|_| Vec3::ZERO, horizon, 0),
                Sun::new(|_| Vec3::new(PI, 0.0, 0.0), horizon, 0),
            ],
        );
        let camera = self
            .resources
            .get::<Camera>()
            .expect("the camera is registered before the world")
            .get::<Camera>();
        let lighting = Lighting::new(canvas, camera, &sky, lighting_settings);

        self.resources.register("sky", sky);
        self.resources.register("lighting", lighting);

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
            center,
            // The radius of each level of detail, from the finest level to the
            // coarsest one; further levels such as 13 and 21 are possible, but
            // every one of them costs another ring of regions.
            smallvec![2, 3],
            1,
            gen_tx,
            meshing_tx,
        );
        let mut generator = Generator::new(self.generator_name()?.to_owned());
        generator.prepare(mods, center, 3, seed, gen_rx)?;

        self.resources.register(
            "world_threads",
            WorldThreads {
                near: near_threads,
                far: far_threads,
            },
        );
        self.resources.register("world", world);
        self.resources.register("generator", generator);

        crate::world::register_updater(&mut self.tick_systems, ChunkMesher::new(meshing_rx));

        Ok(())
    }

    /// The generator the world uses: the one the world type or the save set on
    /// the generator resource.
    fn generator_name(&self) -> anyhow::Result<&str> {
        let generator = self
            .resources
            .get::<Generator>()
            .map_err(|e| anyhow!("the generator resource is missing: {e}"))?
            .get::<Generator>();

        match generator.name() {
            "" => Err(anyhow!(
                "no generator is chosen; name one in the world type or the save"
            )),
            name => Ok(name),
        }
    }

    /// Returns whether an open screen pauses the game.
    fn paused(&self) -> bool {
        self.resources
            .get::<Paused>()
            .is_ok_and(|paused| paused.get::<Paused>().0)
    }

    /// Applies the queued commands and runs one step of the simulation.
    fn tick(&mut self) {
        self.world_entities.flush();

        self.tick_systems
            .update([&self.world_entities], &self.resources);

        // The entities follow the regions the world keeps in memory: the ones
        // that entered the scope are read back, the ones that left are written
        // out and dropped.
        let (entered, left) = self
            .resources
            .get::<World>()
            .unwrap()
            .get_mut::<World>()
            .take_entity_scope();

        self.store.load(&entered);
        self.store.evict(&self.world_entities, &left);
        self.store.sync(&self.world_entities);
    }

    /// Records one frame: the world, the selection, the user interface and the
    /// presentation of the result.
    fn frame(&mut self) {
        self.ui_entities.flush();

        let frame = self
            .resources
            .get::<Canvas>()
            .unwrap()
            .get::<Canvas>()
            .begin();
        self.resources
            .get::<FrameSlot>()
            .unwrap()
            .get_mut::<FrameSlot>()
            .put(frame);

        // The frame stage covers the world and the interface; a system queues
        // its commands on the entity manager it addresses through its commander.
        self.frame_systems
            .update([&self.world_entities, &self.ui_entities], &self.resources);

        self.resources
            .get::<InputState>()
            .unwrap()
            .get_mut::<InputState>()
            .clear();

        if let Some(frame) = self
            .resources
            .get::<FrameSlot>()
            .unwrap()
            .get_mut::<FrameSlot>()
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

    /// Creates the resources that need a window, loads the data that names
    /// them, registers the renderers and starts the first screens.
    ///
    /// The modules load their data in the order it depends on each other: the
    /// textures of the blocks come before the blocks, which name them, and the
    /// blocks come before the screens and the actors, which name a block or a
    /// sprite of them.
    pub(crate) fn init(&mut self) {
        let canvas = pollster::block_on(Canvas::new(&WINDOW));
        let Startup { mods, world, seed } =
            self.startup.take().expect("the game is initialized once");

        crate::render::init(&mut self.resources, &canvas, &mods);
        crate::render::register_canvas(&canvas);
        crate::world::register_canvas(&mut self.frame_systems, &canvas, &mods);
        crate::actor::register_canvas(
            &mut self.world_entities,
            &mut self.resources,
            &mut self.frame_systems,
            &canvas,
            &mods,
        );
        crate::ui::register_canvas(
            &mut self.ui_entities,
            &mut self.frame_systems,
            &mut self.resources,
            &canvas,
            &mods,
        );

        // The scripts and the structures are content of theirs, which a generator
        // names; the blocks and the actors they reference are built by now.
        crate::generation::init_structures(&mods);

        // A world begins with the values of its world type. What the session of an
        // existing world holds replaces them below, so a save keeps the content it
        // was started with.
        if let Err(e) = world.apply(&mut self.resources) {
            panic!("failed to apply the world type {}: {e}", world.name);
        }

        // The world is centered on where the player was left, which only the
        // session knows before a region has been loaded.
        self.session = self.store.read_session();
        if let Some(session) = &self.session {
            session.resources.apply(&mut self.resources);
        }
        let seed = self.session.as_ref().map_or(seed, |session| session.seed);
        let generator = self.generator_name().unwrap_or("none");

        match &self.session {
            Some(session) => info!(
                "the save keeps the world it was started with: seed {}, generator {generator}; \
                 --world and --seed only apply to a new world",
                session.seed
            ),
            None => info!(
                "a new world begins: world type {}, generator {generator}, seed {seed}",
                world.name
            ),
        }

        let center = self
            .session
            .as_ref()
            .and_then(|session| session.player_region(&self.world_entities.components))
            .unwrap_or(RegionPos::ZERO);

        self.register_world(&canvas, &mods, center, seed)
            .expect("failed to initialize the world");

        self.resources.register("canvas", canvas);

        match self.session.take() {
            Some(session) => self.store.load_session(&self.world_entities, &session),
            None => crate::actor::spawn_player(&mut self.world_entities),
        }

        crate::ui::open_initial(&mut self.ui_entities, &self.resources);

        self.tick_systems.init();
        self.frame_systems.init();
    }
}
