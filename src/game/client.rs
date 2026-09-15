use crate::actor::*;
use crate::ecs::*;
use crate::game::{Escaper, InputFlusher, InputState, FRAME_DURATION, TICK_DURATION};
use crate::render::ViewPort;
use crate::render::*;
use crate::ui::{create_sprite_geometry, CrosshairRenderer, SPRITE_GEOMETRY};
use crate::util::coord::{Direction, ICoord3};
use crate::util::OnceInit;
use crate::world::*;
use crossbeam_channel::unbounded;
use glam::{Vec3, Vec3Swizzles};
use noise_functions::{CellDistanceSq, Noise, Perlin};
use rayon::ThreadPoolBuilder;
use smallvec::smallvec;
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
                    .expect("failed to create window"),
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
                self.simulation
                    .resources
                    .get_mut::<ViewPort>()
                    .transform(self.simulation.resources.get::<Canvas>());
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

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.simulation.resources.get_mut::<World>().save();
    }
}

impl GameClient {
    pub fn new() -> Self {
        let (gen_tx, gen_rx) = unbounded();
        let (meshing_tx, meshing_rx) = unbounded();

        let mut simulation = Simulation::new();

        simulation.components.register::<Position>();
        simulation.components.register::<PrevPos>();
        simulation.components.register::<Rotation>();
        simulation.components.register::<Velocity>();
        simulation.components.register::<Speed>();
        simulation.components.register::<PlayerControlled>();
        simulation.components.register::<Bound>();
        simulation.components.register::<Contact>();
        simulation.components.register::<Flight>();
        simulation.components.register::<Option<Selection>>();

        simulation.systems.register(0, PlayerController);
        simulation.systems.register(1, Stalker);
        simulation.systems.register(1, Gravitator);
        simulation.systems.register(2, Translator);
        simulation.systems.register(3, Collider);
        simulation.systems.register(4, Friction);
        simulation.systems.register(5, Selector);
        simulation.systems.register(6, WorldUpdater);
        simulation.systems.register(7, ChunkMeshing);

        simulation.resources.register(InputState::new());
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
        simulation
            .resources
            .register(WorldThreads(near_threads, far_threads));
        simulation.resources.register(World::new(
            RegionPos::ZERO,
            smallvec![3, 7, 13, 21],
            gen_tx,
            meshing_tx,
        ));
        simulation.resources.register(Generator::new(
            RegionPos::ZERO,
            22,
            Field {
                climate: |_| -> Vec3 { Vec3::ZERO },
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
            |meta, chunk, pos| -> Meta {
                if meta == 2 && *chunk.get(pos.step(Direction::Up)) == 0 {
                    3
                } else {
                    meta
                }
            },
            gen_rx,
        ));
        simulation.resources.register(ChunkMesher::new(meshing_rx));
        simulation.resources.register::<Option<Frame>>(None);
        simulation.resources.register(PartialTick(0.0));
        simulation.resources.register(Gravity(0.125));

        let mut render_systems = SystemManager::new();

        render_systems.register(0, Escaper);
        render_systems.register(1, PlayerRotator);
        render_systems.register(2, Interactor);
        render_systems.register(2, CameraTransformer);
        render_systems.register(3, InputFlusher);
        render_systems.register(3, RenderStarter);
        render_systems.register(6, RenderFinisher);

        Self {
            frame_timer: Instant::now(),
            tick_timer: Instant::now(),

            simulation,
            render_systems,
        }
    }

    pub fn init(&mut self) {
        let canvas = pollster::block_on(Canvas::new(&WINDOW));
        SPRITE_GEOMETRY.init(create_sprite_geometry(&canvas));

        let block_textures = BlockTextures(BLOCK_TEXTURES.create_texture_sampler(&canvas, "block"));
        let camera = Camera::new(&canvas);
        let world_renderer = WorldRenderer::new(&canvas);
        let selection_renderer = SelectionRenderer::new(&canvas);
        let crosshair_renderer = CrosshairRenderer::new(&canvas);
        let viewport = ViewPort::new(&canvas);

        self.render_systems.register(4, world_renderer);
        self.render_systems.register(4, selection_renderer);
        self.render_systems.register(5, crosshair_renderer);
        self.simulation.resources.register(canvas);
        self.simulation.resources.register(block_textures);
        self.simulation.resources.register(camera);
        self.simulation.resources.register(viewport);
        self.simulation.spawn(ACTOR_TYPES.get(1).create());

        self.simulation.systems.init();
        self.render_systems.init();
    }
}
