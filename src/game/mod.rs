mod client;
mod input;

use crate::game::client::GameClient;
use crate::{actor, world};
use anyhow::Result;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

pub use input::*;

mod constants {
    use std::sync::OnceLock;
    use std::time::Duration;
    
    static TICK_DURATION: OnceLock<Duration> = OnceLock::new();
    static FRAME_DURATION: OnceLock<Duration> = OnceLock::new();

    pub fn init() {
        let tick_duration = Duration::from_millis(50);
        let frame_duration = Duration::from_millis(20);

        TICK_DURATION
            .set(tick_duration)
            .expect("Failed to set tick duration");
        FRAME_DURATION
            .set(frame_duration)
            .expect("Failed to set frame duration");
    }

    #[inline]
    pub fn tick_duration() -> Duration {
        *TICK_DURATION
            .get()
            .expect("Failed to get Tick Duration. Make sure to call constants::init first!")
    }

    #[inline]
    pub fn frame_duration() -> Duration {
        *FRAME_DURATION
            .get()
            .expect("Failed to get Frame Duration. Make sure to call constants::init first!")
    }
}

pub mod registries {
    use crate::game::*;
    pub use input::registries::{input_map, mouse_sensitivity};
    
    pub fn init() {
        input::registries::init_input_map();
        input::registries::init_mouse_sensitivity();
    }
}

pub struct Game {
    client: GameClient,
}

impl Game {
    pub fn new() -> Self {
        let client = GameClient::new();

        Self { client }
    }

    pub fn init(&self) {
        constants::init();
        registries::init();
        world::registries::init();
        actor::registries::init();
    }

    pub fn run(&mut self) -> Result<(), EventLoopError> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        event_loop.run_app(&mut self.client)
    }
}
