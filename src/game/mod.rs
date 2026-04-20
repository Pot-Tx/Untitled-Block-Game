mod client;
mod input;

use std::backtrace::Backtrace;
use std::panic;
use crate::game::client::GameClient;
use anyhow::Result;
use std::sync::LazyLock;
use std::time::Duration;
use log::error;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

pub use input::*;

pub static TICK_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_millis(50));
pub static FRAME_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_millis(5));

pub struct Game {
    client: GameClient,
}

impl Game {
    pub fn new() -> Self {
        let client = GameClient::new();

        Self { client }
    }

    pub fn init(&self) {}

    pub fn run(&mut self) -> Result<(), EventLoopError> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        event_loop.run_app(&mut self.client)
    }
    
    pub fn crash(info: &panic::PanicHookInfo) {
        error!("{}", info);
        let trace = Backtrace::capture();
        error!("{}", trace);
        error!("Oh no, Game crashed!");
    }
}
