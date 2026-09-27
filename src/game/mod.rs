//! The game loop: the event loop handler, the timers and the input map.

mod client;
mod input;

use crate::game::client::GameClient;
use crate::resources;
use anyhow::Result;
use log::error;
use std::backtrace::Backtrace;
use std::panic;
use std::sync::LazyLock;
use std::time::Duration;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

pub use input::*;

/// Time between two simulation ticks.
pub static TICK_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_millis(50));
/// Shortest time between two frames.
pub static FRAME_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_millis(5));

resources! {
    /// Whether an open screen pauses the game.
    pub struct Paused(bool);
}

/// The game, which owns the client that drives the event loop.
#[derive(Default)]
pub struct Game {
    client: GameClient,
}

impl Game {
    /// Creates the game and the client that handles the window events.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn init(&self) {}

    /// Runs the event loop of the window until it is closed.
    pub fn run(&mut self) -> Result<(), EventLoopError> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        event_loop.run_app(&mut self.client)
    }

    /// Logs a panic and the backtrace it was raised with.
    ///
    /// Installed as the panic hook of the process, so that a crash ends up in
    /// the log instead of on a console that a release build does not have.
    pub fn crash(info: &panic::PanicHookInfo) {
        error!("game crashed: {}", info);
        error!("{}", Backtrace::capture());
    }
}
