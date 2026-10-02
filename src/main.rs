//! Entry point of the game: installs the panic hook and runs the event loop.

use core::game::Game;
use std::panic;
use winit::error::EventLoopError;

fn main() -> Result<(), EventLoopError> {
    // Panics are reported through the logging framework.
    env_logger::init();
    panic::set_hook(Box::new(|info| {
        Game::crash(info);
    }));

    let mut game = Game::new();
    game.init();
    game.run()
}
