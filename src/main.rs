use std::panic;
use core::game::Game;
use winit::error::EventLoopError;

fn main() -> Result<(), EventLoopError> {
    env_logger::init();
    panic::set_hook(Box::new(|info| {
        Game::crash(info);
    }));
    
    let mut game = Game::new();
    game.init();
    game.run()
}
