use core::game::Game;
use winit::error::EventLoopError;

fn main() -> Result<(), EventLoopError> {
	let mut game = Game::new();
	game.init();
	game.run()
}
