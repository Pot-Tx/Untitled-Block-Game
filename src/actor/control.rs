use crate::actor::*;
use crate::components;
use crate::ecs::*;
use crate::game;
use crate::game::*;
use glam::Vec3;

components! {
    pub struct PlayerControlled: Cold;
}

pub struct PlayerController;

pub struct PlayerRotator;

impl System for PlayerController {
	type CompQuery = (
		CompRead<PlayerControlled>,
		CompWrite<Velocity>,
		CompRead<Rotation>,
		CompRead<Speed>,
	);
	type ResQuery = ResRead<InputState>;
	
	fn operate(
		&mut self,
		entry: <Self::CompQuery as CompQuery>::Item<'_>,
		res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
	) -> Option<Vec<Command>> {
		let input_map = game::registries::input_map();
		let mut dir = Vec3::ZERO;
		
		if res.is_input_present(input_map.by_name("forward")) {
			dir.z += 1.0;
		}
		
		if res.is_input_present(input_map.by_name("left")) {
			dir.x -= 1.0;
		}
		
		if res.is_input_present(input_map.by_name("backward")) {
			dir.z -= 1.0;
		}
		
		if res.is_input_present(input_map.by_name("right")) {
			dir.x += 1.0;
		}
		
		if res.is_input_present(input_map.by_name("ascend")) {
			dir.y += 1.0;
		}
		
		if res.is_input_present(input_map.by_name("descend")) {
			dir.y -= 1.0;
		}
		
		entry.2.accelerate(entry.3, entry.4, dir);
		
		None
	}
}

impl System for PlayerRotator {
	type CompQuery = (CompRead<PlayerControlled>, CompWrite<Rotation>);
	type ResQuery = ResRead<InputState>;
	
	fn operate(
		&mut self,
		entry: <Self::CompQuery as CompQuery>::Item<'_>,
		res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
	) -> Option<Vec<Command>> {
		entry
			.2
			.rotate(res.mouse_motion * game::registries::mouse_sensitivity());
		
		None
	}
}
