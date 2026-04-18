use crate::actor::*;
use crate::components;
use crate::ecs::*;
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
        let mut dir = Vec3::ZERO;

        if res.is_action_present(0) {
            dir.z += 1.0;
        }

        if res.is_action_present(1) {
            dir.x -= 1.0;
        }

        if res.is_action_present(2) {
            dir.z -= 1.0;
        }

        if res.is_action_present(3) {
            dir.x += 1.0;
        }

        if res.is_action_present(4) {
            dir.y += 1.0;
        }

        if res.is_action_present(5) {
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
        entry.2.rotate(res.mouse_motion * *MOUSE_SENSITIVITY);

        None
    }
}
