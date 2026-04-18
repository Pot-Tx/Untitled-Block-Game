mod control;

use crate::components;
use crate::ecs::*;
use crate::util::collection::Registry;
pub use control::*;
use glam::{Vec2, Vec3};
use std::f32::consts::FRAC_PI_2;
use std::sync::LazyLock;

pub static ACTOR_TYPES: LazyLock<Registry<ActorType>> = LazyLock::new(|| build_actor_types());

fn build_actor_types() -> Registry<ActorType> {
    let mut actor_types = Registry::new();

    let spectator = ActorType {
        descriptor: || {
            EntityDescriptor::new()
                .with(Position(Vec3::splat(32.0)))
                .with(Rotation(Vec3::ZERO))
                .with(Velocity(Vec3::ZERO))
                .with(Speed(0.5))
                .with(PlayerControlled)
        },
    };

    actor_types.register(0, spectator);

    actor_types
}

pub struct ActorType {
    descriptor: fn() -> EntityDescriptor,
}

impl ActorType {
    pub fn create(&self) -> EntityDescriptor {
        (self.descriptor)()
    }
}

components! {
    pub struct Position(Vec3): Hot;
    pub struct Rotation(Vec3): Hot;
    pub struct Velocity(Vec3): Hot;
    pub struct Omega(Vec3): Hot;
    pub struct Speed(f32): Hot;
}

impl Position {
    #[inline]
    pub fn translate(&mut self, vel: &Velocity) {
        self.0 += vel.0;
    }
}

impl Rotation {
    #[inline]
    pub fn rotate(&mut self, rot: Vec2) {
        self.0[0] += rot.x;
        self.0[1] += rot.y;
        self.0[1] = self.0[1].clamp(-FRAC_PI_2, FRAC_PI_2);
    }
}

impl Velocity {
    #[inline]
    pub fn accelerate(&mut self, rot: &Rotation, spd: &Speed, dir: Vec3) {
        if dir != Vec3::ZERO {
            let vector = dir.normalize() * spd.0;
            let yaw = rot.0[0];
            self.0 += Vec3::new(
                vector.z * yaw.sin() + vector.x * yaw.cos(),
                vector.y,
                -vector.z * yaw.cos() + vector.x * yaw.sin(),
            );
        }
    }
}

pub struct Translator;

pub struct Friction;

impl System for Translator {
    type CompQuery = (CompWrite<Position>, CompRead<Velocity>);
    type ResQuery = ();

    fn operate<'a>(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'a>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'a>,
    ) -> Option<Vec<Command>> {
        entry.1.translate(entry.2);
        None
    }
}

impl System for Friction {
    type CompQuery = CompWrite<Velocity>;
    type ResQuery = ();

    fn operate<'a>(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'a>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'a>,
    ) -> Option<Vec<Command>> {
        entry.1.0 *= 0.5;
        None
    }
}
