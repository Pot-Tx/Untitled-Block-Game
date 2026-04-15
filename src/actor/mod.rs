mod control;

use crate::components;
use crate::ecs::*;
use glam::{Vec2, Vec3};
use std::f32::consts::FRAC_PI_2;

pub use control::*;

pub mod registries {
	use crate::actor::control::PlayerControlled;
	use crate::actor::{ActorType, Position, Rotation, Speed, Velocity};
	use crate::ecs::EntityDescriptor;
	use crate::util::collection::Registry;
	use glam::Vec3;
	use log::error;
	use std::sync::OnceLock;
	
	static ACTOR_TYPES: OnceLock<Registry<ActorType>> = OnceLock::new();
	
	pub fn init() {
		let mut actor_types = Registry::new();
		
		actor_types.register(
			"spectator",
			ActorType {
				descriptor: || {
					EntityDescriptor::new()
						.with(Position(Vec3::splat(32.0)))
						.with(Rotation(Vec3::ZERO))
						.with(Velocity(Vec3::ZERO))
						.with(Speed(0.5))
						.with(PlayerControlled)
				},
			},
		);
		
		if ACTOR_TYPES.set(actor_types).is_err() {
			error!("Actor Types already initialized");
		}
	}
	
	pub fn actor_types() -> &'static Registry<ActorType> {
		ACTOR_TYPES
			.get()
			.expect("Failed to get Actor Types. Make sure to call registries::init first!")
	}
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
