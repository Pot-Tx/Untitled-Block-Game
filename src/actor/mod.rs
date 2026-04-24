mod control;

use crate::components;
use crate::ecs::*;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::coord::{Axis, Coord3};
use crate::world::{BlockPos, TestGen, World};
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
    let survivor = ActorType {
        descriptor: || {
            EntityDescriptor::new()
                .with(Position(Vec3::splat(32.0)))
                .with(Rotation(Vec3::ZERO))
                .with(Velocity(Vec3::ZERO))
                .with(Speed(0.5))
                .with(PlayerControlled)
                .with(Bound(AABB { min: Vec3::new(-0.25, -1.25, -0.25), max: Vec3::new(0.25, 0.25, 0.25) }))
                .with(Option::<Selection>::None)
        }
    };

    actor_types.register(0, spectator);
    actor_types.register(1, survivor);

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
    pub struct Bound(AABB<Vec3>): Hot;
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
    
    pub fn direction(&self) -> Vec3 {
        let (cy, sy, cp, sp) = (self.0[0].cos(), self.0[0].sin(), self.0[1].cos(), self.0[1].sin());
        Vec3::new(sy * cp, sp, -cy * cp).normalize()
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

pub struct Collider;

pub struct Friction;

impl System for Translator {
    type CompQuery = (CompWrite<Position>, CompRead<Velocity>, Without<Bound>);
    type ResQuery = ();
    
    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        entry.1.translate(entry.2);
        
        None
    }
}

impl System for Collider {
    type CompQuery = (CompWrite<Position>, CompWrite<Velocity>, CompRead<Bound>);
    type ResQuery = ResRead<World<TestGen>>;
    
    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        for &axis in Axis::ALL {
            let vel = entry.2.0.get(axis);
            entry.1.0 = entry.1.0.shift(axis, vel);
            let bound = entry.3.0.translate(entry.1.0);
            
            let [(minx, maxx), (miny, maxy), (minz, maxz)] = Axis::ALL.map(|a| {
                if a == axis {
                    if vel < 0.0 {
                        (bound.min.get(a).floor() as i32, bound.min.get(a).ceil() as i32)
                    } else {
                        (bound.max.get(a).floor() as i32, bound.max.get(a).ceil() as i32)
                    }
                } else {
                    (bound.min.get(a).floor() as i32, bound.max.get(a).ceil() as i32)
                }
            });
            
            let mut depth = 0.0;
            
            for x in minx..maxx {
                for y in miny..maxy {
                    for z in minz..maxz {
                        let pos = BlockPos::new(x, y, z);
                        let block = res.get_block(pos);
                        let block_bounds = block.bounds();
                        
                        let d = block_bounds.iter().map(|&b| {
                            let block_bound = b.translate(pos.as_vec3());
                            if bound.intersects_with(block_bound) {
                                if vel < 0.0 {
                                    block_bound.max.get(axis) - bound.min.get(axis)
                                } else {
                                    bound.max.get(axis) - block_bound.min.get(axis)
                                }
                            } else {
                                0.0
                            }
                        })
                            .reduce(f32::max)
                            .unwrap_or(0.0)
                            .max(0.0);
                        
                        if d > depth {
                            depth = d;
                        }
                    }
                }
            }
            
            if depth > 0.0 {
                if vel > 0.0 {
                    depth = -depth;
                }
                
                entry.1.0 = entry.1.0.shift(axis, depth);
                entry.2.0 = entry.2.0.with(axis, 0.0);
            }
        }
        
        None
    }
}

impl System for Friction {
    type CompQuery = CompWrite<Velocity>;
    type ResQuery = ();
    
    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        entry.1.0 *= 0.5;
        
        None
    }
}
