mod control;

use crate::components;
use crate::ecs::*;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::coord::{Axis, Coord3};
use crate::world::{BlockPos, Gravity, World};
pub use control::*;
use glam::{Vec2, Vec3};
use std::f32::consts::FRAC_PI_2;
use std::sync::LazyLock;

/// The actor types the game can spawn, with the components each of them starts
/// with.
pub static ACTOR_TYPES: LazyLock<Registry<EntityDescriptor>> = LazyLock::new(|| {
    let mut actor_types = Registry::new();

    actor_types.register(
        0,
        "spectator",
        EntityDescriptor::new()
            .with(Position(Vec3::splat(16.0)))
            .with(PrevPos(Vec3::splat(16.0)))
            .with(Rotation(Vec3::ZERO))
            .with(Velocity(Vec3::ZERO))
            .with(Speed(0.5))
            .with(PlayerControlled)
            .with(Flight),
    );
    actor_types.register(
        1,
        "survivor",
        EntityDescriptor::new()
            .with(Position(Vec3::splat(16.0)))
            .with(PrevPos(Vec3::splat(16.0)))
            .with(Rotation(Vec3::ZERO))
            .with(Velocity(Vec3::ZERO))
            .with(Speed(0.25))
            .with(PlayerControlled)
            .with(Bound(AABB {
                min: Vec3::new(-0.25, -1.25, -0.25),
                max: Vec3::new(0.25, 0.25, 0.25),
            }))
            .with(Contact([None; 3]))
            .with(Selection(None)),
    );

    actor_types
});

components! {
    /// Position of an actor in the world, in blocks.
    #[derive(Clone, Copy)]
    pub struct Position(Vec3): Hot;
    /// Yaw and pitch of an actor in radians; the third component is unused.
    #[derive(Clone, Copy)]
    pub struct Rotation(Vec3): Hot;
    /// Velocity of an actor, in blocks per tick.
    #[derive(Clone, Copy)]
    pub struct Velocity(Vec3): Hot;
    /// Angular velocity of an actor, in radians per tick.
    #[derive(Clone, Copy)]
    pub struct Omega(Vec3): Hot;
    /// Distance an actor moves per tick while it is at full speed.
    #[derive(Clone, Copy)]
    pub struct Speed(f32): Hot;
    /// The box that collides with the world, relative to the position of the
    /// actor.
    #[derive(Clone, Copy)]
    pub struct Bound(AABB<Vec3>): Hot;
    /// The blocks an actor touches along each axis, and the side they are on:
    /// `Some(true)` for a block in the positive direction, `Some(false)` for one
    /// in the negative direction and `None` for no contact.
    #[derive(Clone, Copy)]
    pub struct Contact([Option<bool>; 3]): Hot;
    /// Marks an actor that ignores gravity and collisions.
    #[derive(Clone, Copy)]
    pub struct Flight: Cold;

    /// Position of an actor at the end of the previous tick, for interpolating
    /// the frames in between.
    #[derive(Clone, Copy)]
    pub struct PrevPos(Vec3): Cold;
}

impl Position {
    /// Moves the position by the velocity of one tick.
    #[inline]
    pub fn translate(&mut self, vel: &Velocity) {
        self.0 += vel.0;
    }
}

impl Rotation {
    /// Adds a rotation, clamping the pitch so that the actor cannot look past
    /// straight up or down.
    #[inline]
    pub fn rotate(&mut self, rot: Vec2) {
        self.0[0] += rot.x;
        self.0[1] += rot.y;
        self.0[1] = self.0[1].clamp(-FRAC_PI_2, FRAC_PI_2);
    }

    /// The unit vector the actor looks along.
    pub fn direction(&self) -> Vec3 {
        let (cy, sy, cp, sp) = (
            self.0[0].cos(),
            self.0[0].sin(),
            self.0[1].cos(),
            self.0[1].sin(),
        );
        Vec3::new(sy * cp, sp, -cy * cp).normalize()
    }
}

impl Velocity {
    /// Adds the movement of `dir`, which is relative to the yaw of the actor, at
    /// the speed `spd`.
    #[inline]
    pub fn accelerate(&mut self, rot: &Rotation, spd: &Speed, dir: Vec3) {
        if dir != Vec3::ZERO {
            let motion = dir.normalize() * spd.0;
            let yaw = rot.0[0];
            // `dir` is in the local space of the actor: `z` along the view
            // direction, `x` to the right.
            self.0 += Vec3::new(
                motion.z * yaw.sin() + motion.x * yaw.cos(),
                motion.y,
                -motion.z * yaw.cos() + motion.x * yaw.sin(),
            );
        }
    }
}

impl Bound {
    /// The collision box of an actor at its current position.
    #[inline]
    pub fn translate(&self, pos: &Position) -> AABB<Vec3> {
        self.0.translate(pos.0)
    }
}

/// Copies the position of every actor into [`PrevPos`].
pub struct Stalker;

/// Adds gravity to every actor that cannot fly.
pub struct Gravitator;

/// Moves every actor that has no collision box by its velocity.
pub struct Translator;

/// Moves every actor with a collision box and stops it at the blocks it hits.
pub struct Collider;

/// Damps the velocity of every actor, more strongly when it stands on a block.
pub struct Friction;

impl System for Stalker {
    type CompQuery = (CompRead<Position>, CompWrite<PrevPos>);
    type ResQuery = ();

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        entry.2.0 = entry.1.0;

        None
    }
}

impl System for Gravitator {
    type CompQuery = (CompWrite<Velocity>, Without<Flight>);
    type ResQuery = ResRead<Gravity>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        entry.1.0.y -= res.0;

        None
    }
}

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
    type CompQuery = (
        CompWrite<Position>,
        CompWrite<Velocity>,
        CompRead<Bound>,
        OptionalWrite<Contact>,
    );
    type ResQuery = ResRead<World>;

    fn operate(
        &mut self,
        mut entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        // The axes are resolved one after another, so movement that is blocked
        // on one axis still happens on the others; that is what lets an actor
        // slide along a wall instead of stopping in front of it.
        for &axis in Axis::ALL {
            let vel = entry.2.0.get(axis);
            entry.1.0 = entry.1.0.shift(axis, vel);
            let bound = entry.3.translate(entry.1);

            // Along the axis of movement only the block the far face ends up in
            // has to be tested; the other two axes are tested over the whole
            // range of the box.
            let [(minx, maxx), (miny, maxy), (minz, maxz)] = Axis::ALL.map(|a| {
                if a == axis {
                    if vel < 0.0 {
                        (
                            bound.min.get(a).floor() as i32,
                            bound.min.get(a).ceil() as i32,
                        )
                    } else {
                        (
                            bound.max.get(a).floor() as i32,
                            bound.max.get(a).ceil() as i32,
                        )
                    }
                } else {
                    (
                        bound.min.get(a).floor() as i32,
                        bound.max.get(a).ceil() as i32,
                    )
                }
            });

            let mut depth = 0.0;

            // The box is pushed back by the deepest overlap, which is the
            // distance that clears every block it entered.
            for x in minx..maxx {
                for y in miny..maxy {
                    for z in minz..maxz {
                        let pos = BlockPos::new(x, y, z);
                        let block = res.get_block(pos);
                        let block_bounds = block.bounds(pos);

                        let d = block_bounds
                            .into_iter()
                            .map(|b| {
                                if bound.intersects_with(b) {
                                    if vel < 0.0 {
                                        b.max.get(axis) - bound.min.get(axis)
                                    } else {
                                        bound.max.get(axis) - b.min.get(axis)
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
                entry.1.0 = entry
                    .1
                    .0
                    .shift(axis, if vel < 0.0 { depth } else { -depth });
                entry.2.0 = entry.2.0.with(axis, 0.0);
            }

            if let Some(contact) = entry.4.as_deref_mut() {
                // Remember which side the block was on, so that other systems
                // can tell whether the actor stands on the ground or hits a
                // ceiling.
                contact.0[axis.idx()] = if depth > 0.0 { Some(vel > 0.0) } else { None }
            }
        }

        None
    }
}

impl System for Friction {
    type CompQuery = (CompWrite<Velocity>, OptionalRead<Contact>);
    type ResQuery = ();

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        // Speed decays every tick, and much faster while the actor is on the
        // ground.
        entry.1.0 *= 0.9375;

        if let Some(contact) = entry.2
            && contact.0[Axis::Y.idx()].is_some()
        {
            entry.1.0 *= 0.25;
        }

        None
    }
}
