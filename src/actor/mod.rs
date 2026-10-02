mod control;

use crate::components;
use crate::ecs::*;
use crate::render::Canvas;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::coord::{Axis, Coord3};
use crate::world::{BlockPos, Gravity, World};
use crate::by_name;
use control::*;
pub(crate) use control::{PlayerControlled, SelectedItem};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::f32::consts::FRAC_PI_2;
use crate::util::OnceInit;

/// Every file names the components of an actor together with the values it
/// starts with; [`register`] and
/// [`EntityDescriptor::from_raw`](crate::ecs::EntityDescriptor::from_raw) turn
/// them into the descriptor an actor is spawned with.
static ACTOR_TYPES: OnceInit<Registry<EntityDescriptor>> = OnceInit::new();

/// Registers the components, the data and the systems of the actors.
pub(crate) fn register(
    entities: &mut EntityManager,
    tick: &mut SystemManager,
    frame: &mut SystemManager,
) {
    entities.components.register::<Position>("position");
    entities.components.register::<Rotation>("rotation");
    entities.components.register::<Velocity>("velocity");
    entities.components.register::<Speed>("speed");
    entities.components.register::<Bound>("bound");
    entities.components.register::<Flight>("flight");
    entities.components.register::<PlayerControlled>("player_controlled");

    ACTOR_TYPES.init(build_actor_types(&entities.components));

    tick.register(0, "player_controller", PlayerController);
    tick.register(1, "stalker", Stalker);
    tick.register(1, "gravitator", Gravitator);
    tick.register(2, "translator", Translator);
    tick.register(3, "collider", Collider);
    tick.register(4, "friction", Friction);
    tick.register(5, "selector", Selector);

    frame.register(1, "player_rotator", PlayerRotator);
    frame.register(2, "interactor", Interactor);
}

/// Registers the systems of the actors that draw, which need a canvas.
pub(crate) fn register_canvas(frame: &mut SystemManager, canvas: &Canvas) {
    frame.register(3, "selection_renderer", SelectionRenderer::new(canvas));
}

/// Spawns the actor the player steers.
pub(crate) fn spawn_player(entities: &mut EntityManager) {
    entities.spawn(by_name!(ACTOR_TYPES, "survivor").clone());
}

/// Builds the actor types of `assets/actors`.
fn build_actor_types(components: &ComponentManager) -> Registry<EntityDescriptor> {
    Registry::load_rons_from("assets/actors")
        .expect("failed to load actor types")
        .map(|raw| {
            EntityDescriptor::from_raw(components, raw)
                .expect("failed to create an actor type from data")
        })
}

components! {
    /// Position of an actor in the world, in blocks: where it is now and where
    /// it was at the end of the previous tick, which is what the frames in
    /// between are interpolated from.
    #[derive(Clone, Copy)]
    pub struct Position { pub cur: Vec3, pub prev: Vec3 }: Hot, data;

    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Rotation(Vec3): Hot, data;
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Velocity(Vec3): Hot, data;
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Speed(f32): Hot, data;

    /// The box that collides with the world, relative to the position of the
    /// actor, and the blocks it touched on the last tick: `Some(true)` for a
    /// block in the positive direction, `Some(false)` for one in the negative
    /// direction and `None` for no contact.
    #[derive(Clone, Copy)]
    pub struct Bound { pub bounds: AABB<Vec3>, pub contact: [Option<bool>; 3] }: Hot, data;

    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Flight: Cold, data;
}

/// Only the current position is written; an actor read from data starts out
/// standing still.
impl Serialize for Position {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.cur.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Position {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let cur = Vec3::deserialize(deserializer)?;

        Ok(Self { cur, prev: cur })
    }
}

/// Only the box is written; an actor read from data starts out touching
/// nothing.
impl Serialize for Bound {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.bounds.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Bound {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            bounds: AABB::deserialize(deserializer)?,
            contact: [None; 3],
        })
    }
}

impl Position {
    #[inline]
    pub fn translate(&mut self, vel: &Velocity) {
        self.cur += vel.0;
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
    #[inline]
    pub fn translate(&self, pos: &Position) -> AABB<Vec3> {
        self.bounds.translate(pos.cur)
    }
}

struct Stalker;

struct Gravitator;

struct Translator;

struct Collider;

struct Friction;

impl System for Stalker {
    type CompQuery = CompWrite<Position>;
    type ResQuery = ();

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        entry.1.prev = entry.1.cur;

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
    type CompQuery = (CompWrite<Position>, CompWrite<Velocity>, CompWrite<Bound>);
    type ResQuery = ResRead<World>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        // The axes are resolved one after another, so movement that is blocked
        // on one axis still happens on the others; that is what lets an actor
        // slide along a wall instead of stopping in front of it.
        for &axis in Axis::ALL {
            let vel = entry.2.0.get(axis);
            entry.1.cur = entry.1.cur.shift(axis, vel);
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
                entry.1.cur = entry
                    .1
                    .cur
                    .shift(axis, if vel < 0.0 { depth } else { -depth });
                entry.2.0 = entry.2.0.with(axis, 0.0);
            }

            // Remember which side the block was on, so that other systems can
            // tell whether the actor stands on the ground or hits a ceiling.
            entry.3.contact[axis.idx()] = if depth > 0.0 { Some(vel > 0.0) } else { None };
        }

        None
    }
}

impl System for Friction {
    type CompQuery = (CompWrite<Velocity>, OptionalRead<Bound>);
    type ResQuery = ();

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        // Speed decays every tick, and much faster while the actor is on the
        // ground.
        entry.1.0 *= 0.9375;

        if let Some(bound) = entry.2
            && bound.contact[Axis::Y.idx()].is_some()
        {
            entry.1.0 *= 0.25;
        }

        None
    }
}
