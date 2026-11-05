mod control;
mod path;
mod sprite;
mod store;

use crate::components;
use crate::ecs::*;
use crate::render::Canvas;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::coord::{Axis, Coord3};
use crate::util::transform::Trans4;
use crate::util::{Id, OnceInit};
use crate::world::{Block, BlockPos, Emission, Gravity, World};
use control::*;
pub(crate) use control::{PlayerControlled, SelectedItem};
use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub use sprite::{Animation, Animations, Appearance, Sprite, SpriteSet};
use std::f32::consts::FRAC_PI_2;
pub use store::{EntityStore, Handle, Scope, Session};

/// Every file names the components of an actor together with the values it
/// starts with; [`register`] and
/// [`EntityDescriptor::from_raw`](crate::ecs::EntityDescriptor::from_raw) turn
/// them into the descriptor an actor is spawned with.
pub static ACTOR_TYPES: OnceInit<Registry<EntityDescriptor>> = OnceInit::new();

/// Registers the components and the systems of the actors.
pub(crate) fn register(
    entities: &mut EntityManager,
    tick: &mut SystemManager<1>,
    frame: &mut SystemManager<2>,
) {
    entities.components.register::<Position>("position");
    entities.components.register::<Rotation>("rotation");
    entities.components.register::<Velocity>("velocity");
    entities.components.register::<Speed>("speed");
    entities.components.register::<Eye>("eye");
    entities.components.register::<Bound>("bound");
    entities.components.register::<Flight>("flight");
    entities
        .components
        .register::<PlayerControlled>("player_controlled");
    entities.components.register::<Inventory>("inventory");
    entities.components.register::<Persistent>("persistent");
    entities
        .components
        .register::<SessionEntity>("session_entity");
    entities.components.register::<Appearance>("appearance");
    sprite::register(entities, frame);
    entities.components.register::<Emission>("emission");

    tick.register(0, "player_controller", PlayerController);
    tick.register(1, "stalker", Stalker);
    tick.register(1, "gravitator", Gravitator);
    tick.register(2, "translator", Translator);
    path::register(entities, tick);
    tick.register(3, "collider", Collider);
    tick.register(4, "friction", Friction);
    tick.register(5, "selector", Selector);

    frame.register(1, "player_rotator", PlayerRotator);
    frame.register(2, "interactor", Interactor);
}

/// Loads the actor types and registers the resources and the systems that draw
/// the actors, which need a canvas.
///
/// A sprite set names the animations that drive it and an actor type names its
/// appearance, so the sets are built after the systems and the types after the
/// sets. The frames of a set are uploaded to the gpu while the set is read,
/// which is why this runs once the window exists.
pub(crate) fn register_canvas(
    entities: &mut EntityManager,
    resources: &mut ResourceManager,
    frame: &mut SystemManager<2>,
    canvas: &Canvas,
) {
    sprite::init(canvas, frame);

    ACTOR_TYPES.init(build_actor_types(&entities.components));

    sprite::register_canvas(resources, frame, canvas);

    // The outline of the selection is drawn after the actors, so that it stays
    // on top of them.
    frame.register(11, "selection_renderer", SelectionRenderer::new(canvas));
}

/// Spawns the actor the player steers.
pub(crate) fn spawn_player(entities: &mut EntityManager) {
    let mut commander = entities.commander().lock();

    commander.spawn(ACTOR_TYPES.by_name("survivor").clone().with(SessionEntity));
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

/// A named spawn point carried by a structure. Its position is local to the
/// region that owns the structure.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ActorMarker {
    pub pos: Vec3,
    pub actor: Id,
}

components! {
    /// Position of an actor in the world, in blocks: where its feet are now and
    /// where they were at the end of the previous tick, which is what the
    /// frames in between are interpolated from.
    ///
    /// Everything else of an actor is placed relative to its feet, so that the
    /// position says which block an actor stands on, no matter how tall it is.
    #[derive(Clone, Copy)]
    pub struct Position { pub cur: Vec3, pub prev: Vec3 }: Hot, asset, save;

    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Rotation(Vec3): Hot, asset, save;
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Velocity(Vec3): Hot, asset, save;
    /// How fast an actor moves under its own control and how fast it leaves the
    /// ground, in blocks per tick.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Speed { pub walk: f32, pub jump: f32 }: Hot, asset, save;

    /// The height of the eyes of an actor above its position, in blocks: where
    /// it looks from and where the camera of the player stands.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Eye(f32): Hot, asset, save;

    /// The box that collides with the world, relative to the position of the
    /// actor, and the blocks it touched on the last tick: `Some(true)` for a
    /// block in the positive direction, `Some(false)` for one in the negative
    /// direction and `None` for no contact.
    #[derive(Clone, Copy)]
    pub struct Bound { pub aabb: AABB<Vec3>, pub contact: [Option<bool>; 3] }: Hot, asset, save;

    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Flight: Cold, asset, save;

    /// An entity that outlives the region it stands in: when the region leaves
    /// the entity scope, the store keeps it instead of dropping it. This does
    /// not control whether the entity is saved while it is in the scope; the
    /// store saves every entity in the scope when the game exits.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Persistent: Cold, save;

    /// An entity that is written to the session file instead of the region it
    /// stands in, and that is therefore never part of the region scope. Only
    /// the player is one for now.
    #[derive(Clone, Copy)]
    pub struct SessionEntity: Cold;

    #[derive(Clone, Serialize, Deserialize)]
    #[serde(into = "RawInventory")]
    #[serde(from = "RawInventory")]
    pub struct Inventory {
        capacity: usize,
        blocks: Vec<Option<Block>>,
        changed: Vec<usize>,
        cur: usize,
        prev: usize,
    }: Cold, asset, save;
}

impl From<Vec3> for Position {
    fn from(value: Vec3) -> Self {
        Self {
            cur: value,
            prev: value,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RawInventory {
    capacity: usize,
    #[serde(default)]
    blocks: Option<Vec<Option<Block>>>,
    #[serde(default)]
    cur: usize,
}

impl From<Inventory> for RawInventory {
    fn from(value: Inventory) -> Self {
        Self {
            capacity: value.capacity,
            blocks: Some(value.blocks),
            cur: value.cur,
        }
    }
}

impl From<RawInventory> for Inventory {
    fn from(value: RawInventory) -> Self {
        let capacity = value.capacity;
        let blocks = value
            .blocks
            .map(|mut blocks| {
                blocks.resize(capacity, None);
                blocks
            })
            .unwrap_or(vec![None; capacity]);

        Self {
            capacity,
            blocks,
            changed: (0..capacity).collect(),
            cur: value.cur,
            prev: usize::MAX,
        }
    }
}

impl Inventory {
    pub fn cap(&self) -> usize {
        self.capacity
    }

    pub fn set_cap(&mut self, capacity: usize) {
        self.capacity = capacity;
        self.blocks.resize(capacity, None);
        self.cur = self.cur.min(capacity);
    }

    pub fn get(&self, index: usize) -> Option<Block> {
        self.blocks.get(index).cloned().unwrap_or(None)
    }

    pub fn set(&mut self, index: usize, value: Option<Block>) -> bool {
        if let Some(block) = self.blocks.get_mut(index) {
            *block = value;
            true
        } else {
            false
        }
    }

    pub fn changed(&self) -> Vec<usize> {
        self.changed.clone()
    }

    pub fn cur(&self) -> usize {
        self.cur
    }

    pub fn set_cur(&mut self, index: usize) {
        self.cur = index % self.capacity;
    }

    pub fn scroll(&mut self, delta: isize) {
        if delta != 0 {
            self.cur = (self.cur as isize + delta).rem_euclid(self.capacity as isize) as usize;
        }
    }

    pub fn prev(&self) -> usize {
        self.prev
    }

    pub fn flush(&mut self) {
        self.changed.clear();
        self.prev = self.cur;
    }
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
        self.aabb.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Bound {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            aabb: AABB::deserialize(deserializer)?,
            contact: [None; 3],
        })
    }
}

impl Position {
    /// Where the actor stands between the last two ticks, which is what the
    /// frames between them are drawn with.
    #[inline]
    pub fn interpolated(&self, partial: f32) -> Vec3 {
        self.prev + (self.cur - self.prev) * partial
    }

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

    /// Rotates a local vector into the world with this rotation.
    pub fn transform(&self, vector: Vec3) -> Vec3 {
        Mat4::rotation(self.0.x, self.0.y, self.0.z).transform_vector3(vector)
    }
}

impl Velocity {
    #[inline]
    pub fn accelerate(&mut self, spd: f32, dir: Vec3) {
        self.0 += dir.normalize_or_zero() * spd;
    }
}

impl Bound {
    #[inline]
    pub fn translate(&self, pos: &Position) -> AABB<Vec3> {
        self.aabb.translate(pos.cur)
    }
}

struct Stalker;

struct Gravitator;

struct Translator;

struct Collider;

struct Friction;

impl System<1> for Stalker {
    type CompQueries = ((CompWrite<Position>,),);
    type ResQuery = ();

    fn update(
        &mut self,
        comp: Self::CompQueries,
        _: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        for entry in comp.0.iter() {
            entry.1.prev = entry.1.cur;
        }
    }
}

impl System<1> for Gravitator {
    type CompQueries = ((CompWrite<Velocity>, Without<Flight>),);
    type ResQuery = (ResRead<Gravity>,);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        for entry in comp.0.iter() {
            entry.1.0.y -= res.0.0;
        }
    }
}

impl System<1> for Translator {
    type CompQueries = ((CompWrite<Position>, CompRead<Velocity>, Without<Bound>),);
    type ResQuery = ();

    fn update(
        &mut self,
        comp: Self::CompQueries,
        _: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        for entry in comp.0.iter() {
            entry.1.translate(entry.2);
        }
    }
}

impl System<1> for Collider {
    type CompQueries = ((CompWrite<Position>, CompWrite<Velocity>, CompWrite<Bound>),);
    type ResQuery = (ResRead<World>,);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        for entry in comp.0.iter() {
            // The axes are resolved one after another, so movement that is
            // blocked on one axis still happens on the others; that is what
            // lets an actor slide along a wall instead of stopping in front of
            // it.
            for &axis in Axis::ALL {
                let vel = entry.2.0.get(axis);
                entry.1.cur = entry.1.cur.shift(axis, vel);
                let bound = entry.3.translate(entry.1);

                // Along the axis of movement only the block the far face ends
                // up in has to be tested; the other two axes are tested over
                // the whole range of the box.
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
                            let block = res.0.get_block(pos);
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

                // Remember which side the block was on, so that other systems
                // can tell whether the actor stands on the ground or hits a
                // ceiling.
                entry.3.contact[axis.idx()] = if depth > 0.0 { Some(vel > 0.0) } else { None };
            }
        }
    }
}

impl System<1> for Friction {
    type CompQueries = ((CompWrite<Velocity>, OptionalRead<Bound>),);
    type ResQuery = ();

    fn update(
        &mut self,
        comp: Self::CompQueries,
        _: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        for entry in comp.0.iter() {
            // Speed decays every tick, and much faster while the actor is on
            // the ground.
            entry.1.0 *= 0.9375;

            if let Some(bound) = entry.2
                && bound.contact[Axis::Y.idx()].is_some()
            {
                entry.1.0 *= 0.25;
            }
        }
    }
}
