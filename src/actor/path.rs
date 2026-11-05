//! Walking an actor to the block it wants to reach.
//!
//! The planner asks the navigator of the world for the blocks between an actor
//! and its goal, and the follower walks them. Neither of the two knows why the
//! actor goes there, so whatever picks the goal is free to change without
//! touching the way it is walked.

use super::{Bound, PlayerControlled, Position, Speed, Velocity};
use crate::components;
use crate::ecs::*;
use crate::util::coord::Axis;
use crate::world::{BlockPos, World, find_path};
use glam::{Vec3, Vec3Swizzles};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// How close to a waypoint a walker has to come to walk on to the next one.
///
/// A walker moves in steps of its own, so it can pass a block without ever
/// standing on its middle; measuring the distance counts that as reached, which
/// keeps it from turning back to a block it has already gone past.
const ARRIVAL: f32 = 0.5;

components! {
    /// The block an actor wants to reach, as the block its feet go to.
    #[derive(Clone, Copy)]
    pub struct Goal(BlockPos): Cold;

    /// How an actor gets there: the block the path leads to, the blocks it
    /// still has to walk through, without the one the actor stands in, and the
    /// tick the path was planned at, which is what the wait between two plans
    /// is counted from.
    ///
    /// A path is derived from the world and from the position of the actor, so
    /// it is not written to a save; the planner makes a new one after a load.
    #[derive(Clone)]
    pub struct Path {
        pub goal: BlockPos,
        pub waypoints: VecDeque<BlockPos>,
        pub planned: u32,
    }: Cold;

    /// How much work an actor may spend on finding a way: the blocks one search
    /// may look at and the ticks between two plans.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Pathing { pub nodes: u32, pub replan: u32 }: Cold, asset, save;
}

/// Registers the components and the systems that walk the actors of the world.
pub(super) fn register(entities: &mut EntityManager, tick: &mut SystemManager<1>) {
    entities.components.register::<Goal>("goal");
    entities.components.register::<Path>("path");
    entities.components.register::<Pathing>("pathing");

    // The planner comes first, so that a path planned in a tick is walked in
    // the same one.
    tick.register(2, "planner", Planner::new());
    tick.register(2, "follower", Follower);
}

/// Plans the way of the actors that were given a goal.
struct Planner {
    /// The tick the planner is at, which is what a path is stamped with.
    tick: u32,
}

impl Planner {
    fn new() -> Self {
        Self { tick: 0 }
    }
}

impl System<1> for Planner {
    type CompQueries = ((
        CompRead<Goal>,
        CompRead<Pathing>,
        CompRead<Position>,
        CompRead<Bound>,
        // The path is written through the query that reads the goal, because a
        // system may only reach the components of the entities it iterates.
        OptionalWrite<Path>,
        // The player is steered by its own controller.
        Without<PlayerControlled>,
    ),);
    type ResQuery = (ResRead<World>,);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        com: [Commander<'_>; 1],
    ) {
        let (world,) = res;

        self.tick = self.tick.wrapping_add(1);

        let mut commands = com[0].lock();

        for (id, goal, pathing, position, bound, path, _) in comp.0.iter() {
            let goal = goal.0;

            // The goal of an actor that follows a moving target changes every
            // few ticks, so an actor waits between two plans even then; the
            // first plan of an actor is made at once.
            let (moved, waited) = match path.as_deref() {
                None => (true, true),
                Some(path) => (
                    path.goal != goal,
                    self.tick.wrapping_sub(path.planned) >= pathing.replan.max(1),
                ),
            };

            // The turns of the actors are staggered by their ids, so that the
            // ones that follow the same goal do not all plan in the same tick.
            let due = self.tick.wrapping_add(id) % pathing.replan.max(1) == 0;

            if !waited || (!moved && !due) {
                continue;
            }

            let mut budget = pathing.nodes;
            let planned = find_path(
                bound.aabb,
                |pos| world.get_block(pos).is_solid(),
                position.cur.floor().as_ivec3(),
                goal,
                &mut budget,
            );

            let waypoints = VecDeque::from(planned.unwrap_or_default());

            match path {
                Some(path) => {
                    path.goal = goal;
                    path.waypoints = waypoints;
                    path.planned = self.tick;
                }

                None => {
                    commands.insert(
                        id,
                        Path {
                            goal,
                            waypoints,
                            planned: self.tick,
                        },
                    );
                }
            }
        }
    }
}

/// Walks the actors that have a path, one waypoint at a time.
struct Follower;

impl System<1> for Follower {
    type CompQueries = ((
        CompWrite<Path>,
        CompWrite<Velocity>,
        CompRead<Position>,
        CompRead<Speed>,
        CompRead<Bound>,
        // The player is steered by its own controller.
        Without<PlayerControlled>,
    ),);
    type ResQuery = ();

    fn update(
        &mut self,
        comp: Self::CompQueries,
        _: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        for (_, path, vel, pos, spd, bound, _) in comp.0.iter() {
            while path.waypoints.front().is_some_and(|waypoint| {
                center(*waypoint).xz().distance_squared(pos.cur.xz()) <= ARRIVAL * ARRIVAL
            }) {
                path.waypoints.pop_front();
            }

            let Some(next) = path.waypoints.front() else {
                // The actor arrived, or there is no way to go; whatever is left
                // of its speed is left to the friction.
                continue;
            };

            let dir = (center(*next) - pos.cur).with_y(0.0);
            vel.accelerate(spd.walk, dir);

            // The next block sits above the one the actor stands in, so it has
            // to leave the ground to get there. Only a block it touched on the
            // last tick can push it off.
            let standing = pos.cur.floor().as_ivec3();

            if next.y > standing.y && bound.contact[Axis::Y.idx()] == Some(false) {
                vel.0.y = spd.jump;
            }
        }
    }
}

/// The middle of a block, which is the point a walker steers at.
#[inline]
fn center(pos: BlockPos) -> Vec3 {
    pos.as_vec3() + Vec3::new(0.5, 0.0, 0.5)
}
