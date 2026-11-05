//! Finding a way over the blocks of the world.
//!
//! A path is the list of blocks an actor walks through, which is what lets it
//! step up over a block or walk around a wall instead of pushing into whatever
//! it happens to hit. The search takes two things from the world: which blocks
//! are solid, and how large the actor is, which it reads from the collision box
//! of the actor.

use super::BlockPos;
use crate::util::bounding::AABB;
use glam::{IVec3, Vec3};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// The horizontal steps of the search: the four sides and the four corners.
const STEPS: [IVec3; 8] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
    IVec3::new(1, 0, 1),
    IVec3::new(1, 0, -1),
    IVec3::new(-1, 0, 1),
    IVec3::new(-1, 0, -1),
];

/// What a step to the side and a step across a corner cost, in the cheapest
/// unit of the search. A corner costs more, so that a walker keeps to the
/// straightest way it can find.
const SIDE: u32 = 2;
const CORNER: u32 = 3;

/// What leaving the level a walker stands on costs, on top of the step itself,
/// so that it walks around a block when that is about as far as climbing over
/// it.
const CLIMB: u32 = 2;

/// Finds the blocks a walker of the given collision box walks through to get
/// from `start` to `goal`, or `None` when there is no way or the budget ran
/// out.
///
/// The blocks the walker reaches are the ones it can stand on, so the search
/// never leaves the surface it walks on. `budget` is how many blocks the search
/// may look at, which bounds the work of one call; whatever is left of it is
/// left in place.
pub fn find_path(
    bound: AABB<Vec3>,
    solid: impl Fn(BlockPos) -> bool,
    start: BlockPos,
    goal: BlockPos,
    budget: &mut u32,
) -> Option<Vec<BlockPos>> {
    let height = height(bound);

    // A walker that is in the air, or whose goal is, is on its way to or from
    // the block it can stand on next to it.
    let start = standing(height, &solid, start)?;
    let goal = standing(height, &solid, goal)?;

    if start == goal {
        return Some(Vec::new());
    }

    let mut open = BinaryHeap::new();
    let mut cost = HashMap::new();
    let mut from = HashMap::new();

    cost.insert(start, 0);
    open.push(Node {
        pos: start,
        cost: 0,
        estimate: estimate(start, goal),
    });

    while let Some(node) = open.pop() {
        // The heap may hold a block more than once; the cost is what tells the
        // entry that is still current from the ones that were overtaken.
        if node.cost > cost[&node.pos] {
            continue;
        }

        if *budget == 0 {
            return None;
        }
        *budget -= 1;

        if node.pos == goal {
            return Some(retrace(&from, start, goal));
        }

        for (next, step) in steps(height, &solid, node.pos) {
            let reached = node.cost + step;

            if cost.get(&next).is_some_and(|&cost| cost <= reached) {
                continue;
            }

            cost.insert(next, reached);
            from.insert(next, node.pos);
            open.push(Node {
                pos: next,
                cost: reached,
                estimate: reached + estimate(next, goal),
            });
        }
    }

    None
}

/// How many blocks above its feet a walker of the given box is made of.
fn height(bound: AABB<Vec3>) -> i32 {
    (bound.max.y.ceil() as i32 - bound.min.y.floor() as i32).max(1)
}

/// Whether the blocks a walker of `height` blocks is made of at `pos` are all
/// free.
fn fits(height: i32, solid: &impl Fn(BlockPos) -> bool, pos: BlockPos) -> bool {
    (0..height).all(|step| !solid(pos + IVec3::Y * step))
}

/// Whether a walker of `height` blocks can stand at `pos`, which needs room for
/// its body and a block to stand on.
fn stand(height: i32, solid: &impl Fn(BlockPos) -> bool, pos: BlockPos) -> bool {
    fits(height, solid, pos) && solid(pos - IVec3::Y)
}

/// The block near `pos` a walker of `height` blocks can stand at, or `None`
/// when it can stand nowhere around it.
fn standing(height: i32, solid: &impl Fn(BlockPos) -> bool, pos: BlockPos) -> Option<BlockPos> {
    [pos, pos - IVec3::Y, pos + IVec3::Y]
        .into_iter()
        .find(|&pos| stand(height, solid, pos))
}

/// The blocks a walker of `height` blocks can reach from `pos`, and what each
/// of them costs.
///
/// A walker steps to the side and across a corner, and it steps up or down by
/// one block. It only steps up where it has room over its head, and it only
/// walks a corner that at least one of the two blocks it passes is walkable
/// at, so that it does not cut through a wall.
fn steps(
    height: i32,
    solid: &impl Fn(BlockPos) -> bool,
    pos: BlockPos,
) -> impl Iterator<Item = (BlockPos, u32)> + '_ {
    STEPS.into_iter().filter_map(move |step| {
        let next = pos + step;
        let corner = step.x != 0 && step.z != 0;
        let cost = if corner { CORNER } else { SIDE };

        if stand(height, solid, next) {
            return Some((next, cost));
        }

        if corner {
            let sides = [
                pos + IVec3::new(step.x, 0, 0),
                pos + IVec3::new(0, 0, step.z),
            ];

            if !sides.into_iter().any(|side| stand(height, solid, side)) {
                return None;
            }
        }

        if fits(height, solid, pos + IVec3::Y) && stand(height, solid, next + IVec3::Y) {
            return Some((next + IVec3::Y, cost + CLIMB));
        }

        if stand(height, solid, next - IVec3::Y) {
            return Some((next - IVec3::Y, cost + CLIMB));
        }

        None
    })
}

/// The cost from `pos` to `goal`, counting the steps of a walk over level
/// ground. Climbing is left out on purpose, so that the estimate never
/// overstates what is really left to walk.
fn estimate(pos: BlockPos, goal: BlockPos) -> u32 {
    let (dx, dz) = (
        (goal.x - pos.x).unsigned_abs(),
        (goal.z - pos.z).unsigned_abs(),
    );

    SIDE * dx.max(dz) + (CORNER - SIDE) * dx.min(dz)
}

/// The blocks from `start` to `goal`, following the parents of `from` back and
/// dropping the blocks in between that lie on the straight line from the one
/// before them to the one after them, so that a walker walks a corridor instead
/// of block by block.
fn retrace(from: &HashMap<BlockPos, BlockPos>, start: BlockPos, goal: BlockPos) -> Vec<BlockPos> {
    let mut path = vec![goal];
    let mut pos = goal;

    while pos != start {
        pos = from[&pos];
        path.push(pos);
    }

    path.reverse();

    // The walker already stands on the block the path starts at, so it is only
    // kept while the blocks are merged; it is what tells the first step of the
    // way from the ones after it.
    let mut merged = merge(path);
    merged.remove(0);

    merged
}

/// Drops the blocks that lie on the straight line between the ones around them,
/// so that a walker walks a corridor instead of block by block.
fn merge(path: Vec<BlockPos>) -> Vec<BlockPos> {
    let mut merged = Vec::with_capacity(path.len());

    for (i, &pos) in path.iter().enumerate() {
        let straight = i > 0 && i + 1 < path.len() && pos - path[i - 1] == path[i + 1] - pos;

        if !straight {
            merged.push(pos);
        }
    }

    merged
}

/// One block the search has reached: what it costs to get there and the
/// estimate of what is left to walk, ordered so that the smallest estimate
/// comes out of the heap first.
struct Node {
    pos: BlockPos,
    cost: u32,
    estimate: u32,
}

impl PartialEq for Node {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
    }
}

impl Eq for Node {}

impl Ord for Node {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        other.estimate.cmp(&self.estimate)
    }
}

impl PartialOrd for Node {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
