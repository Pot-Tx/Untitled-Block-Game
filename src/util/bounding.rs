use crate::actor::SelectedItem;
use crate::util::coord::*;
use crate::world::World;
use glam::{IVec3, Vec3};
use num_traits::{FromPrimitive, Signed, Zero};
use serde::{Deserialize, Serialize};
use std::ops::Neg;

/// An axis aligned box, stored as its minimum and maximum corner.
///
/// All containment tests treat `min` as inclusive and `max` as exclusive, so
/// adjacent boxes never overlap.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct AABB<C: Coord> {
    pub min: C,
    pub max: C,
}

/// A collection of [`AABB`]s that can be joined into the box enclosing all of them.
pub trait AABBGroup {
    type Coord: Coord;

    fn merge(&self) -> Option<AABB<Self::Coord>>;
}

impl<C: Coord> AABB<C> {
    #[inline]
    pub fn size(&self) -> C {
        self.max - self.min
    }

    #[inline]
    pub fn center(&self) -> C {
        (self.min + self.max)
            / C::Scalar::from_i32(2).expect("two should be representable as a scalar")
    }

    /// Moves the box by `dpos`.
    #[inline]
    #[must_use]
    pub fn translate(mut self, dpos: C) -> Self {
        self.min += dpos;
        self.max += dpos;
        self
    }

    #[inline]
    pub fn is_point_inside(&self, point: C) -> bool {
        for i in 0..C::DIM {
            if point[i] < self.min[i] || point[i] >= self.max[i] {
                return false;
            }
        }
        true
    }

    /// Tests whether the two half open boxes share any volume.
    #[inline]
    pub fn intersects_with(&self, other: Self) -> bool {
        (0..C::DIM).all(|i| self.min[i] < other.max[i] && self.max[i] > other.min[i])
    }

    /// The overlapping box, or `None` when the boxes are disjoint.
    #[inline]
    pub fn intersection(&self, other: Self) -> Option<Self> {
        let mut intersection = *self;
        for i in 0..C::DIM {
            if other.min[i] > intersection.min[i] {
                intersection.min[i] = other.min[i];
            }
            if other.max[i] < intersection.max[i] {
                intersection.max[i] = other.max[i];
            }
            if intersection.min[i] >= intersection.max[i] {
                return None;
            }
        }
        Some(intersection)
    }

    /// The smallest box enclosing both boxes.
    #[inline]
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        (0..C::DIM).for_each(|i| {
            if other.min[i] < self.min[i] {
                self.min[i] = other.min[i];
            }
            if other.max[i] > self.max[i] {
                self.max[i] = other.max[i];
            }
        });
        self
    }
}

impl<C: Coord> AABBGroup for [AABB<C>] {
    type Coord = C;

    fn merge(&self) -> Option<AABB<Self::Coord>> {
        if self.is_empty() {
            return None;
        }

        let mut joined = self[0];
        for &aabb in self.iter().skip(1) {
            joined = joined.merge(aabb);
        }

        Some(joined)
    }
}

/// A plane in the form `normal · point + d = 0`.
///
/// The positive side of the plane is the half space the normal points into,
/// which is the side [`Self::is_point_inside`] reports.
#[derive(Copy, Clone, Default, Debug)]
pub struct Plane<C: FCoord3> {
    pub normal: C,
    pub d: C::Scalar,
}

/// A set of planes describing a convex volume, such as a view frustum.
pub trait PlaneGroup {
    type Coord: Coord3<Scalar: Signed> + Neg<Output = Self::Coord>;

    fn is_point_inside(&self, point: Self::Coord) -> bool;

    fn is_aabb_inside(&self, aabb: AABB<Self::Coord>) -> bool;
}

impl<C: FCoord3> Plane<C> {
    /// Builds the plane through `p0`, `p1` and `p2`, with the normal flipped so
    /// that `orient` lies on the positive side.
    pub fn from_points(p0: C, p1: C, p2: C, orient: C) -> Self {
        let dir1 = p1 - p0;
        let dir2 = p2 - p0;
        let mut normal = dir1.cross(dir2).normalize();
        let mut d = -normal.dot(p0);

        if normal.dot(orient) + d < C::Scalar::zero() {
            normal = -normal;
            d = -d;
        }

        Self { normal, d }
    }

    #[inline]
    pub fn is_point_inside(&self, point: C) -> bool {
        self.normal.dot(point) + self.d > C::Scalar::zero()
    }

    #[inline]
    pub fn is_aabb_inside(&self, aabb: AABB<C>) -> bool {
        // Testing the corner with the smallest signed distance is enough: the
        // signed distance of a box is minimal in that corner.
        let mut point = C::default();

        Axis::ALL.iter().for_each(|&axis| {
            if self.normal.get(axis) < C::Scalar::zero() {
                point = point.with(axis, aabb.min.get(axis));
            } else {
                point = point.with(axis, aabb.max.get(axis));
            }
        });

        self.is_point_inside(point)
    }
}

impl<C: FCoord3> PlaneGroup for [Plane<C>] {
    type Coord = C;

    fn is_point_inside(&self, point: C) -> bool {
        for plane in self.iter() {
            if !plane.is_point_inside(point) {
                return false;
            }
        }
        true
    }

    fn is_aabb_inside(&self, aabb: AABB<Self::Coord>) -> bool {
        for plane in self.iter() {
            if !plane.is_aabb_inside(aabb) {
                return false;
            }
        }
        true
    }
}

/// A half line starting at `origin` and extending along `direction`.
#[derive(Copy, Clone, Default, Debug)]
pub struct Ray<C: FCoord> {
    pub origin: C,
    pub direction: C,
}

impl<C: FCoord> Ray<C> {
    /// Tests the box with the slab method, counting a ray that starts inside the
    /// box as an intersection.
    pub fn intersects_with(&self, aabb: AABB<C>) -> bool {
        let mind = aabb.min - self.origin;
        let maxd = aabb.max - self.origin;

        let mut mint = C::default();
        let mut maxt = C::default();
        (0..C::DIM).for_each(|i| {
            mint[i] = if self.direction[i] > C::Scalar::zero() {
                mind[i]
            } else {
                maxd[i]
            } / self.direction[i];
            maxt[i] = if self.direction[i] < C::Scalar::zero() {
                mind[i]
            } else {
                maxd[i]
            } / self.direction[i];
        });

        mint.max_element() < maxt.min_element()
    }

    pub fn intersects_with_group(&self, aabbs: &[AABB<C>]) -> bool {
        aabbs.iter().any(|&aabb| self.intersects_with(aabb))
    }
}

impl Ray<Vec3> {
    /// Marches through the voxels the ray crosses, up to `reach`, and returns the
    /// first block that the ray actually hits.
    ///
    /// The traversal is the 3D DDA of Amanatides and Woo: `origin` stays inside
    /// the current voxel while `pos` names that voxel, and the next voxel is
    /// entered across the axis whose boundary is crossed first.
    pub fn traverse(&self, world: &World, reach: f32) -> Option<SelectedItem> {
        let mut origin = self.origin;
        let mut pos = self.origin.floor().as_ivec3();

        // `offset` is the corner of the current voxel in the direction of travel.
        let step = self.direction.signum().as_ivec3();
        let offset = step.max(IVec3::ZERO);

        // The face the ray enters the current voxel through.
        let mut axis = Axis::Y;

        while origin.distance(self.origin) < reach {
            let block = world.get_block(pos);

            if block.bounds(pos).iter().any(|&b| self.intersects_with(b)) {
                return Some(SelectedItem::Block {
                    pos,
                    block,
                    face: axis.direction(self.direction.get(axis) < 0.0),
                });
            }

            let dpos = (pos + offset).as_vec3() - origin;
            let times = dpos / self.direction;

            let mut time = f32::INFINITY;

            for &a in Axis::ALL {
                let t = times.get(a);
                if t >= 0.0 && t < time {
                    time = t;
                    axis = a;
                }
            }

            origin += self.direction * time;
            pos = pos.shift(axis, step.get(axis));
        }

        None
    }
}
