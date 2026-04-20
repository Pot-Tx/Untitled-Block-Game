use crate::util::coord::{Axis, Coord, Coord3, FCoord, FCoord3};
use num_traits::{FromPrimitive, Signed, Zero};
use std::ops::Neg;
use glam::{IVec3, Vec3};
use crate::actor::SelectedItem;
use crate::world::{Generate, World};

#[derive(Copy, Clone, Debug)]
pub struct AABB<C: Coord> {
    pub min: C,
    pub max: C,
}

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
        (self.min + self.max) / C::Scalar::from_i32(2).unwrap()
    }

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

    #[inline]
    pub fn intersects_with(&self, other: Self) -> bool {
        (0..C::DIM).all(|i| self.min[i] < other.max[i] && self.max[i] > other.min[i])
    }

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
        for i in 1..self.len() {
            joined = joined.merge(self[i]);
        }

        Some(joined)
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Plane<C: FCoord3> {
    pub normal: C,
    pub d: C::Scalar,
}

pub trait PlaneGroup {
    type Coord: Coord3<Scalar: Signed> + Neg<Output = Self::Coord>;

    fn is_point_inside(&self, point: Self::Coord) -> bool;

    fn is_aabb_inside(&self, aabb: AABB<Self::Coord>) -> bool;
}

impl<C: FCoord3> Plane<C> {
    pub fn from_points(p0: C, p1: C, p2: C, orient: C) -> Self {
        let dir1 = p1 - p0;
        let dir2 = p2 - p0;
        let mut normal = dir1.cross(dir2).normalize();
        let mut d = -normal.dot(p0);

        if normal.dot(orient) + d.clone() < C::Scalar::zero() {
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

#[derive(Copy, Clone, Default, Debug)]
pub struct Ray<C: FCoord> {
    pub origin: C,
    pub direction: C,
}

impl<C: FCoord> Ray<C> {
    pub fn intersects_with(&self, aabb: AABB<C>) -> bool {
        let mind = aabb.min - self.origin;
        let maxd = aabb.max - self.origin;
        
        let mut mint = C::default();
        let mut maxt = C::default();
        (0..C::DIM).for_each(|i| {
            mint[i] = if self.direction[i] > C::Scalar::zero() { mind[i] } else { maxd[i] } / self.direction[i];
            maxt[i] = if self.direction[i] < C::Scalar::zero() { mind[i] } else { maxd[i] } / self.direction[i];
        });
        
        mint.max_element() < maxt.min_element()
    }
    
    pub fn intersects_with_group(&self, aabbs: &[AABB<C>]) -> bool {
        aabbs.iter().any(|&aabb| self.intersects_with(aabb))
    }
}

impl Ray<Vec3> {
    pub fn traverse<G: Generate>(&self, world: &World<G>, reach: f32) -> Option<SelectedItem> {
        let mut origin = self.origin;
        let mut pos = self.origin.floor().as_ivec3();
        
        let step = self.direction.signum().as_ivec3();
        let offset = step.max(IVec3::ZERO);
        
        let mut axis = Axis::Y;
        
        while origin.distance(self.origin) < reach {
            let block = world.get_block(pos);
            
            if block.bounds().iter().any(|&aabb| self.intersects_with(aabb.translate(pos.as_vec3()))) {
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
