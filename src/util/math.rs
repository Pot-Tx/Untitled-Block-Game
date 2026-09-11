use crate::util::coord::{Axis, SCoord3};
use num_traits::{One, Zero};
use smallvec::SmallVec;
use std::ops::Index;

pub trait AllEq {
    fn all_eq(&self) -> bool;
}

impl<T, const N: usize> AllEq for [T; N]
where
    T: Eq + PartialEq,
    [T; N]: Index<usize, Output = T>,
{
    fn all_eq(&self) -> bool {
        let value = &self[0];
        for i in 1..N {
            if value != &self[i] {
                return false;
            }
        }
        true
    }
}

impl<T: Eq + PartialEq> AllEq for Vec<T> {
    fn all_eq(&self) -> bool {
        let len = self.len();
        if len > 0 {
            let value = &self[0];
            for i in 1..len {
                if value != &self[i] {
                    return false;
                }
            }
        }
        true
    }
}

pub struct L1ShellIter<C: SCoord3> {
    pub radius: C::Scalar,
    pub center: C,
    x: C::Scalar,
    y: C::Scalar,
    z: C::Scalar,
    poses: SmallVec<[C; 8]>,
    idx: usize,
}

impl<C: SCoord3> L1ShellIter<C> {
    pub fn new(center: C, radius: C::Scalar) -> Self {
        L1ShellIter {
            radius: radius,
            center,
            x: C::Scalar::zero(),
            y: -C::Scalar::one(),
            z: radius + C::Scalar::one(),
            poses: SmallVec::new(),
            idx: 0,
        }
    }
}

impl<C: SCoord3> Iterator for L1ShellIter<C> {
    type Item = C;

    fn next(&mut self) -> Option<Self::Item> {
        self.idx += 1;
        if self.idx < self.poses.len() {
            return Some(self.poses[self.idx]);
        }

        if self.y < self.radius - self.x {
            self.y += C::Scalar::one();
            self.z -= C::Scalar::one();
        } else if self.x < self.radius {
            self.x += C::Scalar::one();
            self.y = C::Scalar::zero();
            self.z = self.radius - self.x;
        } else {
            return None;
        }

        let mut poses = SmallVec::<[C; 8]>::new();

        let mut xs = SmallVec::<[C::Scalar; 2]>::new();
        xs.push(self.x);
        if self.x != C::Scalar::zero() {
            xs.push(C::Scalar::zero() - self.x);
        }
        let mut ys = SmallVec::<[C::Scalar; 2]>::new();
        ys.push(self.y);
        if self.y != C::Scalar::zero() {
            ys.push(C::Scalar::zero() - self.y);
        }
        let mut zs = SmallVec::<[C::Scalar; 2]>::new();
        zs.push(self.z);
        if self.z != C::Scalar::zero() {
            zs.push(C::Scalar::zero() - self.z);
        }

        for &x in xs.iter() {
            for &y in ys.iter() {
                for &z in zs.iter() {
                    let p = C::new(x, y, z);
                    poses.push(self.center + p);
                }
            }
        }

        self.poses = poses;
        self.idx = 0;
        Some(self.poses[0])
    }
}

pub struct CubeShellIter<C: SCoord3> {
    pub origin: C,
    pub max: C,
    face: u8,
    i: C::Scalar,
    j: C::Scalar,
    i_min: C::Scalar,
    i_max: C::Scalar,
    j_min: C::Scalar,
    j_max: C::Scalar,
}

impl<C: SCoord3> CubeShellIter<C> {
    pub fn new(origin: C, side: C::Scalar) -> Self {
        let side = side - C::Scalar::one();
        let max = origin + C::new(side, side, side);
        let mut iter = Self {
            origin,
            max,
            face: 0,
            i: C::Scalar::zero(),
            j: C::Scalar::zero(),
            i_min: C::Scalar::zero(),
            i_max: C::Scalar::zero(),
            j_min: C::Scalar::zero(),
            j_max: C::Scalar::zero(),
        };
        iter.setup_face();
        iter.j = iter.j_min - C::Scalar::one();
        iter
    }

    pub fn from_center(center: C, radius: C::Scalar) -> Self {
        let one = C::Scalar::one();
        let side = radius + radius + one;
        let origin = C::new(
            center.get(Axis::X) - radius,
            center.get(Axis::Y) - radius,
            center.get(Axis::Z) - radius,
        );
        CubeShellIter::new(origin, side)
    }

    fn setup_face(&mut self) -> bool {
        let one = C::Scalar::one();
        let ox: C::Scalar = self.origin.get(Axis::X);
        let oy: C::Scalar = self.origin.get(Axis::Y);
        let oz: C::Scalar = self.origin.get(Axis::Z);
        let mx: C::Scalar = self.max.get(Axis::X);
        let my: C::Scalar = self.max.get(Axis::Y);
        let mz: C::Scalar = self.max.get(Axis::Z);

        match self.face {
            0 => {
                self.i_min = oy;
                self.i_max = my;
                self.j_min = oz;
                self.j_max = mz;
            }
            1 => {
                self.i_min = oy;
                self.i_max = my;
                self.j_min = oz;
                self.j_max = mz;
            }
            2 => {
                if mx <= ox + one {
                    return false;
                }
                self.i_min = ox + one;
                self.i_max = mx - one;
                self.j_min = oz;
                self.j_max = mz;
            }
            3 => {
                if mx <= ox + one {
                    return false;
                }
                self.i_min = ox + one;
                self.i_max = mx - one;
                self.j_min = oz;
                self.j_max = mz;
            }
            4 => {
                if mx <= ox + one || my <= oy + one {
                    return false;
                }
                self.i_min = ox + one;
                self.i_max = mx - one;
                self.j_min = oy + one;
                self.j_max = my - one;
            }
            5 => {
                if mx <= ox + one || my <= oy + one {
                    return false;
                }
                self.i_min = ox + one;
                self.i_max = mx - one;
                self.j_min = oy + one;
                self.j_max = my - one;
            }
            _ => return false,
        }

        self.i = self.i_min;
        self.j = self.j_min;
        true
    }

    fn current(&self) -> C {
        let ox: C::Scalar = self.origin.get(Axis::X);
        let oy: C::Scalar = self.origin.get(Axis::Y);
        let oz: C::Scalar = self.origin.get(Axis::Z);
        let mx: C::Scalar = self.max.get(Axis::X);
        let my: C::Scalar = self.max.get(Axis::Y);
        let mz: C::Scalar = self.max.get(Axis::Z);

        match self.face {
            0 => C::new(ox, self.i, self.j),
            1 => C::new(mx, self.i, self.j),
            2 => C::new(self.i, oy, self.j),
            3 => C::new(self.i, my, self.j),
            4 => C::new(self.i, self.j, oz),
            5 => C::new(self.i, self.j, mz),
            _ => unreachable!(),
        }
    }

    fn advance(&mut self) -> bool {
        let one = C::Scalar::one();
        self.j = self.j + one;
        if self.j <= self.j_max {
            return true;
        }
        self.j = self.j_min;
        self.i = self.i + one;
        if self.i <= self.i_max {
            return true;
        }
        loop {
            self.face += 1;
            if self.face > 5 {
                return false;
            }
            if self.setup_face() {
                return true;
            }
        }
    }
}

impl<C: SCoord3> Iterator for CubeShellIter<C> {
    type Item = C;

    fn next(&mut self) -> Option<Self::Item> {
        if self.advance() {
            Some(self.current())
        } else {
            None
        }
    }
}
