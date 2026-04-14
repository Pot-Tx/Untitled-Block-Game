use crate::util::bounding::Plane;
use bytemuck::{Pod, Zeroable};
use glam::*;
use num_traits::*;
use std::ops::*;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Direction {
    West,
    East,
    Down,
    Up,
    North,
    South,
}

impl Axis {
    pub const ALL: &'static [Self] = &[Self::X, Self::Y, Self::Z];

    #[inline]
    pub const fn by_idx(idx: usize) -> Self {
        Self::ALL[idx]
    }

    #[inline]
    pub const fn idx(&self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    #[inline]
    pub const fn others(&self) -> &[Self] {
        match self {
            Self::X => &[Self::Y, Self::Z],
            Self::Y => &[Self::Z, Self::X],
            Self::Z => &[Self::X, Self::Y],
        }
    }

    #[inline]
    pub const fn direction(&self, positive: bool) -> Direction {
        match self {
            Self::X => {
                if positive {
                    Direction::East
                } else {
                    Direction::West
                }
            }
            Self::Y => {
                if positive {
                    Direction::Up
                } else {
                    Direction::Down
                }
            }
            Self::Z => {
                if positive {
                    Direction::South
                } else {
                    Direction::North
                }
            }
        }
    }
}

impl Direction {
    pub const ALL: &'static [Self] = &[
        Self::West,
        Self::East,
        Self::Down,
        Self::Up,
        Self::North,
        Self::South,
    ];

    #[inline]
    pub const fn by_idx(idx: usize) -> Self {
        Self::ALL[idx]
    }

    #[inline]
    pub const fn idx(&self) -> usize {
        match self {
            Self::West => 0,
            Self::East => 1,
            Self::Down => 2,
            Self::Up => 3,
            Self::North => 4,
            Self::South => 5,
        }
    }

    #[inline]
    pub const fn plane(&self) -> &[Self] {
        match self {
            Self::West | Self::East => &[Self::Down, Self::Up, Self::North, Self::South],
            Self::Down | Self::Up => &[Self::North, Self::South, Self::West, Self::East],
            Self::North | Self::South => &[Self::West, Self::East, Self::Down, Self::Up],
        }
    }

    #[inline]
    pub const fn axis(&self) -> Axis {
        match self {
            Self::West | Self::East => Axis::X,
            Self::Down | Self::Up => Axis::Y,
            Self::North | Self::South => Axis::Z,
        }
    }

    #[inline]
    pub const fn positive(&self) -> bool {
        match self {
            Self::East | Self::Up | Self::South => true,
            Self::West | Self::Down | Self::North => false,
        }
    }

    #[inline]
    pub const fn opposite(&self) -> Self {
        match self {
            Self::West => Self::East,
            Self::East => Self::West,
            Self::Down => Self::Up,
            Self::Up => Self::Down,
            Self::North => Self::South,
            Self::South => Self::North,
        }
    }

    #[inline]
    pub fn vector<C: Coord3>(&self) -> C
    where
        <C as Coord>::Scalar: Copy + Clone + Signed,
    {
        let zero = C::Scalar::zero();
        let one = C::Scalar::one();
        match self {
            Self::West => C::new(-one, zero, zero),
            Self::East => C::new(one, zero, zero),
            Self::Down => C::new(zero, -one, zero),
            Self::Up => C::new(zero, one, zero),
            Self::North => C::new(zero, zero, -one),
            Self::South => C::new(zero, zero, one),
        }
    }
}

pub trait Coord:
    Copy
    + Clone
    + Pod
    + Zeroable
    + Default
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Self::Scalar, Output = Self>
    + MulAssign
    + Div<Self::Scalar, Output = Self>
    + DivAssign
    + Index<usize, Output = Self::Scalar>
    + IndexMut<usize>
{
    type Scalar: Copy
        + Clone
        + Num
        + PartialOrd
        + FromPrimitive
        + Add<Output = Self::Scalar>
        + AddAssign
        + Sub<Output = Self::Scalar>
        + SubAssign
        + Mul<Self::Scalar, Output = Self::Scalar>
        + MulAssign
        + Div<Self::Scalar, Output = Self::Scalar>
        + DivAssign;

    const DIM: usize;
}

pub trait Coord3: Coord {
    #[must_use]
    fn new(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self;

    fn get(&self, a: Axis) -> Self::Scalar;

    #[must_use]
    fn with(self, a: Axis, v: Self::Scalar) -> Self;

    #[must_use]
    fn shift(self, a: Axis, v: Self::Scalar) -> Self;

    #[must_use]
    fn normalize(self) -> Self;

    #[must_use]
    fn dot(self, other: Self) -> Self::Scalar;

    #[must_use]
    fn cross(self, other: Self) -> Self;
}

pub trait ICoord3: Coord3<Scalar: PrimInt> {
    #[must_use]
    fn step(self, d: Direction) -> Self;
}

pub trait SCoord3: Coord3<Scalar: Signed> + Neg<Output = Self> {}

macro_rules! impl_coord_for {
    ($($vec:ty : $scalar:ty, $dim:expr),* $(,)?) => {
        $(
            impl Coord for $vec {
                type Scalar = $scalar;
                const DIM: usize = $dim;
            }
        )*
    };
}

impl_coord_for!(Vec2: f32, 2, Vec3: f32, 3, Vec3A: f32, 3, Vec4: f32, 4);
impl_coord_for!(DVec2: f64, 2, DVec3: f64, 3, DVec4: f64, 4);
impl_coord_for!(I8Vec2: i8, 2, I8Vec3: i8, 3, I8Vec4: i8, 4);
impl_coord_for!(U8Vec2: u8, 2, U8Vec3: u8, 3, U8Vec4: u8, 4);
impl_coord_for!(I16Vec2: i16, 2, I16Vec3: i16, 3, I16Vec4: i16, 4);
impl_coord_for!(U16Vec2: u16, 2, U16Vec3: u16, 3, U16Vec4: u16, 4);
impl_coord_for!(IVec2: i32, 2, IVec3: i32, 3, IVec4: i32, 4);
impl_coord_for!(UVec2: u32, 2, UVec3: u32, 3, UVec4: u32, 4);
impl_coord_for!(I64Vec2: i64, 2, I64Vec3: i64, 3, I64Vec4: i64, 4);
impl_coord_for!(U64Vec2: u64, 2, U64Vec3: u64, 3, U64Vec4: u64, 4);
impl_coord_for!(ISizeVec2: isize, 2, ISizeVec3: isize, 3, ISizeVec4: isize, 4);
impl_coord_for!(USizeVec2: usize, 2, USizeVec3: usize, 3, USizeVec4: usize, 4);

macro_rules! impl_coord3_for {
    ($($vec:ty),* $(,)?) => {
        $(
            impl Coord3 for $vec {
	            #[inline]
	            fn new(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self {
                    Self::new(x, y, z)
                }

	            #[inline]
                fn get(&self, a: Axis) -> Self::Scalar {
                    match a {
                        Axis::X => self.x,
                        Axis::Y => self.y,
                        Axis::Z => self.z,
                    }
                }

	            #[inline]
                fn with(mut self, a: Axis, v: Self::Scalar) -> Self {
                    match a {
                        Axis::X => self.x = v,
                        Axis::Y => self.y = v,
                        Axis::Z => self.z = v,
                    }
                    self
                }

	            #[inline]
                fn shift(mut self, a: Axis, v: Self::Scalar) -> Self {
                    match a {
                        Axis::X => self.x = self.x + v,
                        Axis::Y => self.y = self.y + v,
                        Axis::Z => self.z = self.z + v,
                    }
                    self
                }

                #[inline]
                fn normalize(mut self) -> Self {
                    self.normalize()
                }

                #[inline]
                fn dot(self, other: Self) -> Self::Scalar {
                    self.dot(other)
                }

                #[inline]
                fn cross(self, other: Self) -> Self {
                    self.cross(other)
                }
            }
        )*
    };
}

impl_coord3_for!(
    Vec3, Vec3A, DVec3, I8Vec3, U8Vec3, I16Vec3, U16Vec3, IVec3, UVec3, I64Vec3, U64Vec3,
    ISizeVec3, USizeVec3,
);

macro_rules! impl_icoord3_for {
    ($($vec:ty),* $(,)?) => {
        $(
            impl ICoord3 for $vec {
	            #[inline]
                fn step(mut self, d: Direction) -> Self {
                    match d {
                        Direction::West => self.x -= 1,
                        Direction::East => self.x += 1,
                        Direction::North => self.z -= 1,
                        Direction::South => self.z += 1,
                        Direction::Down => self.y -= 1,
                        Direction::Up => self.y += 1,
                    }
                    self
                }
            }
        )*
    };
}

impl_icoord3_for!(
    I8Vec3, U8Vec3, I16Vec3, U16Vec3, IVec3, UVec3, I64Vec3, U64Vec3, ISizeVec3, USizeVec3,
);

impl<C: Coord3<Scalar: Signed> + Neg<Output = C>> SCoord3 for C {}
