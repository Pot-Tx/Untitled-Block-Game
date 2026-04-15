use crate::util::coord::SCoord3;
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
			radius: radius.clone(),
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
