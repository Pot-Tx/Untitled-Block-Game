use crate::util::Id;
use glam::{U8Vec3, USizeVec3};
use log::error;
use std::mem;

#[derive(Default)]
pub struct SparseSet {
    dense: Vec<Id>,
    sparse: Vec<Id>,
}

pub struct DenseMap<T> {
    ids: SparseSet,
    items: Vec<T>,
}

pub struct SparseSetIter<'a> {
    dense: &'a [Id],
    pub pos: usize,
}

pub struct DenseMapIter<'a, T> {
    id_iter: SparseSetIter<'a>,
    items: &'a [T],
}

pub struct DenseMapIterMut<'a, T> {
    id_iter: SparseSetIter<'a>,
    items: *mut T,
}

impl SparseSet {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn find(&self, id: Id) -> Option<Id> {
        if let Some(&idx) = self.sparse.get(id as usize) {
            if let Some(&id1) = self.dense.get(idx as usize)
                && id1 == id
            {
                return Some(idx);
            }
        }
        None
    }

    #[inline]
    pub fn contains(&self, id: Id) -> bool {
        self.find(id).is_some()
    }

    pub fn iter(&'_ self) -> SparseSetIter<'_> {
        SparseSetIter {
            dense: &self.dense,
            pos: 0,
        }
    }

    #[inline]
    pub fn put(&mut self, id: Id, idx: Id) {
        if id as usize >= self.sparse.len() {
            self.sparse.resize(id as usize + 1, 0);
        }

        self.dense.push(id);
        self.sparse[id as usize] = idx;
    }

    #[inline]
    pub fn kick(&mut self, _id: Id, idx: Id) {
        let last_idx = self.dense.len() - 1;
        let last_id = self.dense[last_idx];

        self.sparse[last_id as usize] = idx;
        self.dense.swap_remove(idx as usize);
    }

    #[inline]
    pub fn insert(&mut self, id: Id) -> bool {
        if self.contains(id) {
            false
        } else {
            let idx = self.dense.len();
            self.put(id, idx as Id);
            true
        }
    }

    #[inline]
    pub fn remove(&mut self, id: Id) -> bool {
        match self.find(id) {
            Some(idx) => {
                self.kick(id, idx);
                true
            }
            None => false,
        }
    }
}

impl<T> Default for DenseMap<T> {
    fn default() -> Self {
        Self {
            ids: SparseSet::new(),
            items: Vec::new(),
        }
    }
}

impl<T> DenseMap<T> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn contains(&self, id: Id) -> bool {
        self.ids.contains(id)
    }

    #[inline]
    pub fn get(&self, id: Id) -> Option<&T> {
        match self.ids.find(id) {
            Some(idx) => Some(&self.items[idx as usize]),
            None => None,
        }
    }

    #[inline]
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        match self.ids.find(id) {
            Some(idx) => Some(&mut self.items[idx as usize]),
            None => None,
        }
    }

    pub fn iter(&'_ self) -> DenseMapIter<'_, T> {
        DenseMapIter {
            id_iter: SparseSetIter {
                dense: &self.ids.dense,
                pos: 0,
            },
            items: &self.items,
        }
    }

    pub fn iter_mut(&'_ mut self) -> DenseMapIterMut<'_, T> {
        DenseMapIterMut {
            id_iter: SparseSetIter {
                dense: &self.ids.dense,
                pos: 0,
            },
            items: self.items.as_mut_ptr(),
        }
    }

    #[inline]
    pub fn insert(&mut self, id: Id, item: T) -> Option<T> {
        match self.ids.find(id) {
            Some(idx) => Some(mem::replace(&mut self.items[idx as usize], item)),
            None => {
                let idx = self.items.len();
                self.ids.put(id, idx as Id);
                self.items.push(item);
                None
            }
        }
    }

    #[inline]
    pub fn remove(&mut self, id: Id) -> Option<T> {
        match self.ids.find(id) {
            Some(idx) => {
                self.ids.kick(id, idx);
                Some(self.items.swap_remove(idx as usize))
            }
            None => None,
        }
    }
}

impl<'a> Iterator for SparseSetIter<'a> {
    type Item = Id;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.dense.len() {
            None
        } else {
            let id = self.dense[self.pos];
            self.pos += 1;
            Some(id)
        }
    }
}

impl<'a, T> Iterator for DenseMapIter<'a, T> {
    type Item = (Id, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.id_iter.pos;
        match self.id_iter.next() {
            Some(id) => Some((id, &self.items[pos])),
            None => None,
        }
    }
}

impl<'a, T: 'a> Iterator for DenseMapIterMut<'a, T> {
    type Item = (Id, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.id_iter.pos;
        match self.id_iter.next() {
            Some(id) => {
                let item = unsafe { &mut *self.items.add(pos) };
                Some((id, item))
            }
            None => None,
        }
    }
}

#[derive(Clone)]
pub struct CubicVec<T> {
    pub side: u8,
    pub vec: Vec<T>,
}

impl<T: Clone + Default> CubicVec<T> {
    #[inline]
    pub fn new(side: u8) -> Self {
        Self {
            side,
            vec: vec![T::default(); (side as usize) * (side as usize) * (side as usize)],
        }
    }

    #[inline]
    pub fn part(&self, min: U8Vec3, side: u8) -> CubicVec<T> {
        let max = min + side;
        assert!(max.x <= self.side && max.y <= self.side && max.z <= self.side);

        let n = self.side as usize;
        let m = side as usize;
        let dz = min.z as usize;
        let dy = min.y as usize;
        let mut part = CubicVec::<T>::new(side);

        let x = min.x as usize;
        for z in 0..m {
            for y in 0..m {
                let dst = z * m * m + y * m;
                let src = (z + dz) * n * n + (y + dy) * n + x;
                part.vec[dst..dst + m].clone_from_slice(&self.vec[src..src + m]);
            }
        }

        part
    }
}

impl<T: Clone> CubicVec<T> {
    #[inline]
    pub fn splat(side: u8, value: T) -> Self {
        Self {
            side,
            vec: vec![value; (side as usize) * (side as usize) * (side as usize)],
        }
    }

    #[inline]
    pub fn fill(&mut self, min: U8Vec3, max: U8Vec3, value: T) {
        assert!(min.x < self.side && min.y < self.side && min.z < self.side);
        assert!(max.x <= self.side && max.y <= self.side && max.z <= self.side);

        let n = self.side as usize;
        let min = USizeVec3::from(min);
        let max = USizeVec3::from(max);

        for z in min.z..max.z {
            for y in min.y..max.y {
                let offset = z * n * n + y * n;
                self.vec[offset + min.x..offset + max.x].fill(value.clone());
            }
        }
    }

    #[inline]
    pub fn fit(&mut self, min: U8Vec3, part: &CubicVec<T>) {
        let max = min + part.side;
        assert!(max.x <= self.side && max.y <= self.side && max.z <= self.side);

        let n = self.side as usize;
        let m = part.side as usize;
        let dz = min.z as usize;
        let dy = min.y as usize;

        let x = min.x as usize;
        for z in 0..m {
            for y in 0..m {
                let src = z * m * m + y * m;
                let dst = (z + dz) * n * n + (y + dy) * n + x;
                self.vec[dst..dst + m].clone_from_slice(&part.vec[src..src + m]);
            }
        }
    }
}

impl<T> CubicVec<T> {
    pub fn from_fn<F: Fn(U8Vec3) -> T>(side: u8, pos_to_item: F) -> Self {
        let n = side as usize;
        let vec = (0..n * n * n)
            .map(|idx| {
                let pos = U8Vec3::new(
                    (idx % n) as u8,
                    ((idx / n) % n) as u8,
                    (idx / (n * n)) as u8,
                );
                pos_to_item(pos)
            })
            .collect();

        Self { side, vec }
    }

    #[inline]
    pub const fn idx_of_pos(&self, pos: U8Vec3) -> usize {
        assert!(pos.x < self.side && pos.y < self.side && pos.z < self.side);

        let n = self.side as usize;
        pos.x as usize + pos.y as usize * n + pos.z as usize * n * n
    }

    #[inline]
    pub fn get(&self, pos: U8Vec3) -> &T {
        &self.vec[self.idx_of_pos(pos)]
    }

    #[inline]
    pub fn get_mut(&mut self, pos: U8Vec3) -> &mut T {
        let idx = self.idx_of_pos(pos);
        &mut self.vec[idx]
    }

    #[inline]
    pub fn set(&mut self, pos: U8Vec3, value: T) {
        let idx = self.idx_of_pos(pos);
        self.vec[idx] = value;
    }
}

pub struct Registry<T> {
    pub items: Vec<T>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self {
            items: Vec::default(),
        }
    }
}

impl<T> Registry<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, id: Id, item: T) {
        if id == self.items.len() as u32 {
            self.items.push(item);
        } else {
            error!(
                "Cannot register item with id {}. Make sure to register in order!",
                id
            );
        }
    }

    #[inline]
    pub fn get(&self, id: Id) -> &T {
        self.items
            .get(id as usize)
            .expect(&format!("Item with id {} hasn't been registered", id))
    }

    #[inline]
    pub fn get_mut(&mut self, id: Id) -> &mut T {
        self.items
            .get_mut(id as usize)
            .expect(&format!("Item with id {} hasn't been registered", id))
    }
}
