use crate::util::Id;
use anyhow::Result;
use bimap::BiMap;
use glam::{U8Vec3, USizeVec3};
use log::error;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::BufReader;
use std::{fs, mem};

/// A set of [`Id`]s kept as a dense list plus a sparse lookup table.
///
/// `dense` holds the ids and `sparse` maps an id to its position in `dense`.
/// Removing an id swaps the last entry into the freed slot, so iteration order
/// is unstable.
#[derive(Default)]
pub struct SparseSet {
    dense: Vec<Id>,
    sparse: Vec<Id>,
}

/// A map from [`Id`] to `T` whose values are stored in a dense vector.
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

/// Mutable iterator over a [`DenseMap`].
///
/// It holds a raw pointer because the items are lent out for a lifetime that
/// outlives the borrow of the map, and because a shared borrow of the map is
/// enough to hand out `&mut T`.
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
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The position of `id` in `dense`, or `None` when the id is not in the set.
    #[inline]
    pub fn find(&self, id: Id) -> Option<Id> {
        if let Some(&idx) = self.sparse.get(id as usize)
            && let Some(&id1) = self.dense.get(idx as usize)
            && id1 == id
        {
            Some(idx)
        } else {
            None
        }
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

    /// Appends `id` at `idx`, which must be the current length of the set.
    #[inline]
    pub fn put(&mut self, id: Id, idx: Id) {
        if id as usize >= self.sparse.len() {
            self.sparse.resize(id as usize + 1, 0);
        }

        self.dense.push(id);
        self.sparse[id as usize] = idx;
    }

    /// Removes the entry at `idx` by moving the last entry into the freed slot.
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
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

/// A box of `T` sampled on a regular grid, indexed by positions in `0..size`.
///
/// Values are stored with `z` varying fastest, then `y`, then `x`, so a row
/// along `z` is contiguous.
#[derive(Clone)]
pub struct Volume<T> {
    pub size: U8Vec3,
    pub vec: Vec<T>,
}

impl<T: Clone + Default> Volume<T> {
    #[inline]
    pub fn new(size: U8Vec3) -> Self {
        Self {
            size,
            vec: vec![T::default(); size.as_usizevec3().element_product()],
        }
    }

    /// Copies the `size` sized sub-volume at `min` into a new volume.
    #[inline]
    pub fn part(&self, min: U8Vec3, size: U8Vec3) -> Volume<T> {
        let max = min + size;
        debug_assert!(max.x <= self.size.x && max.y <= self.size.y && max.z <= self.size.z);

        let [_, h, d] = self.size.as_usizevec3().to_array();
        let [sw, sh, sd] = size.as_usizevec3().to_array();
        let [dx, dy, dz] = min.as_usizevec3().to_array();

        let mut part = Volume::<T>::new(size);

        for x in 0..sw {
            for y in 0..sh {
                let dst = x * sh * sd + y * sd;
                let src = (x + dx) * h * d + (y + dy) * d + dz;
                part.vec[dst..dst + sd].clone_from_slice(&self.vec[src..src + sd]);
            }
        }

        part
    }
}

impl<T: Clone> Volume<T> {
    #[inline]
    pub fn splat(size: U8Vec3, value: T) -> Self {
        Self {
            size,
            vec: vec![value; size.as_usizevec3().element_product()],
        }
    }

    #[inline]
    pub fn fill(&mut self, min: U8Vec3, max: U8Vec3, value: T) {
        debug_assert!(min.x < self.size.x && min.y < self.size.y && min.z < self.size.z);
        debug_assert!(max.x <= self.size.x && max.y <= self.size.y && max.z <= self.size.z);

        let [_, h, d] = self.size.as_usizevec3().to_array();
        let min = USizeVec3::from(min);
        let max = USizeVec3::from(max);

        for x in min.x..max.x {
            for y in min.y..max.y {
                let offset = x * h * d + y * d;
                self.vec[offset + min.z..offset + max.z].fill(value.clone());
            }
        }
    }

    /// Writes `part` into this volume at `min`.
    #[inline]
    pub fn fit(&mut self, min: U8Vec3, part: &Volume<T>) {
        let max = min + part.size;
        debug_assert!(max.x <= self.size.x && max.y <= self.size.y && max.z <= self.size.z);

        let [_, h, d] = self.size.as_usizevec3().to_array();
        let [sw, sh, sd] = part.size.as_usizevec3().to_array();
        let [dx, dy, dz] = min.as_usizevec3().to_array();

        for x in 0..sw {
            for y in 0..sh {
                let src = x * sh * sd + y * sd;
                let dst = (x + dx) * h * d + (y + dy) * d + dz;
                self.vec[dst..dst + sd].clone_from_slice(&part.vec[src..src + sd]);
            }
        }
    }
}

impl<T> Volume<T> {
    /// Builds a volume of `size` by calling `pos_to_item` for every position.
    pub fn from_fn<F: Fn(U8Vec3) -> T>(size: U8Vec3, pos_to_item: F) -> Self {
        let [w, h, d] = size.as_usizevec3().to_array();
        let total = w * h * d;
        let vec = (0..total)
            .map(|idx| {
                let pos = U8Vec3::new(
                    (idx / (h * d)) as u8,
                    ((idx / d) % h) as u8,
                    (idx % d) as u8,
                );
                pos_to_item(pos)
            })
            .collect();

        Self { size, vec }
    }

    /// The index of `pos` in the backing vector.
    #[inline]
    pub fn idx_of_pos(&self, pos: U8Vec3) -> usize {
        debug_assert!(pos.x < self.size.x && pos.y < self.size.y && pos.z < self.size.z);
        let h = self.size.y as usize;
        let d = self.size.z as usize;
        pos.x as usize * h * d + pos.y as usize * d + pos.z as usize
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
    pub fn set(&mut self, pos: U8Vec3, value: T) -> T {
        let idx = self.idx_of_pos(pos);
        mem::replace(&mut self.vec[idx], value)
    }

    /// Iterates the `z` oriented rows of the range `min..max`.
    #[inline]
    pub fn rows(&self, min: U8Vec3, max: U8Vec3) -> impl Iterator<Item = &[T]> + '_ {
        debug_assert!(max.x <= self.size.x && max.y <= self.size.y && max.z <= self.size.z);

        let [_, h, d] = self.size.as_usizevec3().to_array();
        let [sw, sh, sd] = (max - min).as_usizevec3().to_array();
        let [ox, oy, oz] = min.as_usizevec3().to_array();
        let data: &[T] = &self.vec;

        (0..sw).flat_map(move |x| {
            (0..sh).map(move |y| {
                let start = (ox + x) * h * d + (oy + y) * d + oz;
                &data[start..start + sd]
            })
        })
    }

    /// Mutably iterates the `z` oriented rows of the range `min..max`.
    ///
    /// The rows are built from a raw pointer because they all alias the same
    /// vector; the ranges are disjoint by construction, so no row is handed out
    /// twice.
    #[inline]
    pub fn rows_mut(&mut self, min: U8Vec3, max: U8Vec3) -> impl Iterator<Item = &mut [T]> + '_ {
        debug_assert!(max.x <= self.size.x && max.y <= self.size.y && max.z <= self.size.z);

        let [_, h, d] = self.size.as_usizevec3().to_array();
        let [sw, sh, sd] = (max - min).as_usizevec3().to_array();
        let [ox, oy, oz] = min.as_usizevec3().to_array();
        let base = self.vec.as_mut_ptr();

        (0..sw).flat_map(move |x| {
            (0..sh).map(move |y| {
                let start = (ox + x) * h * d + (oy + y) * d + oz;
                unsafe { std::slice::from_raw_parts_mut(base.add(start), sd) }
            })
        })
    }

    pub fn iter(&self, min: U8Vec3, max: U8Vec3) -> impl Iterator<Item = &'_ T> + '_ {
        self.rows(min, max).flatten()
    }

    pub fn iter_mut<'a>(
        &mut self,
        min: U8Vec3,
        max: U8Vec3,
    ) -> impl Iterator<Item = &'_ mut T> + '_ {
        self.rows_mut(min, max).flatten()
    }
}

/// A collection of entries indexed by [`Id`], alongside each entry's name.
pub struct Registry<T> {
    pub entries: Vec<T>,
    pub names: BiMap<Id, String>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self {
            entries: Vec::default(),
            names: BiMap::default(),
        }
    }
}

impl<T: DeserializeOwned> Registry<T> {
    /// Loads every file of `path`, numbering the entries in the order the
    /// directory lists them.
    pub fn load_rons_from(path: &str) -> Result<Self> {
        let mut new = Self::new();

        let mut id = 0;
        for entry in fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_file() {
                let name = path
                    .file_stem()
                    .expect("path should have a file name")
                    .to_string_lossy()
                    .into_owned();
                let file = File::open(path)?;
                let item = ron::de::from_reader(BufReader::new(file))?;

                new.register(id, name, item);
                id += 1;
            }
        }

        Ok(new)
    }
}

impl<T> Registry<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `item` under `name`.
    ///
    /// Ids must be registered in ascending order starting at zero, because the
    /// id of an entry is also its position in [`Self::entries`].
    pub fn register<S>(&mut self, id: Id, name: S, item: T)
    where
        String: From<S>,
    {
        if id == self.entries.len() as u32 {
            self.entries.push(item);
            self.names.insert(id, String::from(name));
        } else {
            error!(
                "could not register entry with id {}: entries should be registered in ascending order",
                id
            );
        }
    }

    /// The id registered under `name`.
    #[inline]
    pub fn id_of(&self, name: &str) -> Id {
        *self
            .names
            .get_by_right(name)
            .unwrap_or_else(|| panic!("entry with name {} not found", name))
    }

    /// The name registered for `id`.
    #[inline]
    pub fn name_of(&self, id: Id) -> &str {
        self.names
            .get_by_left(&id)
            .unwrap_or_else(|| panic!("entry with id {} not found", id))
            .as_str()
    }

    /// The entry registered for `id`.
    #[inline]
    pub fn get(&self, id: Id) -> &T {
        self.entries
            .get(id as usize)
            .unwrap_or_else(|| panic!("entry with id {} not found", id))
    }
}
