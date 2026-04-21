use crate::util::collection::{SparseSet, SparseSetIter};
use crate::util::Id;
use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::collections::{hash_map, HashMap};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::{alloc, fmt, mem, ptr};

pub struct ErasedBox {
    ptr: *mut u8,
    layout: Layout,
    drop: Option<fn(*mut u8)>,
}

pub struct ErasedVec {
    data: UnsafeCell<Vec<u8>>,
    layout: Layout,
    len: *mut usize,
    drop: Option<fn(*mut u8)>,
}

pub struct ErasedDenseMap {
    ids: SparseSet,
    items: ErasedVec,
}

pub struct ErasedHashMap {
    ids: HashMap<Id, Id>,
    items: ErasedVec,
}

pub struct ErasedDenseMapIter<'a, T> {
    id_iter: SparseSetIter<'a>,
    items: &'a ErasedVec,
    _marker: PhantomData<T>,
}

pub struct ErasedDenseMapIterMut<'a, T> {
    id_iter: SparseSetIter<'a>,
    items: &'a ErasedVec,
    _marker: PhantomData<T>,
}

pub struct ErasedHashMapIter<'a, T> {
    id_iter: hash_map::Iter<'a, Id, Id>,
    items: &'a ErasedVec,
    _marker: PhantomData<T>,
}

pub struct ErasedHashMapIterMut<'a, T> {
    id_iter: hash_map::Iter<'a, Id, Id>,
    items: &'a ErasedVec,
    _marker: PhantomData<T>,
}

unsafe impl Sync for ErasedBox {}

unsafe impl Send for ErasedBox {}

impl Drop for ErasedBox {
    #[inline]
    fn drop(&mut self) {
        if let Some(drop) = self.drop {
            drop(self.ptr);
            unsafe {
                alloc::dealloc(self.ptr, self.layout);
            }
        }
    }
}

impl ErasedBox {
    #[inline]
    pub fn new<T: 'static>(value: T) -> Self {
        let layout = Layout::new::<T>();
        let ptr = unsafe { alloc::alloc(layout) } as *mut T;
        assert!(!ptr.is_null());
        unsafe {
            ptr.write(value);
        }

        Self {
            ptr: ptr as *mut u8,
            layout,
            drop: if mem::needs_drop::<T>() {
                Some(|p| unsafe {
                    ptr::drop_in_place(p as *mut T);
                })
            } else {
                None
            },
        }
    }

    #[inline]
    pub fn cast<T: 'static>(&self) -> &T {
        assert_eq!(self.layout, Layout::new::<T>());
        unsafe { &*(self.ptr as *const T) }
    }

    #[inline]
    pub fn cast_mut<T: 'static>(&self) -> &mut T {
        assert_eq!(self.layout, Layout::new::<T>());
        unsafe { &mut *(self.ptr as *mut T) }
    }

    #[inline]
    pub fn forget(&mut self) {
        self.drop = None;
    }
}

impl Drop for ErasedVec {
    #[inline]
    fn drop(&mut self) {
        self.clear();
        unsafe {
            ptr::drop_in_place(self.len);
        }
    }
}

impl ErasedVec {
    #[inline]
    pub fn new<T>() -> Self {
        let len = unsafe { alloc::alloc(Layout::new::<usize>()) } as *mut usize;
        assert!(!len.is_null());
        unsafe {
            len.write(0);
        }
        
        Self {
            data: UnsafeCell::new(Vec::new()),
            layout: Layout::new::<T>(),
            len,
            drop: if mem::needs_drop::<T>() {
                Some(|p| unsafe {
                    ptr::drop_in_place(p as *mut T);
                })
            } else {
                None
            },
        }
    }
    
    #[inline]
    const fn len(&self) -> usize {
        unsafe {
            *self.len
        }
    }

    #[inline]
    fn reserve(&self, additional: usize) {
        unsafe {
            (&mut *self.data.get()).reserve(additional * self.layout.size());
        }
    }

    #[inline]
    fn set_len(&self, new_len: usize) {
        unsafe {
            assert!(new_len * self.layout.size() <= (&*self.data.get()).capacity());
            (&mut *self.data.get()).set_len(new_len * self.layout.size());
            *self.len = new_len;
        }
    }

    #[inline]
    const fn get_ptr<T>(&self, idx: usize) -> *const T {
        let offset = idx * self.layout.size();
        unsafe { (*self.data.get()).as_ptr().add(offset) as *const T }
    }

    #[inline]
    const fn get_mut_ptr<T>(&self, idx: usize) -> *mut T {
        let offset = idx * self.layout.size();
        unsafe { (*self.data.get()).as_mut_ptr().add(offset) as *mut T }
    }

    #[inline]
    pub fn get<T>(&self, idx: usize) -> &T {
        assert_eq!(self.layout, Layout::new::<T>());
        unsafe { &*self.get_ptr(idx) }
    }

    #[inline]
    pub fn get_mut<T>(&self, idx: usize) -> &mut T {
        assert_eq!(self.layout, Layout::new::<T>());
        unsafe { &mut *self.get_mut_ptr(idx) }
    }

    #[inline]
    pub fn insert<T>(&mut self, idx: usize, item: T) {
        assert_eq!(self.layout, Layout::new::<T>());
        assert!(idx <= self.len());
        self.reserve(1);
        unsafe {
            let ptr = self.get_mut_ptr::<T>(idx);
            if idx < self.len() {
                ptr::copy(ptr, ptr.add(1), self.len() - idx);
            }
            ptr.write(item);
            self.set_len(self.len() + 1);
        }
    }

    #[inline]
    pub fn push<T>(&mut self, item: T) {
        assert_eq!(self.layout, Layout::new::<T>());
        self.reserve(1);
        unsafe {
            let ptr = self.get_mut_ptr::<T>(self.len());
            ptr.write(item);
            self.set_len(self.len() + 1);
        }
    }

    #[inline]
    pub fn push_erased(&mut self, mut item: ErasedBox) {
        assert_eq!(self.layout, item.layout);
        self.reserve(1);
        let size = self.layout.size();
        unsafe {
            let dst = self.get_mut_ptr::<u8>(self.len());
            ptr::copy_nonoverlapping(item.ptr, dst, size);
            item.forget();
        }
        self.set_len(self.len() + 1);
    }

    #[inline]
    pub fn swap_remove<T>(&mut self, idx: usize) -> T {
        assert_eq!(self.layout, Layout::new::<T>());
        assert!(idx < self.len());
        unsafe {
            let ptr = self.get_mut_ptr::<T>(idx);
            let value = ptr.read();
            ptr::copy(self.get_mut_ptr(self.len() - 1), ptr, 1);
            self.set_len(self.len() - 1);
            value
        }
    }

    #[inline]
    pub fn swap_remove_and_drop(&mut self, idx: usize) {
        assert!(idx < self.len());
        let last = self.len() - 1;
        let size = self.layout.size();
        unsafe {
            let dst = self.get_mut_ptr(idx);
            let src = self.get_ptr(last);
            if let Some(drop) = self.drop {
                drop(dst);
            }
            if idx != last {
                ptr::copy_nonoverlapping(src, dst, size);
            }
        }
        self.set_len(self.len() - 1);
    }

    #[inline]
    pub fn remove<T>(&mut self, idx: usize) -> T {
        assert_eq!(self.layout, Layout::new::<T>());
        assert!(idx < self.len());
        unsafe {
            let ptr = self.get_mut_ptr::<T>(idx);
            let value = ptr.read();
            ptr::copy(ptr.add(1), ptr, self.len() - idx - 1);
            self.set_len(self.len() - 1);
            value
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        if let Some(drop) = self.drop {
            for i in 0..self.len() {
                let ptr = self.get_mut_ptr(i);
                drop(ptr);
            }
        }
        self.set_len(0);
        unsafe { (&mut *self.data.get()).clear(); }
    }

    pub fn fmt<T: Debug>(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "BlobVec [ ")?;
        let ptr = self.get_ptr::<T>(0);
        for i in 0..self.len() {
            let item = unsafe { &*ptr.add(i) };
            write!(f, "{:?}, ", item)?;
        }
        write!(f, "]")
    }
}

impl ErasedDenseMap {
    #[inline]
    pub fn new<T>() -> Self {
        Self {
            ids: SparseSet::new(),
            items: ErasedVec::new::<T>(),
        }
    }

    #[inline]
    pub fn contains(&self, id: Id) -> bool {
        self.ids.contains(id)
    }

    #[inline]
    pub fn get<T>(&self, id: Id) -> Option<&T> {
        match self.ids.find(id) {
            Some(_) => Some(&self.items.get(id as usize)),
            None => None,
        }
    }

    #[inline]
    pub fn get_mut<T>(&self, id: Id) -> Option<&mut T> {
        match self.ids.find(id) {
            Some(idx) => Some(self.items.get_mut(idx as usize)),
            None => None,
        }
    }
    
    pub fn iter<T>(&self) -> ErasedDenseMapIter<'_, T> {
        ErasedDenseMapIter {
            id_iter: self.ids.iter(),
            items: &self.items,
            _marker: PhantomData,
        }
    }
    
    pub fn iter_mut<T>(&self) -> ErasedDenseMapIterMut<'_, T> {
        ErasedDenseMapIterMut {
            id_iter: self.ids.iter(),
            items: &self.items,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn insert<T>(&mut self, id: Id, item: T) -> Option<T> {
        match self.ids.find(id) {
            Some(idx) => Some(mem::replace(&mut self.items.get_mut(idx as usize), item)),
            None => {
                let idx = self.items.len();
                self.ids.put(id, idx as Id);
                self.items.push(item);
                None
            }
        }
    }

    #[inline]
    pub fn insert_erased(&mut self, id: Id, mut item: ErasedBox) {
        match self.ids.find(id) {
            Some(idx) => {
                unsafe {
                    ptr::copy(
                        item.ptr,
                        self.items.get_mut_ptr(idx as usize),
                        self.items.layout.size(),
                    );
                }
                item.forget();
            }
            None => {
                let idx = self.items.len();
                self.ids.put(id, idx as Id);
                self.items.push_erased(item);
            }
        }
    }

    #[inline]
    pub fn remove<T>(&mut self, id: Id) -> Option<T> {
        match self.ids.find(id) {
            Some(idx) => {
                self.ids.kick(id, idx);
                Some(self.items.swap_remove(idx as usize))
            }
            None => None,
        }
    }

    #[inline]
    pub fn remove_and_drop(&mut self, id: Id) {
        if let Some(idx) = self.ids.find(id) {
            self.ids.kick(id, idx);
            self.items.swap_remove_and_drop(idx as usize);
        }
    }
}

impl ErasedHashMap {
    #[inline]
    pub fn new<T>() -> Self {
        Self {
            ids: HashMap::new(),
            items: ErasedVec::new::<T>(),
        }
    }

    #[inline]
    pub fn contains(&self, id: Id) -> bool {
        self.ids.contains_key(&id)
    }

    #[inline]
    pub fn get<T>(&self, id: Id) -> Option<&T> {
        match self.ids.get(&id) {
            Some(&idx) => Some(&self.items.get(idx as usize)),
            None => None,
        }
    }

    #[inline]
    pub fn get_mut<T>(&self, id: Id) -> Option<&mut T> {
        match self.ids.get(&id) {
            Some(&idx) => Some(self.items.get_mut(idx as usize)),
            None => None,
        }
    }
    
    pub fn iter<T>(&self) -> ErasedHashMapIter<'_, T> {
        ErasedHashMapIter {
            id_iter: self.ids.iter(),
            items: &self.items,
            _marker: PhantomData,
        }
    }
    
    pub fn iter_mut<T>(&self) -> ErasedHashMapIterMut<'_, T> {
        ErasedHashMapIterMut {
            id_iter: self.ids.iter(),
            items: &self.items,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn insert<T>(&mut self, id: Id, item: T) -> Option<T> {
        match self.ids.get(&id) {
            Some(&idx) => Some(mem::replace(&mut self.items.get_mut(idx as usize), item)),
            None => {
                let idx = self.items.len();
                self.ids.insert(id, idx as Id);
                self.items.push(item);
                None
            }
        }
    }

    #[inline]
    pub fn insert_erased(&mut self, id: Id, mut item: ErasedBox) {
        match self.ids.get(&id) {
            Some(&idx) => {
                unsafe {
                    ptr::copy(
                        item.ptr,
                        self.items.get_mut_ptr(idx as usize),
                        self.items.layout.size(),
                    );
                }
                item.forget();
            }
            None => {
                let idx = self.items.len();
                self.ids.insert(id, idx as Id);
                self.items.push_erased(item);
            }
        }
    }

    #[inline]
    pub fn remove<T>(&mut self, id: Id) -> Option<T> {
        match self.ids.get(&id) {
            Some(&idx) => {
                self.ids.remove(&id);
                if let Some((&id, _idx)) = self
                    .ids
                    .iter()
                    .find(|(_id, idx)| **idx as usize + 1 == self.items.len())
                {
                    self.ids.insert(id, idx);
                }
                Some(self.items.swap_remove(idx as usize))
            }
            None => None,
        }
    }

    #[inline]
    pub fn remove_and_drop(&mut self, id: Id) {
        if let Some(&idx) = self.ids.get(&id) {
            self.ids.remove(&id);
            if let Some((&id, _idx)) = self
                .ids
                .iter()
                .find(|(_id, idx)| **idx as usize + 1 == self.items.len())
            {
                self.ids.insert(id, idx);
            }
            self.items.swap_remove_and_drop(idx as usize);
        }
    }
}

impl<'a, T: 'a> Iterator for ErasedDenseMapIter<'a, T> {
    type Item = (Id, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.id_iter.pos;
        match self.id_iter.next() {
            Some(id) => Some((id, self.items.get(pos))),
            None => None,
        }
    }
}

impl<'a, T: 'a> Iterator for ErasedDenseMapIterMut<'a, T> {
    type Item = (Id, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.id_iter.pos;
        match self.id_iter.next() {
            Some(id) => Some((id, self.items.get_mut(pos))),
            None => None,
        }
    }
}

impl<'a, T: 'a> Iterator for ErasedHashMapIter<'a, T> {
    type Item = (Id, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.id_iter.next() {
            Some((&id, &idx)) => Some((id, self.items.get(idx as usize))),
            None => None,
        }
    }
}

impl<'a, T: 'a> Iterator for ErasedHashMapIterMut<'a, T> {
    type Item = (Id, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.id_iter.next() {
            Some((&id, &idx)) => Some((id, self.items.get_mut(idx as usize))),
            None => None,
        }
    }
}
