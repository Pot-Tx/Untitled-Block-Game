pub mod bounding;
pub mod collection;
pub mod coord;
pub mod erasure;
pub mod math;
pub mod transform;

use log::error;
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::OnceLock;

pub type Id = u32;

#[derive(Default)]
pub struct IdManager {
    recycled: VecDeque<Id>,
    next: Id,
}

impl IdManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&mut self) -> Id {
        let id = match self.recycled.pop_front() {
            Some(id) => id,
            None => {
                self.next += 1;
                self.next - 1
            }
        };
        id
    }

    pub fn recycle(&mut self, id: Id) {
        self.recycled.push_back(id);
    }
}

pub struct IdAllocator {
    free: Vec<IdBlock>,
}

struct IdBlock {
    base: Id,
    len: u32,
}

impl Default for IdAllocator {
    fn default() -> Self {
        Self {
            free: vec![IdBlock {
                base: 0,
                len: u32::MAX,
            }],
        }
    }
}

impl IdAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&mut self, len: u32) -> Id {
        for i in 0..self.free.len() {
            let block = &mut self.free[i];
            let base = block.base;
            if block.len > len {
                block.base += len;
                block.len -= len;
                return base;
            } else if block.len == len {
                self.free.remove(i);
                return base;
            }
        }
        unreachable!()
    }

    pub fn free(&mut self, base: Id, len: u32) {
        for i in 0..self.free.len() {
            let block = &mut self.free[i];
            if block.base > base {
                if base + len == block.base {
                    block.base -= len;
                    block.len += len;
                } else if i > 0 {
                    let block = &mut self.free[i - 1];
                    if block.base + block.len == base {
                        block.len += len;
                    }
                } else {
                    self.free.insert(i, IdBlock { base, len });
                }
                break;
            }
        }
    }
}

pub struct OnceInit<T> {
    inner: OnceLock<T>,
}

impl<T> Deref for OnceInit<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
            .get()
            .expect("OnceInit should be initialized before dereferenced")
    }
}

impl<T> OnceInit<T> {
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    pub fn ready(&self) -> bool {
        self.inner.get().is_some()
    }

    pub fn init(&self, value: T) {
        if self.inner.set(value).is_err() {
            error!("OnceInit already initialized");
        }
    }
}

pub struct SwapPair<T> {
    left: Option<T>,
    right: Option<T>,
    on_right: bool,
    timer: u8,
}

impl<T> Default for SwapPair<T> {
    fn default() -> Self {
        Self {
            left: None,
            right: None,
            on_right: false,
            timer: u8::MAX,
        }
    }
}

impl<T: Clone> Clone for SwapPair<T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            on_right: self.on_right,
            timer: self.timer,
        }
    }
}

impl<T> SwapPair<T> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn set(&mut self, item: T, time: u8) {
        if self.on_right {
            self.left = Some(item);
        } else {
            self.right = Some(item);
        }
        self.timer = time;

        if time == 0 {
            self.on_right = !self.on_right;
            if self.on_right {
                self.left = None;
            } else {
                self.right = None;
            }
        }
    }

    #[inline]
    pub fn update(&mut self) -> bool {
        if self.timer > 0 {
            self.timer -= 1;
            if self.timer == 0 {
                self.on_right = !self.on_right;
                if self.on_right {
                    self.left = None;
                } else {
                    self.right = None;
                }

                return true;
            }
            false
        } else {
            true
        }
    }

    #[inline]
    pub fn get(&self) -> Option<&T> {
        if self.on_right {
            self.right.as_ref()
        } else {
            self.left.as_ref()
        }
    }
}
