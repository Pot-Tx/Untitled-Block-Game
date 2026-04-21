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
        self.inner.get().expect("OnceInit hasn't been initialized")
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
    counter: u8,
}

impl<T> SwapPair<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            left: None,
            right: None,
            on_right: false,
            counter: u8::MAX,
        }
    }

    #[inline]
    pub fn set(&mut self, item: Option<T>, time: u8) {
        if self.on_right {
            self.left = item;
        } else {
            self.right = item;
        }
        self.counter = time;
    }

    #[inline]
    pub fn update(&mut self) -> bool {
        if self.counter > 0 {
            self.counter -= 1;
            if self.counter == 0 {
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
    pub fn current(&self) -> Option<&T> {
        if self.on_right {
            self.right.as_ref()
        } else {
            self.left.as_ref()
        }
    }
}
