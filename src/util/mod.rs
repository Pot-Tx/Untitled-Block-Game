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

/// Identifier of an entity, component, resource, or registry entry.
pub type Id = u32;

/// Hands out entity ids, reusing the ids of removed entities through a free list.
#[derive(Default)]
pub struct IdManager {
    recycled: VecDeque<Id>,
    next: Id,
}

impl IdManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a recycled id when one is available, otherwise the next fresh id.
    pub fn create(&mut self) -> Id {
        match self.recycled.pop_front() {
            Some(id) => id,
            None => {
                self.next += 1;
                self.next - 1
            }
        }
    }

    /// Returns `id` to the free list so that [`Self::create`] can hand it out again.
    pub fn recycle(&mut self, id: Id) {
        self.recycled.push_back(id);
    }
}

/// Hands out contiguous blocks of ids and keeps the released blocks sorted by base.
pub struct IdAllocator {
    free: Vec<IdBlock>,
}

/// A run of `len` consecutive free ids starting at `base`.
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

    /// Removes `len` ids from the first free block that is large enough.
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
        unreachable!("id allocator has no free block of length {len}")
    }

    /// Releases the `len` ids starting at `base`, merging them with adjacent free blocks.
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

/// A [`OnceLock`] that panics when it is dereferenced before [`Self::init`] has run.
///
/// Used for global values that are only available once a window exists.
#[derive(Default)]
pub struct OnceInit<T> {
    inner: OnceLock<T>,
}

impl<T> Deref for OnceInit<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
            .get()
            .expect("OnceInit should be initialized before it is dereferenced")
    }
}

impl<T> OnceInit<T> {
    /// Creates an uninitialised cell; usable in `const` context.
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    /// Returns whether the cell already holds a value.
    pub fn ready(&self) -> bool {
        self.inner.get().is_some()
    }

    /// Stores `value`, logging an error when the cell was already initialised.
    pub fn init(&self, value: T) {
        if self.inner.set(value).is_err() {
            error!("OnceInit already initialized");
        }
    }
}

/// Two slots that a producer fills alternately, so a consumer keeps reading the
/// previous value until the new one is ready.
///
/// [`Self::set`] stages a value in the slot that [`Self::get`] is not reading;
/// [`Self::update`] counts the staged value down and flips the slots when it
/// expires, dropping the value that was visible before.
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

    /// Stages `item`, which becomes visible after `time` calls to [`Self::update`]
    /// (immediately when `time` is zero).
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

    /// Ticks the countdown and returns whether no swap is pending, flipping the
    /// slots and dropping the previous value once the countdown expires.
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

    /// Returns the value that is currently visible.
    #[inline]
    pub fn get(&self) -> Option<&T> {
        if self.on_right {
            self.right.as_ref()
        } else {
            self.left.as_ref()
        }
    }
}
