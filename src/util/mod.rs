pub mod bounding;
pub mod collection;
pub mod coord;
pub mod erasure;
pub mod math;
pub mod transform;

use std::collections::VecDeque;

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
