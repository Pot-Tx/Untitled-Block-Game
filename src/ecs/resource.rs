use crate::ecs::*;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[macro_export]
macro_rules! resources {
    (
        $(
            $vis:vis struct $name:ident $(($($ty:ty),*))?;
        )*
    ) => {
        $(
            $vis struct $name$(($(pub $ty),*))?;

            impl Resource for $name {}
        )*
    };
}

pub trait Resource: Send + Sync + 'static {}

pub struct ResourceManager {
	resources: HashMap<TypeId, RwLock<ErasedBox>>,
}

impl ResourceManager {
	pub fn new() -> Self {
		Self {
			resources: HashMap::new(),
		}
	}
	
	pub fn register<R: Resource>(&mut self, value: R) {
		let id = TypeId::of::<R>();
		self.resources
			.insert(id, RwLock::new(ErasedBox::new(value)));
	}
	
	pub fn get<R: Resource>(&self) -> RwLockReadGuard<'_, ErasedBox> {
		let id = TypeId::of::<R>();
		self.resources
			.get(&id)
			.expect(&format!("Resource with id {:?} not found", id))
			.read()
			.unwrap()
	}
	
	pub fn get_mut<R: Resource>(&self) -> RwLockWriteGuard<'_, ErasedBox> {
		let id = TypeId::of::<R>();
		self.resources
			.get(&id)
			.expect(&format!("Resource with id {:?} not found", id))
			.write()
			.unwrap()
	}
}
