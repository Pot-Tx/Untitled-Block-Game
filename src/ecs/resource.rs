use crate::ecs::*;
use std::any::TypeId;
use std::collections::HashMap;

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
    resources: HashMap<TypeId, ErasedBox>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }

    pub fn register<R: Resource>(&mut self, value: R) {
        let id = TypeId::of::<R>();
        self.resources.insert(id, ErasedBox::new(value));
    }
    
    pub fn get<R: Resource>(&self) -> &R {
        let id = TypeId::of::<R>();
        self.resources
            .get(&id)
            .expect(&format!("Resource with id {:?} not found", id))
            .cast()
    }
    
    pub fn get_mut<R: Resource>(&self) -> &mut R {
        let id = TypeId::of::<R>();
        self.resources
            .get(&id)
            .expect(&format!("Resource with id {:?} not found", id))
            .cast_mut()
    }
}
