use crate::ecs::*;
use crate::util::erasure::*;
use std::any::TypeId;
use std::collections::HashMap;

#[macro_export]
macro_rules! components {
    (
        $(
            $vis:vis struct $name:ident $(($($ty:ty),*))?: $storage:ident;
        )*
    ) => {
        $(
            $vis struct $name$(($(pub $ty),*))?;

            impl Component for $name {
                const STORAGE_TYPE: StorageType = StorageType::$storage;
            }
        )*
    };
}

enum ComponentStorage {
    Dense(ErasedDenseMap),
    Hash(ErasedHashMap),
}

pub enum StorageType {
    Hot,
    Cold,
}

pub trait Component: Sync + Send + 'static {
    const STORAGE_TYPE: StorageType;
}

pub struct ErasedComponent {
    id: TypeId,
    storage: ComponentStorage,
}

pub struct ComponentManager {
    components: HashMap<TypeId, ErasedComponent>,
}

pub enum ComponentIter<'a, C> {
    Dense(ErasedDenseMapIter<'a, C>),
    Hash(ErasedHashMapIter<'a, C>),
}

pub enum ComponentIterMut<'a, C> {
    Dense(ErasedDenseMapIterMut<'a, C>),
    Hash(ErasedHashMapIterMut<'a, C>),
}

unsafe impl Sync for ComponentManager {}

unsafe impl Send for ComponentManager {}

impl ComponentManager {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }

    pub fn register<C: Component>(&mut self) {
        let id = TypeId::of::<C>();
        self.components.insert(
            id,
            ErasedComponent {
                id,
                storage: match C::STORAGE_TYPE {
                    StorageType::Hot => ComponentStorage::Dense(ErasedDenseMap::new::<C>()),
                    StorageType::Cold => ComponentStorage::Hash(ErasedHashMap::new::<C>()),
                },
            },
        );
    }

    #[inline]
    pub fn by_id(&self, id: TypeId) -> &ErasedComponent {
        self.components
            .get(&id)
            .expect(&format!("Component with id {:?} not found", id))
    }

    #[inline]
    pub fn by_id_mut(&mut self, id: TypeId) -> &mut ErasedComponent {
        self.components
            .get_mut(&id)
            .expect(&format!("Component with id {:?} not found", id))
    }

    #[inline]
    pub fn get<C: Component>(&self) -> &ErasedComponent {
        self.by_id(TypeId::of::<C>())
    }

    #[inline]
    pub fn get_mut<C: Component>(&mut self) -> &mut ErasedComponent {
        self.by_id_mut(TypeId::of::<C>())
    }
    
    pub fn remove_all(&mut self, entity: Id) {
        self.components.iter_mut().for_each(|(_, c)| {
            c.remove_and_drop(entity);
        });
    }
}

impl ErasedComponent {
    #[inline]
    pub fn contains(&self, entity: Id) -> bool {
        match &self.storage {
            ComponentStorage::Dense(map) => map.contains(entity),
            ComponentStorage::Hash(map) => map.contains(entity),
        }
    }

    #[inline]
    pub fn get<C: Component>(&self, entity: Id) -> Option<&C> {
        assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => map.get(entity),
            ComponentStorage::Hash(map) => map.get(entity),
        }
    }

    #[inline]
    pub fn get_mut<C: Component>(&self, entity: Id) -> Option<&mut C> {
        assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => map.get_mut(entity),
            ComponentStorage::Hash(map) => map.get_mut(entity),
        }
    }
    
    pub fn iter<C: Component>(&self) -> ComponentIter<'_, C> {
        assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => ComponentIter::Dense(map.iter()),
            ComponentStorage::Hash(map) => ComponentIter::Hash(map.iter()),
        }
    }
    
    pub fn iter_mut<C: Component>(&self) -> ComponentIterMut<'_, C> {
        assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => ComponentIterMut::Dense(map.iter_mut()),
            ComponentStorage::Hash(map) => ComponentIterMut::Hash(map.iter_mut()),
        }
    }

    #[inline]
    pub fn insert<C: Component>(&mut self, entity: Id, value: C) -> Option<C> {
        assert_eq!(self.id, TypeId::of::<C>());
        match &mut self.storage {
            ComponentStorage::Dense(map) => map.insert(entity, value),
            ComponentStorage::Hash(map) => map.insert(entity, value),
        }
    }

    #[inline]
    pub fn insert_erased(&mut self, entity: Id, value: ErasedBox) {
        match &mut self.storage {
            ComponentStorage::Dense(map) => map.insert_erased(entity, value),
            ComponentStorage::Hash(map) => map.insert_erased(entity, value),
        }
    }

    #[inline]
    pub fn remove<C: Component>(&mut self, entity: Id) -> Option<C> {
        match &mut self.storage {
            ComponentStorage::Dense(map) => map.remove(entity),
            ComponentStorage::Hash(map) => map.remove(entity),
        }
    }

    #[inline]
    pub fn remove_and_drop(&mut self, entity: Id) {
        match &mut self.storage {
            ComponentStorage::Dense(map) => map.remove_and_drop(entity),
            ComponentStorage::Hash(map) => map.remove_and_drop(entity),
        }
    }
}

impl<'a, C: 'a> Iterator for ComponentIter<'a, C> {
    type Item = (Id, &'a C);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ComponentIter::Dense(it) => it.next(),
            ComponentIter::Hash(it) => it.next(),
        }
    }
}

impl<'a, C: 'a> Iterator for ComponentIterMut<'a, C> {
    type Item = (Id, &'a mut C);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ComponentIterMut::Dense(it) => it.next(),
            ComponentIterMut::Hash(it) => it.next(),
        }
    }
}
