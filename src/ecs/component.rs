use crate::ecs::*;
use crate::util::erasure::*;
use std::any::TypeId;
use std::collections::HashMap;

/// Declares components and registers them with the ECS.
///
/// Every component is written as
/// `#[attributes] pub struct Name { ... }: StorageType;`, followed by a
/// semicolon, and the macro implements [`Component`] with the given storage
/// type for it.
#[macro_export]
macro_rules! components {
    () => {};

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ( $($ty:ty),* $(,)? ): $storage:ident;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name($(pub $ty),*);

        impl $crate::ecs::Component for $name {
            const STORAGE_TYPE: $crate::ecs::StorageType = $crate::ecs::StorageType::$storage;
        }

        $crate::components! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident { $($fvis:vis $fname:ident : $fty:ty),* $(,)? }: $storage:ident;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name { $($fvis $fname : $fty),* }

        impl $crate::ecs::Component for $name {
            const STORAGE_TYPE: $crate::ecs::StorageType = $crate::ecs::StorageType::$storage;
        }

        $crate::components! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident: $storage:ident;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name;

        impl $crate::ecs::Component for $name {
            const STORAGE_TYPE: $crate::ecs::StorageType = $crate::ecs::StorageType::$storage;
        }

        $crate::components! { $($rest)* }
    };
}

enum ComponentStorage {
    Dense(ErasedDenseMap),
    Hash(ErasedHashMap),
}

/// How the values of a component are stored.
pub enum StorageType {
    /// Dense storage, for components that most entities have.
    Hot,
    /// Hash map storage, for components that only few entities have.
    Cold,
}

/// A value that can be attached to entities.
pub trait Component: Sync + Send + 'static {
    const STORAGE_TYPE: StorageType;
}

/// The type erased storage of a single component, for every entity.
pub struct ErasedComponent {
    id: TypeId,
    storage: ComponentStorage,
}

/// The type erased storage of every registered component.
#[derive(Default)]
pub struct ComponentManager {
    components: HashMap<TypeId, ErasedComponent>,
}

/// Iterates the values of a component storage, read as `C`.
pub enum ComponentIter<'a, C> {
    Dense(ErasedDenseMapIter<'a, C>),
    Hash(ErasedHashMapIter<'a, C>),
}

/// Mutably iterates the values of a component storage, read as `C`.
pub enum ComponentIterMut<'a, C> {
    Dense(ErasedDenseMapIterMut<'a, C>),
    Hash(ErasedHashMapIterMut<'a, C>),
}

// SAFETY: the manager is only reachable through the storage of each component,
// which requires its values to be `Send + Sync` (see [`Component`]).
unsafe impl Sync for ComponentManager {}

unsafe impl Send for ComponentManager {}

impl ComponentManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates the storage of `C`, whose values can be inserted afterwards.
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

    /// The storage of the component with `id`.
    #[inline]
    pub fn by_id(&self, id: TypeId) -> &ErasedComponent {
        self.components
            .get(&id)
            .unwrap_or_else(|| panic!("component with id {:?} not found", id))
    }

    /// Mutable access to the storage of the component with `id`.
    #[inline]
    pub fn by_id_mut(&mut self, id: TypeId) -> &mut ErasedComponent {
        self.components
            .get_mut(&id)
            .unwrap_or_else(|| panic!("component with id {:?} not found", id))
    }

    /// Mutable access to the storage of the component with `id`, which has to be
    /// registered.
    #[inline]
    pub fn try_by_id_mut(&mut self, id: TypeId) -> Result<&mut ErasedComponent, QueryError> {
        self.components
            .get_mut(&id)
            .ok_or(QueryError::MissingComponentId(id))
    }

    /// The storage of `C`.
    #[inline]
    pub fn get<C: Component>(&self) -> &ErasedComponent {
        self.by_id(TypeId::of::<C>())
    }

    /// The storage of `C`, which has to be registered.
    #[inline]
    pub fn try_get<C: Component>(&self) -> Result<&ErasedComponent, QueryError> {
        self.components
            .get(&TypeId::of::<C>())
            .ok_or_else(QueryError::component::<C>)
    }

    /// Mutable access to the storage of `C`.
    #[inline]
    pub fn get_mut<C: Component>(&mut self) -> &mut ErasedComponent {
        self.by_id_mut(TypeId::of::<C>())
    }

    /// Mutable access to the storage of `C`, which has to be registered.
    #[inline]
    pub fn try_get_mut<C: Component>(&mut self) -> Result<&mut ErasedComponent, QueryError> {
        self.components
            .get_mut(&TypeId::of::<C>())
            .ok_or_else(QueryError::component::<C>)
    }

    /// Removes the components of `entity` from every registered storage.
    pub fn remove_all(&mut self, entity: Id) {
        self.components.iter_mut().for_each(|(_, c)| {
            c.remove_and_drop(entity);
        });
    }
}

impl ErasedComponent {
    /// Returns whether `entity` has a value in this storage.
    #[inline]
    pub fn contains(&self, entity: Id) -> bool {
        match &self.storage {
            ComponentStorage::Dense(map) => map.contains(entity),
            ComponentStorage::Hash(map) => map.contains(entity),
        }
    }

    /// Borrows the value of `entity` as `C`.
    #[inline]
    pub fn get<C: Component>(&self, entity: Id) -> Option<&C> {
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => map.get(entity),
            ComponentStorage::Hash(map) => map.get(entity),
        }
    }

    /// Borrows the value of `entity` as `C`.
    ///
    /// Takes `&self` because queries hand out one mutable borrow per entity
    /// while sharing the storage among all of them; see
    /// [`ErasedBox::cast_mut`](crate::util::erasure::ErasedBox::cast_mut).
    #[inline]
    pub fn get_mut<C: Component>(&self, entity: Id) -> Option<&mut C> {
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => map.get_mut(entity),
            ComponentStorage::Hash(map) => map.get_mut(entity),
        }
    }

    /// Iterates every entity that has a value in this storage.
    pub fn iter<C: Component>(&self) -> ComponentIter<'_, C> {
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => ComponentIter::Dense(map.iter()),
            ComponentStorage::Hash(map) => ComponentIter::Hash(map.iter()),
        }
    }

    /// Mutably iterates every entity that has a value in this storage.
    pub fn iter_mut<C: Component>(&self) -> ComponentIterMut<'_, C> {
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => ComponentIterMut::Dense(map.iter_mut()),
            ComponentStorage::Hash(map) => ComponentIterMut::Hash(map.iter_mut()),
        }
    }

    /// Sets the value of `entity`, returning the previous value.
    #[inline]
    pub fn insert<C: Component>(&mut self, entity: Id, value: C) -> Option<C> {
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &mut self.storage {
            ComponentStorage::Dense(map) => map.insert(entity, value),
            ComponentStorage::Hash(map) => map.insert(entity, value),
        }
    }

    /// Sets the value of `entity` from an erased value, which has to have the
    /// layout of `C`.
    #[inline]
    pub fn insert_erased(&mut self, entity: Id, value: ErasedBox) {
        match &mut self.storage {
            ComponentStorage::Dense(map) => map.insert_erased(entity, value),
            ComponentStorage::Hash(map) => map.insert_erased(entity, value),
        }
    }

    /// Removes the value of `entity` and returns it.
    #[inline]
    pub fn remove<C: Component>(&mut self, entity: Id) -> Option<C> {
        match &mut self.storage {
            ComponentStorage::Dense(map) => map.remove(entity),
            ComponentStorage::Hash(map) => map.remove(entity),
        }
    }

    /// Removes and drops the value of `entity`.
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
