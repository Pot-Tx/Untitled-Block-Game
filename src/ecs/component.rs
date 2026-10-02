use crate::ecs::*;
use crate::util::erasure::*;
use bimap::BiMap;
use log::error;
use std::any::{type_name, TypeId};
use std::collections::HashMap;

#[macro_export]
macro_rules! components {
    () => {};

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ( $($ty:ty),* $(,)? ): $storage:ident, data;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name($(pub $ty),*);

        impl $crate::ecs::Component for $name {
            const SETTINGS: $crate::ecs::ComponentSettings = $crate::ecs::ComponentSettings {
                storage: $crate::ecs::StorageType::$storage,
                asset: Some($crate::ecs::asset_of_component::<$name>()),
            };
        }

        $crate::components! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ( $($ty:ty),* $(,)? ): $storage:ident;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name($(pub $ty),*);

        impl $crate::ecs::Component for $name {
            const SETTINGS: $crate::ecs::ComponentSettings = $crate::ecs::ComponentSettings {
                storage: $crate::ecs::StorageType::$storage,
                asset: None,
            };
        }

        $crate::components! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident { $($fvis:vis $fname:ident : $fty:ty),* $(,)? }: $storage:ident, data;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name { $($fvis $fname : $fty),* }

        impl $crate::ecs::Component for $name {
            const SETTINGS: $crate::ecs::ComponentSettings = $crate::ecs::ComponentSettings {
                storage: $crate::ecs::StorageType::$storage,
                asset: Some($crate::ecs::asset_of_component::<$name>()),
            };
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
            const SETTINGS: $crate::ecs::ComponentSettings = $crate::ecs::ComponentSettings {
                storage: $crate::ecs::StorageType::$storage,
                asset: None,
            };
        }

        $crate::components! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident: $storage:ident, data;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name;

        impl $crate::ecs::Component for $name {
            const SETTINGS: $crate::ecs::ComponentSettings = $crate::ecs::ComponentSettings {
                storage: $crate::ecs::StorageType::$storage,
                asset: Some($crate::ecs::asset_of_component::<$name>()),
            };
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
            const SETTINGS: $crate::ecs::ComponentSettings = $crate::ecs::ComponentSettings {
                storage: $crate::ecs::StorageType::$storage,
                asset: None,
            };
        }

        $crate::components! { $($rest)* }
    };
}

enum ComponentStorage {
    Dense(ErasedDenseMap),
    Hash(ErasedHashMap),
}

#[derive(Clone, Copy, Debug)]
pub enum StorageType {
    /// Dense storage, for components that most entities have.
    Hot,
    /// Hash map storage, for components that only few entities have.
    Cold,
}

#[derive(Clone, Copy, Debug)]
pub struct ComponentSettings {
    pub storage: StorageType,
    pub asset: Option<Asset>,
}

pub trait Component: Clone + Sync + Send + 'static {
    const SETTINGS: ComponentSettings;
}

pub struct ErasedComponent {
    pub id: TypeId,
    storage: ComponentStorage,
    pub settings: &'static ComponentSettings,
}

#[derive(Default)]
pub struct ComponentManager {
    components: HashMap<TypeId, ErasedComponent>,
    names: BiMap<TypeId, &'static str>,
}

pub enum ComponentIter<'a, C> {
    Dense(ErasedDenseMapIter<'a, C>),
    Hash(ErasedHashMapIter<'a, C>),
}

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

    pub fn register<C: Component>(&mut self, name: &'static str) {
        let id = TypeId::of::<C>();

        if let Some(taken) = self.names.get_by_right(name)
            && *taken != id
        {
            error!("component name {} is already used by another component", name);
        }

        self.names.insert(id, name);
        self.components.insert(
            id,
            ErasedComponent {
                id,
                storage: match C::SETTINGS.storage {
                    StorageType::Hot => ComponentStorage::Dense(ErasedDenseMap::new::<C>()),
                    StorageType::Cold => ComponentStorage::Hash(ErasedHashMap::new::<C>()),
                },
                settings: &C::SETTINGS,
            },
        );
    }

    #[inline]
    pub fn id_of(&self, name: &str) -> Option<TypeId> {
        self.names.get_by_right(name).copied()
    }

    #[inline]
    pub fn name_of(&self, id: TypeId) -> Option<&'static str> {
        self.names.get_by_left(&id).copied()
    }
    
    #[inline]
    pub fn by_id(&self, id: TypeId) -> Result<&ErasedComponent, QueryError> {
        self.components
            .get(&id)
            .ok_or(QueryError::MissingComponentId(id))
    }

    #[inline]
    pub fn by_id_mut(&mut self, id: TypeId) -> Result<&mut ErasedComponent, QueryError> {
        self.components
            .get_mut(&id)
            .ok_or(QueryError::MissingComponentId(id))
    }
    
    #[inline]
    pub fn by_name(&self, name: &str) -> Result<&ErasedComponent, QueryError> {
        let id = self.names.get_by_right(name).ok_or(QueryError::MissingComponentName(name.into()))?;
        
        self.components
            .get(id)
            .ok_or_else(move || QueryError::MissingComponentId(*id))
    }
    
    #[inline]
    pub fn by_name_mut(&mut self, name: &str) -> Result<&mut ErasedComponent, QueryError> {
        let id = self.names.get_by_right(name).ok_or(QueryError::MissingComponentName(name.into()))?;
        
        self.components
            .get_mut(id)
            .ok_or_else(|| QueryError::MissingComponentId(*id))
    }

    #[inline]
    pub fn get<C: Component>(&self) -> Result<&ErasedComponent, QueryError> {
        self.components
            .get(&TypeId::of::<C>())
            .ok_or_else(|| QueryError::MissingComponent(type_name::<C>()))
    }

    #[inline]
    pub fn get_mut<C: Component>(&mut self) -> Result<&mut ErasedComponent, QueryError> {
        self.components
            .get_mut(&TypeId::of::<C>())
            .ok_or_else(|| QueryError::MissingComponent(type_name::<C>()))
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
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => map.get(entity),
            ComponentStorage::Hash(map) => map.get(entity),
        }
    }

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

    pub fn iter<C: Component>(&self) -> ComponentIter<'_, C> {
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => ComponentIter::Dense(map.iter()),
            ComponentStorage::Hash(map) => ComponentIter::Hash(map.iter()),
        }
    }

    pub fn iter_mut<C: Component>(&self) -> ComponentIterMut<'_, C> {
        debug_assert_eq!(self.id, TypeId::of::<C>());
        match &self.storage {
            ComponentStorage::Dense(map) => ComponentIterMut::Dense(map.iter_mut()),
            ComponentStorage::Hash(map) => ComponentIterMut::Hash(map.iter_mut()),
        }
    }

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

pub const fn asset_of_component<C: Component + DeserializeOwned + Serialize>() -> Asset {
    Asset {
        read: |raw| Ok(ErasedBox::new_clone(
            ASSET_OPTIONS.from_str::<C>(raw.get_ron())?,
        )),
        write: |value| Ok(RawValue::from_boxed_ron(
            ASSET_OPTIONS.to_string(value.cast::<C>())?.into_boxed_str()
        )?),
    }
}