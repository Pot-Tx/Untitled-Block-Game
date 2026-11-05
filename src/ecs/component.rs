use crate::ecs::*;
use crate::util::erasure::*;
use bimap::BiMap;
use log::error;
use std::any::{TypeId, type_name};
use std::collections::HashMap;

#[macro_export]
macro_rules! components {
    () => {};

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ( $($ty:ty),* $(,)? ): $storage:ident $(, $flag:ident)*;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name($(pub $ty),*);

        impl $crate::ecs::Component for $name {
            const SETTINGS: $crate::ecs::ComponentSettings =
                $crate::component_settings!($name, $storage $(, $flag)*);
        }

        $crate::components! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident { $($fvis:vis $fname:ident : $fty:ty),* $(,)? }: $storage:ident $(, $flag:ident)*;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name { $($fvis $fname : $fty),* }

        impl $crate::ecs::Component for $name {
            const SETTINGS: $crate::ecs::ComponentSettings =
                $crate::component_settings!($name, $storage $(, $flag)*);
        }

        $crate::components! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident: $storage:ident $(, $flag:ident)*;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name;

        impl $crate::ecs::Component for $name {
            const SETTINGS: $crate::ecs::ComponentSettings =
                $crate::component_settings!($name, $storage $(, $flag)*);
        }

        $crate::components! { $($rest)* }
    };
}
/// The settings of one component: where it is stored and which of its data is
/// written to a file.
///
/// `asset` adds the codec of the text files a human edits, `save` the codec of
/// the binary files the game writes; both can be given, in either order.
#[doc(hidden)]
#[macro_export]
macro_rules! component_settings {
    ($name:ident, $storage:ident) => {
        $crate::ecs::ComponentSettings {
            storage: $crate::ecs::StorageType::$storage,
            asset: None,
            save: None,
        }
    };

    ($name:ident, $storage:ident, asset) => {
        $crate::ecs::ComponentSettings {
            storage: $crate::ecs::StorageType::$storage,
            asset: Some($crate::ecs::asset_of::<$name>()),
            save: None,
        }
    };

    ($name:ident, $storage:ident, save) => {
        $crate::ecs::ComponentSettings {
            storage: $crate::ecs::StorageType::$storage,
            asset: None,
            save: Some($crate::ecs::codec_of::<$name>($name::SAVE_VERSION)),
        }
    };

    ($name:ident, $storage:ident, asset, save) => {
        $crate::component_settings!($name, $storage, save, asset)
    };

    ($name:ident, $storage:ident, save, asset) => {
        $crate::ecs::ComponentSettings {
            storage: $crate::ecs::StorageType::$storage,
            asset: Some($crate::ecs::asset_of::<$name>()),
            save: Some($crate::ecs::codec_of::<$name>($name::SAVE_VERSION)),
        }
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
    pub save: Option<Codec>,
}

pub trait Component: Clone + Sync + Send + 'static {
    const SETTINGS: ComponentSettings;

    /// The version of the shape this component is saved with; it is written
    /// next to the component name so that a save can tell the shapes apart.
    const SAVE_VERSION: u8 = 0;
}

pub struct ErasedComponent {
    pub id: TypeId,
    /// The name the component is registered under.
    pub name: &'static str,
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
            error!(
                "component name {} is already used by another component",
                name
            );
        }

        self.names.insert(id, name);
        self.components.insert(
            id,
            ErasedComponent {
                id,
                name,
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
        let id = self
            .names
            .get_by_right(name)
            .ok_or(QueryError::MissingComponentName(name.into()))?;

        self.components
            .get(id)
            .ok_or(QueryError::MissingComponentId(*id))
    }

    #[inline]
    pub fn by_name_mut(&mut self, name: &str) -> Result<&mut ErasedComponent, QueryError> {
        let id = self
            .names
            .get_by_right(name)
            .ok_or(QueryError::MissingComponentName(name.into()))?;

        self.components
            .get_mut(id)
            .ok_or(QueryError::MissingComponentId(*id))
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

    /// The registered components together with their names, ordered by name, so
    /// that a file does not depend on the order they were registered in.
    pub fn registered(&self) -> impl Iterator<Item = (&'static str, &ErasedComponent)> {
        let mut entries = self
            .components
            .iter()
            .filter_map(|(&id, component)| {
                self.names.get_by_left(&id).map(|name| (*name, component))
            })
            .collect::<Vec<_>>();

        entries.sort_by_key(|(name, _)| *name);

        entries.into_iter()
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
        debug_assert_eq!(self.id, TypeId::of::<C>());
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

    /// A borrowed, type-erased view of the value of `entity`.
    #[inline]
    pub fn erased_ref(&self, entity: Id) -> Option<ErasedRef<'_>> {
        match &self.storage {
            ComponentStorage::Dense(map) => map.ref_of(entity),
            ComponentStorage::Hash(map) => map.ref_of(entity),
        }
    }

    /// Decodes a value of this component from an asset file.
    pub fn read_asset(&self, raw: &RawValue) -> anyhow::Result<ErasedBox> {
        let asset = self
            .settings
            .asset
            .ok_or_else(|| anyhow::anyhow!("component {} cannot be read from data", self.name))?;

        (asset.read)(raw)
    }

    /// Encodes a value of this component for an asset file.
    pub fn write_asset(&self, value: ErasedRef<'_>) -> anyhow::Result<Box<RawValue>> {
        let asset = self
            .settings
            .asset
            .ok_or_else(|| anyhow::anyhow!("component {} cannot be written to data", self.name))?;

        (asset.write)(value)
    }

    /// Decodes a value of this component from a save payload.
    pub fn read_save(&self, payload: &[u8]) -> anyhow::Result<ErasedBox> {
        let codec = self
            .settings
            .save
            .ok_or_else(|| anyhow::anyhow!("component {} cannot be read from a save", self.name))?;

        (codec.read)(payload)
    }

    /// Encodes a value of this component as a save payload.
    pub fn write_save(&self, value: ErasedRef<'_>) -> anyhow::Result<Vec<u8>> {
        let codec = self.settings.save.ok_or_else(|| {
            anyhow::anyhow!("component {} cannot be written to a save", self.name)
        })?;

        (codec.write)(value)
    }

    /// The version of the shape this component is saved with.
    #[inline]
    pub fn save_version(&self) -> u8 {
        self.settings.save.map_or(0, |codec| codec.version)
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
