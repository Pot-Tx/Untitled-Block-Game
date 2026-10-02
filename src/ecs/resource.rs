use crate::ecs::*;
use bimap::BiMap;
use log::error;
use std::any::{type_name, TypeId};
use std::collections::HashMap;

#[macro_export]
macro_rules! resources {
    () => {};

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ( $($ty:ty),* $(,)? );
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name($(pub $ty),*);

        impl $crate::ecs::Resource for $name {}

        $crate::resources! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident { $($fvis:vis $fname:ident : $fty:ty),* $(,)? };
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name { $($fvis $fname : $fty),* }

        impl $crate::ecs::Resource for $name {}

        $crate::resources! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name;

        impl $crate::ecs::Resource for $name {}

        $crate::resources! { $($rest)* }
    };
}

/// A resource has no storage to choose, so the settings are empty for now; they
/// are the place where the data capabilities of a resource will be declared.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResourceSettings;

/// A value that exists once for the whole world.
pub trait Resource: Send + Sync + 'static {
    const SETTINGS: ResourceSettings = ResourceSettings;
}

pub struct ErasedResource {
    pub id: TypeId,
    value: ErasedBox,
    pub settings: &'static ResourceSettings,
}

#[derive(Default)]
pub struct ResourceManager {
    resources: HashMap<TypeId, ErasedResource>,
    names: BiMap<TypeId, &'static str>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces any earlier resource of the same type.
    pub fn register<R: Resource>(&mut self, name: &'static str, value: R) {
        let id = TypeId::of::<R>();

        if let Some(taken) = self.names.get_by_right(name)
            && *taken != id
        {
            error!("resource name {} is already used by another resource", name);
        }

        self.names.insert(id, name);
        self.resources.insert(id, ErasedResource {
            id,
            value: ErasedBox::new(value),
            settings: &R::SETTINGS,
        });
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
    pub fn by_id(&self, id: TypeId) -> Result<&ErasedResource, QueryError> {
        self.resources
            .get(&id)
            .ok_or_else(|| QueryError::MissingResourceId(id))
    }
    
    #[inline]
    pub fn by_name(&self, name: &str) -> Result<&ErasedResource, QueryError> {
        let id = self.names.get_by_right(name).ok_or(QueryError::MissingResourceName(name.into()))?;
        
        self.resources
            .get(id)
            .ok_or_else(|| QueryError::MissingResourceId(*id))
    }
    
    #[inline]
    pub fn get<R: Resource>(&self) -> Result<&ErasedResource, QueryError> {
        self.resources
            .get(&TypeId::of::<R>())
            .ok_or_else(|| QueryError::MissingResource(type_name::<R>()))
    }
}

impl ErasedResource {
    #[inline]
    pub fn get<R: Resource>(&self) -> &R {
        debug_assert_eq!(self.id, TypeId::of::<R>());
        self.value.cast()
    }
    
    #[inline]
    pub fn get_mut<R: Resource>(&self) -> &mut R {
        debug_assert_eq!(self.id, TypeId::of::<R>());
        self.value.cast_mut()
    }
}
