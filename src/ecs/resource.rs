use crate::ecs::*;
use std::any::TypeId;
use std::collections::HashMap;

/// Declares resources and registers them with the ECS.
///
/// Every resource is written as `#[attributes] pub struct Name { ... };`,
/// followed by a semicolon, and the macro implements [`Resource`] for it.
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

/// A value that exists once for the whole world.
pub trait Resource: Send + Sync + 'static {}

/// The type erased storage of every registered resource.
#[derive(Default)]
pub struct ResourceManager {
    resources: HashMap<TypeId, ErasedBox>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `value`, replacing any earlier resource of the same type.
    pub fn register<R: Resource>(&mut self, value: R) {
        let id = TypeId::of::<R>();
        self.resources.insert(id, ErasedBox::new(value));
    }

    /// The resource of type `R`.
    ///
    /// Takes `&self` because systems read a resource they also write; the caller
    /// is responsible for not creating two mutable borrows at once.
    pub fn get<R: Resource>(&self) -> &R {
        let id = TypeId::of::<R>();
        self.resources
            .get(&id)
            .unwrap_or_else(|| panic!("resource with id {:?} not found", id))
            .cast()
    }

    /// The resource of type `R`, which has to be registered.
    pub fn try_get<R: Resource>(&self) -> Result<&R, QueryError> {
        self.resources
            .get(&TypeId::of::<R>())
            .map(|resource| resource.cast())
            .ok_or_else(QueryError::resource::<R>)
    }

    /// Mutable access to the resource of type `R`.
    pub fn get_mut<R: Resource>(&self) -> &mut R {
        let id = TypeId::of::<R>();
        self.resources
            .get(&id)
            .unwrap_or_else(|| panic!("resource with id {:?} not found", id))
            .cast_mut()
    }

    /// Mutable access to the resource of type `R`, which has to be registered.
    pub fn try_get_mut<R: Resource>(&self) -> Result<&mut R, QueryError> {
        self.resources
            .get(&TypeId::of::<R>())
            .map(|resource| resource.cast_mut())
            .ok_or_else(QueryError::resource::<R>)
    }
}
