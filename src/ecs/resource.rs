use crate::ecs::*;
use bimap::BiMap;
use log::{error, warn};
use ron::value::RawValue;
use serde::{Deserialize, Serialize};
use std::any::{TypeId, type_name};
use std::collections::HashMap;

#[macro_export]
macro_rules! resources {
    () => {};

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident ( $($ty:ty),* $(,)? ) $(: $($flag:ident),*)?;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name($(pub $ty),*);

        impl $crate::ecs::Resource for $name {
            const SETTINGS: $crate::ecs::ResourceSettings =
                $crate::resource_settings!($name $(, $($flag),*)?);
        }

        $crate::resources! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident { $($fvis:vis $fname:ident : $fty:ty),* $(,)? }
        $(: $($flag:ident),*)?;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name { $($fvis $fname : $fty),* }

        impl $crate::ecs::Resource for $name {
            const SETTINGS: $crate::ecs::ResourceSettings =
                $crate::resource_settings!($name $(, $($flag),*)?);
        }

        $crate::resources! { $($rest)* }
    };

    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident $(: $($flag:ident),*)?;
        $($rest:tt)*
    ) => {
        $(#[$attr])*
        $vis struct $name;

        impl $crate::ecs::Resource for $name {
            const SETTINGS: $crate::ecs::ResourceSettings =
                $crate::resource_settings!($name $(, $($flag),*)?);
        }

        $crate::resources! { $($rest)* }
    };
}

/// The data files one resource takes part in.
#[doc(hidden)]
#[macro_export]
macro_rules! resource_settings {
    ($name:ident) => {
        $crate::ecs::ResourceSettings::NONE
    };

    ($name:ident, asset) => {
        $crate::ecs::ResourceSettings {
            asset: Some($crate::ecs::asset_of::<$name>()),
            save: None,
        }
    };

    ($name:ident, save) => {
        $crate::ecs::ResourceSettings {
            asset: None,
            save: Some($crate::ecs::codec_of::<$name>($name::SAVE_VERSION)),
        }
    };

    ($name:ident, asset, save) => {
        $crate::resource_settings!($name, save, asset)
    };

    ($name:ident, save, asset) => {
        $crate::ecs::ResourceSettings {
            asset: Some($crate::ecs::asset_of::<$name>()),
            save: Some($crate::ecs::codec_of::<$name>($name::SAVE_VERSION)),
        }
    };
}

/// The settings of one resource: which of its data is written to an asset or
/// to a save.
///
/// `asset` drives the text files a human edits, `save` the binary files the
/// game writes. Both use the same `Serialize` and `Deserialize`
/// implementations.
#[derive(Clone, Copy, Debug)]
pub struct ResourceSettings {
    pub asset: Option<Asset>,
    pub save: Option<Codec>,
}

impl ResourceSettings {
    pub const NONE: Self = Self {
        asset: None,
        save: None,
    };
}

impl Default for ResourceSettings {
    fn default() -> Self {
        Self::NONE
    }
}

/// A value that exists once for the whole world.
pub trait Resource: Send + Sync + 'static {
    const SETTINGS: ResourceSettings = ResourceSettings::NONE;

    /// The version of the shape this resource is saved with.
    const SAVE_VERSION: u8 = 0;
}

pub struct ErasedResource {
    pub id: TypeId,
    /// The name the resource is registered under.
    pub name: &'static str,
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
        self.resources.insert(
            id,
            ErasedResource {
                id,
                name,
                value: ErasedBox::new(value),
                settings: &R::SETTINGS,
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
    pub fn by_id(&self, id: TypeId) -> Result<&ErasedResource, QueryError> {
        self.resources
            .get(&id)
            .ok_or(QueryError::MissingResourceId(id))
    }

    fn by_id_mut(&mut self, id: TypeId) -> Result<&mut ErasedResource, QueryError> {
        self.resources
            .get_mut(&id)
            .ok_or(QueryError::MissingResourceId(id))
    }

    #[inline]
    pub fn by_name(&self, name: &str) -> Result<&ErasedResource, QueryError> {
        let id = self
            .names
            .get_by_right(name)
            .ok_or(QueryError::MissingResourceName(name.into()))?;

        self.resources
            .get(id)
            .ok_or(QueryError::MissingResourceId(*id))
    }

    #[inline]
    pub fn get<R: Resource>(&self) -> Result<&ErasedResource, QueryError> {
        self.resources
            .get(&TypeId::of::<R>())
            .ok_or_else(|| QueryError::MissingResource(type_name::<R>()))
    }

    /// Replaces the registered resources with the values in a text asset.
    ///
    /// Every value is decoded before any resource is replaced, so a bad entry
    /// leaves the manager unchanged.
    pub fn read_asset(&mut self, raw: &RawResources) -> anyhow::Result<()> {
        let mut values = Vec::with_capacity(raw.resources.len());

        for (name, value) in &raw.resources {
            let resource = self.by_name(name)?;
            values.push((resource.id, resource.read_asset(value)?));
        }

        for (id, value) in values {
            self.by_id_mut(id)?.replace(value);
        }

        Ok(())
    }

    /// Encodes every resource that can be read from a text asset.
    pub fn write_asset(&self) -> anyhow::Result<RawResources> {
        let mut resources = Vec::new();

        for (name, resource) in self.registered() {
            if resource.settings.asset.is_some() {
                resources.push((
                    String::from(name),
                    resource.write_asset(resource.erased_ref())?,
                ));
            }
        }

        Ok(RawResources { resources })
    }

    /// The registered resources and their names, ordered by name so that files
    /// do not depend on the registration order.
    fn registered(&self) -> impl Iterator<Item = (&'static str, &ErasedResource)> {
        let mut entries = self
            .resources
            .iter()
            .filter_map(|(&id, resource)| self.names.get_by_left(&id).map(|name| (*name, resource)))
            .collect::<Vec<_>>();

        entries.sort_by_key(|(name, _)| *name);

        entries.into_iter()
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

    /// Decodes a value of this resource from an asset file.
    pub fn read_asset(&self, raw: &RawValue) -> anyhow::Result<ErasedBox> {
        let asset = self
            .settings
            .asset
            .ok_or_else(|| anyhow::anyhow!("resource {} cannot be read from data", self.name))?;

        (asset.read)(raw)
    }

    /// Encodes a value of this resource for an asset file.
    pub fn write_asset(&self, value: ErasedRef<'_>) -> anyhow::Result<Box<RawValue>> {
        let asset = self
            .settings
            .asset
            .ok_or_else(|| anyhow::anyhow!("resource {} cannot be written to data", self.name))?;

        (asset.write)(value)
    }

    /// Decodes a value of this resource from a save payload.
    pub fn read_save(&self, payload: &[u8]) -> anyhow::Result<ErasedBox> {
        let codec = self
            .settings
            .save
            .ok_or_else(|| anyhow::anyhow!("resource {} cannot be read from a save", self.name))?;

        (codec.read)(payload)
    }

    /// Encodes a value of this resource as a save payload.
    pub fn write_save(&self, value: ErasedRef<'_>) -> anyhow::Result<Vec<u8>> {
        let codec = self
            .settings
            .save
            .ok_or_else(|| anyhow::anyhow!("resource {} cannot be written to a save", self.name))?;

        (codec.write)(value)
    }

    /// The version of the shape this resource is saved with.
    #[inline]
    pub fn save_version(&self) -> u8 {
        self.settings.save.map_or(0, |codec| codec.version)
    }

    fn erased_ref(&self) -> ErasedRef<'_> {
        self.value.as_ref()
    }

    fn replace(&mut self, value: ErasedBox) {
        self.value = value;
    }
}

/// The resources of one text asset document.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawResources {
    #[serde(default)]
    pub resources: Vec<(String, Box<RawValue>)>,
}

/// The resources of one save.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ResourceSet {
    pub resources: Vec<SavedResource>,
}

/// One resource and the payload of its saved value.
#[derive(Clone, Serialize, Deserialize)]
pub struct SavedResource {
    pub name: String,
    pub version: u8,
    pub payload: Vec<u8>,
}

impl ResourceSet {
    /// Encodes every registered resource that takes part in a save.
    pub fn build(resources: &ResourceManager) -> anyhow::Result<Self> {
        let mut saved = Vec::new();

        for (name, resource) in resources.registered() {
            if resource.settings.save.is_some() {
                saved.push(SavedResource {
                    name: String::from(name),
                    version: resource.save_version(),
                    payload: resource.write_save(resource.erased_ref())?,
                });
            }
        }

        Ok(Self { resources: saved })
    }

    /// Replaces the saveable resources with the values in this set.
    ///
    /// A resource this build does not know, one that cannot be read from a
    /// save, or one whose shape changed is left at its registered value.
    pub fn apply(&self, resources: &mut ResourceManager) {
        for saved in &self.resources {
            let resource = match resources.by_name(&saved.name) {
                Ok(resource) => resource,
                Err(_) => {
                    warn!(
                        "a save names the resource {}, which this build does not have",
                        saved.name
                    );
                    continue;
                }
            };

            if resource.settings.save.is_none() {
                warn!("the resource {} cannot be read from a save", saved.name);
                continue;
            }

            if resource.save_version() != saved.version {
                warn!(
                    "a save holds the resource {} of shape {}, this build reads {}",
                    saved.name,
                    saved.version,
                    resource.save_version()
                );
                continue;
            }

            match resource.read_save(&saved.payload) {
                Ok(value) => {
                    let id = resource.id;

                    match resources.by_id_mut(id) {
                        Ok(resource) => resource.replace(value),
                        Err(e) => error!("failed to restore resource {}: {}", saved.name, e),
                    }
                }
                Err(e) => error!("failed to read resource \"{}\": {}", saved.name, e),
            }
        }
    }
}
