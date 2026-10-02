mod component;
mod query;
mod resource;
mod system;

use crate::util::collection::SparseSet;
use crate::util::erasure::ErasedBox;
use crate::util::{Id, IdManager};
use anyhow::anyhow;
pub use component::*;
pub use query::*;
pub use resource::*;
use ron::extensions::Extensions;
use ron::value::RawValue;
use ron::Options;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::{HashMap, VecDeque};
use std::sync::LazyLock;
pub use system::*;

static ASSET_OPTIONS: LazyLock<Options> =
    LazyLock::new(|| Options::default().with_default_extension(Extensions::UNWRAP_NEWTYPES));

#[derive(Default)]
pub struct EntityManager {
    entities: SparseSet,
    manager: IdManager,
    pub components: ComponentManager,
    commands: VecDeque<Command>,
}

#[derive(Clone, Default)]
pub struct EntityDescriptor {
    pub values: HashMap<TypeId, ErasedBox>,
}

/// One entity as it is written in a data file.
///
/// The text of a component is only parsed once the component it names is known,
/// so a file may carry components that this build does not read.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawEntity {
    #[serde(default)]
    pub components: Vec<(String, Box<RawValue>)>,
}

pub struct ValueDescriptor {
    pub id: TypeId,
    pub value: ErasedBox,
}

pub enum Command {
    Spawn(EntityDescriptor),
    Despawn(Id),
    Insert(Id, ValueDescriptor),
    Remove(Id, TypeId),
}

impl EntityManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&mut self) -> Id {
        let entity = self.manager.create();
        self.entities.insert(entity);
        entity
    }

    pub fn remove(&mut self, entity: Id) {
        if self.entities.remove(entity) {
            self.manager.recycle(entity);
        }
    }

    /// The queue is drained from the back, so commands queued by [`Self::submit`]
    /// are applied before the ones queued by [`Self::spawn`] and [`Self::despawn`].
    pub fn flush(&mut self) -> Result<(), QueryError> {
        while let Some(command) = self.commands.pop_back() {
            match command {
                Command::Spawn(desc) => {
                    let entity = self.create();

                    for (id, value) in desc.values.into_iter() {
                        self.components.by_id_mut(id)?.insert_erased(entity, value);
                    }
                }
                Command::Despawn(entity) => {
                    self.remove(entity);
                    self.components.remove_all(entity);
                }
                Command::Insert(entity, desc) => {
                    self.components
                        .by_id_mut(desc.id)?
                        .insert_erased(entity, desc.value);
                }
                Command::Remove(entity, id) => {
                    self.components.by_id_mut(id)?.remove_and_drop(entity);
                }
            }
        }

        Ok(())
    }

    pub fn spawn(&mut self, desc: EntityDescriptor) {
        self.commands.push_front(Command::Spawn(desc));
    }

    pub fn despawn(&mut self, entity: Id) {
        self.commands.push_front(Command::Despawn(entity));
    }

    pub fn submit(&mut self, commands: Vec<Command>) {
        self.commands.extend(commands);
    }
}

impl EntityDescriptor {
    pub fn new() -> Self {
        Self::default()
    }

    /// The value is cloned whenever the descriptor is cloned, which is how the
    /// registered actor and screen templates can be spawned repeatedly.
    pub fn with<C: Component>(mut self, value: C) -> Self {
        self.values
            .insert(TypeId::of::<C>(), ErasedBox::new_clone(value));

        self
    }

    pub fn from_raw(components: &ComponentManager, raw: &RawEntity) -> anyhow::Result<Self> {
        let mut new = EntityDescriptor::new();

        for (name, value) in raw.components.iter() {
            let component = components.by_name(name)?;
            let asset = component
                .settings
                .asset
                .ok_or_else(|| anyhow!("component {} cannot be set from data", name))?;

            new.values.insert(component.id, (asset.read)(value)?);
        }

        Ok(new)
    }

    /// The components are ordered by their names, so that the result does not
    /// depend on the order the values happen to be stored in.
    pub fn to_raw(&self, components: &ComponentManager) -> anyhow::Result<RawEntity> {
        let mut entries = Vec::new();

        for (id, value) in self.values.iter() {
            let name = components
                .name_of(*id)
                .ok_or_else(|| anyhow!("component {:?} is not registered", id))?;
            let asset = components
                .by_id(*id)?
                .settings
                .asset
                .ok_or_else(|| anyhow!("component {} cannot be written to data", name))?;

            entries.push((String::from(name), (asset.write)(value)?));
        }

        entries.sort_by(|(left, _), (right, _)| left.cmp(right));

        Ok(RawEntity {
            components: entries,
        })
    }
}

impl ValueDescriptor {
    pub fn new<C: Component>(value: C) -> Self {
        Self {
            id: TypeId::of::<C>(),
            value: ErasedBox::new(value),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Asset {
    pub read: fn(&RawValue) -> anyhow::Result<ErasedBox>,
    pub write: fn(&ErasedBox) -> anyhow::Result<Box<RawValue>>,
}
