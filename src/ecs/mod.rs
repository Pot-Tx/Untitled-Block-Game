mod component;
mod query;
mod resource;
mod system;

use crate::game::ASSET_OPTIONS;
use crate::util::collection::SparseSet;
use crate::util::erasure::{ErasedBox, ErasedRef};
use crate::util::{Id, IdManager};
pub use component::*;
use log::error;
pub use query::*;
pub use resource::*;
use ron::value::RawValue;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};
pub use system::*;

/// The mutable state of an entity manager: the live entities, the id allocator
/// and the queue of commands still to be applied.
#[derive(Default)]
struct Pending {
    live: SparseSet,
    ids: IdManager,
    commands: VecDeque<Command>,
    /// The entities that were despawned since the last [`EntityManager::flush`],
    /// in the order they were despawned. The flush clears it, so it never holds
    /// more than one tick of despawns.
    removed: Vec<Id>,
}

#[derive(Default)]
pub struct EntityManager {
    pub components: ComponentManager,
    pending: Mutex<Pending>,
}

/// A command-only view of an [`EntityManager`], which a system locks to submit a
/// batch of commands.
///
/// A system receives this instead of the manager itself, so that it cannot
/// reach the components it did not declare through its queries.
#[derive(Clone, Copy)]
pub struct Commander<'a> {
    components: &'a ComponentManager,
    pending: &'a Mutex<Pending>,
}

/// The held lock of a [`Commander`], through which the commands are submitted.
pub struct CommandGuard<'a> {
    components: &'a ComponentManager,
    pending: MutexGuard<'a, Pending>,
}

struct ValueDescriptor {
    pub id: TypeId,
    pub value: ErasedBox,
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

enum Command {
    Spawn(Id, EntityDescriptor),
    Despawn(Id),
    Insert(Id, ValueDescriptor),
    Remove(Id, TypeId),
}

impl EntityManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the command-only view of this manager.
    pub fn commander(&self) -> Commander<'_> {
        Commander {
            components: &self.components,
            pending: &self.pending,
        }
    }

    /// Applies the queued commands in the order they were queued: commands are
    /// pushed to the front of the queue and drained from the back.
    pub fn flush(&mut self) {
        let pending = self.pending.get_mut().unwrap();

        // The despawns of the tick that just ended have been read by now, so
        // the log only has to hold the ones of the tick that follows.
        pending.removed.clear();

        let live = &pending.live;
        let commands = &mut pending.commands;

        while let Some(command) = commands.pop_back() {
            match command {
                Command::Spawn(entity, desc) => {
                    for (id, value) in desc.values.into_iter() {
                        match self.components.by_id_mut(id) {
                            Ok(c) => c.insert_erased(entity, value),
                            Err(e) => error!("failed to spawn entity {}: {}", entity, e),
                        }
                    }
                }

                Command::Despawn(entity) => {
                    self.components.remove_all(entity);
                }

                Command::Insert(entity, desc) => {
                    if live.contains(entity) {
                        match self.components.by_id_mut(desc.id) {
                            Ok(c) => c.insert_erased(entity, desc.value),
                            Err(e) => error!("failed to insert component: {}", e),
                        }
                    }
                }

                Command::Remove(entity, id) => {
                    if live.contains(entity) {
                        match self.components.by_id_mut(id) {
                            Ok(c) => c.remove_and_drop(entity),
                            Err(e) => error!("failed to remove component: {}", e),
                        }
                    }
                }
            }
        }
    }
}

impl<'a> Commander<'a> {
    /// Locks the manager to submit a batch of commands.
    pub fn lock(&self) -> CommandGuard<'a> {
        CommandGuard {
            components: self.components,
            pending: self.pending.lock().unwrap(),
        }
    }
}

impl CommandGuard<'_> {
    #[inline]
    pub fn spawn(&mut self, desc: EntityDescriptor) -> Id {
        let entity = self.pending.ids.create();
        self.pending.live.insert(entity);
        self.pending
            .commands
            .push_front(Command::Spawn(entity, desc));
        entity
    }

    #[inline]
    pub fn despawn(&mut self, entity: Id) -> bool {
        if self.pending.live.remove(entity) {
            self.pending.ids.recycle(entity);
            self.pending.removed.push(entity);
            self.pending.commands.push_front(Command::Despawn(entity));
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn insert<C: Component>(&mut self, entity: Id, value: C) -> bool {
        if self.pending.live.contains(entity) && self.components.get::<C>().is_ok() {
            let value = ValueDescriptor::new(value);
            self.pending
                .commands
                .push_front(Command::Insert(entity, value));
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn remove<C: Component>(&mut self, entity: Id) -> bool {
        let present = self.pending.live.contains(entity)
            && self.components.get::<C>().is_ok_and(|c| c.contains(entity));

        if present {
            self.pending
                .commands
                .push_front(Command::Remove(entity, TypeId::of::<C>()));
            true
        } else {
            false
        }
    }

    /// Whether `entity` is still alive.
    ///
    /// The components of a despawned entity outlive it until the next flush, so
    /// a reader that walks the component storages has to ask this to tell the
    /// entities of this tick apart from the ones that just died.
    #[inline]
    pub fn contains(&self, entity: Id) -> bool {
        self.pending.live.contains(entity)
    }

    /// The entities that were despawned since the last flush, in the order they
    /// were despawned.
    ///
    /// The flush clears the log, so a reader has to look at it between two
    /// flushes to see every despawn.
    #[inline]
    pub fn removed(&self) -> &[Id] {
        &self.pending.removed
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

            new.values
                .insert(component.id, component.read_asset(value)?);
        }

        Ok(new)
    }

    /// The components are ordered by their names, so that the result does not
    /// depend on the order the values happen to be stored in.
    pub fn to_raw(&self, components: &ComponentManager) -> anyhow::Result<RawEntity> {
        let mut entries = Vec::new();

        for (id, value) in self.values.iter() {
            let component = components.by_id(*id)?;

            entries.push((
                String::from(component.name),
                component.write_asset(value.as_ref())?,
            ));
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
    pub write: fn(ErasedRef<'_>) -> anyhow::Result<Box<RawValue>>,
}

/// How one component is read from and written to a save.
///
/// It mirrors [`Asset`], which drives the text data files, but a save holds a
/// binary payload that only the game writes and reads.
#[derive(Clone, Copy, Debug)]
pub struct Codec {
    /// The version of the shape the payload is written with.
    pub version: u8,
    /// Decodes one value from the payload the container stored for it.
    pub read: fn(&[u8]) -> anyhow::Result<ErasedBox>,
    /// Encodes one value into the payload the container stores for it.
    pub write: fn(ErasedRef<'_>) -> anyhow::Result<Vec<u8>>,
}

/// The text codec of a value that may be written by hand.
pub const fn asset_of<T: Clone + DeserializeOwned + Serialize + 'static>() -> Asset {
    Asset {
        read: |raw| {
            Ok(ErasedBox::new_clone(
                ASSET_OPTIONS.from_str::<T>(raw.get_ron())?,
            ))
        },
        write: |value| {
            Ok(RawValue::from_boxed_ron(
                ASSET_OPTIONS.to_string(value.cast::<T>())?.into_boxed_str(),
            )?)
        },
    }
}

/// The binary codec of a value that is written to a save.
pub const fn codec_of<T: Clone + DeserializeOwned + Serialize + 'static>(version: u8) -> Codec {
    Codec {
        version,
        read: |payload| {
            let value = postcard::from_bytes::<T>(payload)
                .map_err(|e| anyhow::anyhow!("failed to decode a value: {e}"))?;

            Ok(ErasedBox::new_clone(value))
        },
        write: |value| {
            postcard::to_stdvec(value.cast::<T>())
                .map_err(|e| anyhow::anyhow!("failed to encode a value: {e}"))
        },
    }
}
