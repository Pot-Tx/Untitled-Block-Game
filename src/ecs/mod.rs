mod component;
mod query;
mod resource;
mod system;

use crate::util::collection::SparseSet;
use crate::util::erasure::ErasedBox;
use crate::util::{Id, IdManager};
use std::any::TypeId;
use std::collections::{HashMap, VecDeque};

pub use component::*;
pub use query::*;
pub use resource::*;
pub use system::*;

/// Owns the entities, their components and the queue of deferred commands.
#[derive(Default)]
pub struct EntityManager {
    entities: SparseSet,
    manager: IdManager,
    pub components: ComponentManager,
    commands: VecDeque<Command>,
}

/// The set of components an entity is spawned with.
#[derive(Clone, Default)]
pub struct EntityDescriptor {
    pub values: HashMap<TypeId, ErasedBox>,
}

/// Erased value for a deferred write, used by both component inserts and resource writes.
pub struct ValueDescriptor {
    pub id: TypeId,
    pub value: ErasedBox,
}

/// A single deferred change to the entity world.
pub enum Command {
    /// Spawns a new entity with the described components.
    Spawn(EntityDescriptor),
    /// Removes an entity and all of its components.
    Despawn(Id),
    /// Sets one component of an entity.
    Insert((Id, ValueDescriptor)),
    /// Removes one component from an entity.
    Remove((Id, TypeId)),
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

    /// Removes `entity` when it exists and recycles its id.
    pub fn remove(&mut self, entity: Id) {
        if self.entities.remove(entity) {
            self.manager.recycle(entity);
        }
    }

    /// Applies every queued command, returning the first component that turned
    /// out not to be registered.
    ///
    /// The queue is drained from the back, so commands queued by [`Self::submit`]
    /// are applied before the ones queued by [`Self::spawn`] and [`Self::despawn`].
    pub fn flush(&mut self) -> Result<(), QueryError> {
        while let Some(command) = self.commands.pop_back() {
            match command {
                Command::Spawn(desc) => {
                    let entity = self.create();

                    for (id, value) in desc.values.into_iter() {
                        self.components
                            .try_by_id_mut(id)?
                            .insert_erased(entity, value);
                    }
                }
                Command::Despawn(entity) => {
                    self.remove(entity);
                    self.components.remove_all(entity);
                }
                Command::Insert((entity, desc)) => {
                    self.components
                        .try_by_id_mut(desc.id)?
                        .insert_erased(entity, desc.value);
                }
                Command::Remove((entity, id)) => {
                    self.components.try_by_id_mut(id)?.remove_and_drop(entity);
                }
            }
        }

        Ok(())
    }

    /// Queues a spawn of the described entity.
    pub fn spawn(&mut self, desc: EntityDescriptor) {
        self.commands.push_front(Command::Spawn(desc));
    }

    /// Queues the removal of `entity`.
    pub fn despawn(&mut self, entity: Id) {
        self.commands.push_front(Command::Despawn(entity));
    }

    /// Queues every command of `commands`.
    pub fn submit(&mut self, commands: Vec<Command>) {
        self.commands.extend(commands);
    }
}

impl EntityDescriptor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a component to the entity.
    ///
    /// The value is cloned whenever the descriptor is cloned, which is how the
    /// registered actor and screen templates can be spawned repeatedly.
    pub fn with<C: Component + Clone>(mut self, value: C) -> Self {
        self.values
            .insert(TypeId::of::<C>(), ErasedBox::new_clone(value));

        self
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
