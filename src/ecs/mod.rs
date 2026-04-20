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

pub struct EntityManager {
    entities: SparseSet,
    manager: IdManager,
}

pub struct EntityDescriptor {
    pub values: HashMap<TypeId, ErasedBox>,
}

pub struct ValueDescriptor {
    pub id: TypeId,
    pub value: ErasedBox,
}

pub enum Command {
    Spawn(EntityDescriptor),
    Despawn(Id),
    Insert((Id, ValueDescriptor)),
    Remove((Id, TypeId)),
}

pub struct Simulation {
    pub entities: EntityManager,
    pub components: ComponentManager,
    pub resources: ResourceManager,
    pub systems: SystemManager,
    commands: VecDeque<Command>,
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            entities: EntityManager::new(),
            components: ComponentManager::new(),
            resources: ResourceManager::new(),
            systems: SystemManager::new(),
            commands: VecDeque::new(),
        }
    }

    pub fn update(&mut self) {
        self.flush();
        let new_commands = self.systems.update(&self.components, &self.resources);
        self.commands.extend(new_commands);
    }

    fn flush(&mut self) {
        while let Some(command) = self.commands.pop_back() {
            match command {
                Command::Spawn(desc) => {
                    let entity = self.entities.create();
                    for (id, value) in desc.values.into_iter() {
                        self.components.by_id_mut(id).insert_erased(entity, value);
                    }
                }
                Command::Despawn(entity) => {
                    self.entities.remove(entity);
                    self.components.remove_all(entity);
                }
                Command::Insert((entity, desc)) => {
                    self.components
                        .by_id_mut(desc.id)
                        .insert_erased(entity, desc.value);
                }
                Command::Remove((entity, id)) => {
                    self.components.by_id_mut(id).remove_and_drop(entity);
                }
            }
        }
    }

    pub fn spawn(&mut self, desc: EntityDescriptor) {
        self.commands.push_front(Command::Spawn(desc));
    }

    pub fn despawn(&mut self, entity: Id) {
        self.commands.push_front(Command::Despawn(entity));
    }
}

impl EntityManager {
    pub fn new() -> Self {
        Self {
            entities: SparseSet::new(),
            manager: IdManager::new(),
        }
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
}

impl EntityDescriptor {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }
    
    pub fn with<C: Component>(mut self, value: C) -> Self {
        self.values.insert(TypeId::of::<C>(), ErasedBox::new(value));
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
