use crate::ecs::*;
use log::error;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use std::any::type_name;
use bimap::BiMap;

/// A system declares what it reads and writes through its queries, which is what
/// allows [`SystemManager`] to run systems that do not conflict in parallel.
pub trait System: 'static + Sync + Send {
    type CompQuery: CompQuery;
    type ResQuery: ResQuery;

    fn update(
        &mut self,
        comp: Self::CompQuery,
        res: Self::ResQuery,
    ) -> Vec<Command> {
        let mut commands = Vec::new();

        let mut res = res.get();
        for entry in comp.iter() {
            if let Some(new_commands) = self.operate(entry, &mut res) {
                commands.extend(new_commands);
            }
        }

        commands
    }

    /// Defaults to doing nothing, so that a system may only override
    /// [`Self::update`] when it does not work per entity.
    fn operate(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        None
    }
}

/// Object safe view of a [`System`], used to store systems of different types
/// together.
trait SystemBridge: 'static + Sync + Send {
    fn access(&self) -> Access;

    fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError>;
}

#[derive(Default)]
pub struct SystemManager {
    stages: Vec<Vec<Box<dyn SystemBridge>>>,
    names: BiMap<TypeId, &'static str>,
}

impl SystemManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<S: System>(&mut self, order: usize, name: &'static str, system: S) {
        if self.stages.len() <= order {
            self.stages.resize_with(order + 1, Vec::new);
        }

        self.stages[order].push(Box::new(system));
        self.names.insert(TypeId::of::<S>(), name);
    }

    /// The type id of the system that was registered under `name`.
    pub fn type_id_of(&self, name: &str) -> Option<TypeId> {
        self.names.get_by_right(name).copied()
    }

    /// Has to be called once after all systems have been registered.
    pub fn init(&mut self) {
        let mut stages = Vec::new();

        for stage in self.stages.iter_mut() {
            while !stage.is_empty() {
                let mut stage1 = Vec::new();
                let mut access = Access::new();
                let mut remaining = Vec::new();

                for system in stage.drain(..) {
                    if access.add(&system.access()) {
                        stage1.push(system);
                    } else {
                        remaining.push(system);
                    }
                }

                *stage = remaining;
                stages.push(stage1);
            }
        }

        self.stages = stages;
    }

    /// The first error of a stage is reported after the stage finished, so that
    /// the systems that could run did run.
    pub fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        let mut commands = Vec::new();
        let mut error = None;

        for stage in self.stages.iter_mut() {
            let results = stage
                .par_iter_mut()
                .map(|system| system.update(entities, resources))
                .collect::<Vec<_>>();

            for result in results {
                match result {
                    Ok(new_commands) => commands.extend(new_commands),

                    Err(e) => error = error.or(Some(e)),
                }
            }
        }

        match error {
            Some(e) => Err(e),

            None => Ok(commands),
        }
    }
}

impl<S: System> SystemBridge for S {
    fn access(&self) -> Access {
        let mut access = S::CompQuery::access();
        if !access.add(&S::ResQuery::access()) {
            error!(
                "system {:?} accesses the same type as both a component and a resource",
                type_name::<S>(),
            );
        }
        access
    }

    fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        let comp = <S as System>::CompQuery::new(&entities.components)?;
        let res = <S as System>::ResQuery::new(resources)?;
        Ok(<S as System>::update(self, comp, res))
    }
}
