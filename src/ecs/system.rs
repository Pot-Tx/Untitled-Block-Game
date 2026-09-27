use crate::ecs::*;
use log::error;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

/// One step of the game loop, which visits every entity that matches its
/// component query.
///
/// A system declares what it reads and writes through its queries, which is what
/// allows [`SystemManager`] to run systems that do not conflict in parallel.
pub trait System: 'static + Sync + Send {
    type CompQuery: CompQuery;
    type ResQuery: ResQuery;

    /// Validates the queries and calls [`Self::operate`] for every matching
    /// entity, collecting the commands the system wants to queue.
    fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        Self::CompQuery::validate(&entities.components)?;
        Self::ResQuery::validate(resources)?;

        let mut commands = Vec::new();

        Self::ResQuery::run(resources, |mut res| {
            Self::CompQuery::for_each(&entities.components, |entry| {
                if let Some(new_commands) = self.operate(entry, &mut res) {
                    commands.extend(new_commands);
                }
            });
        });

        Ok(commands)
    }

    /// Processes one entity, returning the commands it produced.
    ///
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

/// Runs the registered systems, grouped into stages that may run in parallel.
#[derive(Default)]
pub struct SystemManager {
    stages: Vec<Vec<Box<dyn SystemBridge>>>,
}

impl SystemManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues `system` for the stage with the given `order`; lower orders run
    /// first.
    pub fn register<S: System>(&mut self, order: usize, system: S) {
        if self.stages.len() <= order {
            self.stages.resize_with(order + 1, Vec::new);
        }

        self.stages[order].push(Box::new(system));
    }

    /// Splits every stage into sub-stages whose systems do not conflict, so that
    /// they can run in parallel.
    ///
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

    /// Runs every stage in order, executing the systems of a stage in parallel
    /// and collecting the commands they produced.
    ///
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
                TypeId::of::<S>(),
            );
        }
        access
    }

    fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        <S as System>::update(self, entities, resources)
    }
}
