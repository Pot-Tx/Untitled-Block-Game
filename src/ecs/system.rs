use crate::ecs::*;
use log::error;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

pub trait System: 'static + Sync + Send {
    type CompQuery: CompQuery;
    type ResQuery: ResQuery;

    fn update(
        &mut self,
        components: &ComponentManager,
        resources: &ResourceManager,
    ) -> Vec<Command> {
        let mut commands = Vec::new();

        Self::ResQuery::run(resources, |mut res| {
            Self::CompQuery::for_each(components, |entry| {
                if let Some(new_commands) = self.operate(entry, &mut res) {
                    commands.extend(new_commands);
                }
            });
        });

        commands
    }

    fn operate(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'_>,
        _: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        None
    }
}

trait SystemBridge: 'static + Sync + Send {
    fn access(&self) -> Access;
    
    fn update(
        &mut self,
        components: &ComponentManager,
        resources: &ResourceManager,
    ) -> Vec<Command>;
}

#[derive(Default)]
pub struct SystemManager {
    stages: Vec<Vec<Box<dyn SystemBridge>>>,
}

impl SystemManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<S: System>(&mut self, order: usize, system: S) {
        if self.stages.len() <= order {
            self.stages.resize_with(order + 1, Vec::new);
        }

        self.stages[order].push(Box::new(system));
    }
    
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

    pub fn update(
        &mut self,
        components: &ComponentManager,
        resources: &ResourceManager,
    ) -> Vec<Command> {
        let mut commands = Vec::new();

        for stage in self.stages.iter_mut() {
            let new_commands = stage
                .par_iter_mut()
                .map(|system| system.update(components, resources))
                .flatten()
                .collect::<Vec<_>>();

            commands.extend(new_commands);
        }

        commands
    }
}

impl<S: System> SystemBridge for S {
    fn access(&self) -> Access {
        let mut access = S::CompQuery::access();
        if !access.add(&S::ResQuery::access()) {
            error!(
                "System of id {:?}'s CompQuery and ResQuery Access intersects.
                Make sure not to make a type both Component and Resource!",
                TypeId::of::<S>(),
            );
        }
        access
    }
    
    fn update(
        &mut self,
        components: &ComponentManager,
        resources: &ResourceManager,
    ) -> Vec<Command> {
        self.update(components, resources)
    }
}
