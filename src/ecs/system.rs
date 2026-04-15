use crate::ecs::*;
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
	fn update(
		&mut self,
		components: &ComponentManager,
		resources: &ResourceManager,
	) -> Vec<Command> {
		self.update(components, resources)
	}
}
