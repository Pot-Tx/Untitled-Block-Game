use crate::ecs::*;
use bimap::BiMap;
use log::error;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use std::array;

pub trait CompQueries<const N: usize> {
    fn accesses() -> [Access; N];

    fn new(components: [&ComponentManager; N]) -> Result<Self, QueryError>
    where
        Self: Sized;
}

impl CompQueries<0> for () {
    fn accesses() -> [Access; 0] {
        []
    }

    fn new(_: [&ComponentManager; 0]) -> Result<Self, QueryError> {
        Ok(())
    }
}

macro_rules! impl_queries {
    ($n:literal, $($t:ident),+) => {
        impl<$($t: CompQuery),+> CompQueries<$n> for ($($t,)+) {
            fn accesses() -> [Access; $n] {
                [$($t::access(),)+]
            }

            fn new(components: [&ComponentManager; $n]) -> Result<Self, QueryError> {
                let mut components = components.into_iter();

                Ok(($(
                    <$t as CompQuery>::new(
                        components.next().expect("query count matches component managers"),
                    )?,
                )+))
            }
        }
    };
}

impl_queries!(1, Q);
impl_queries!(2, Q, R);
impl_queries!(3, Q, R, S);

/// A system declares what it reads and writes through its queries, which is what
/// allows [`SystemManager`] to run systems that do not conflict in parallel.
pub trait System<const N: usize>: 'static + Sync + Send {
    type CompQueries: CompQueries<N>;
    type ResQuery: ResQuery;

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        com: [Commander<'_>; N],
    );
}

/// Object safe view of a [`System`], used to store systems of different types
/// together.
trait SystemBridge<const N: usize>: 'static + Sync + Send {
    fn id(&self) -> TypeId;

    fn accesses(&self) -> ([Access; N], Access);

    fn update(
        &mut self,
        entities: [&EntityManager; N],
        resources: &ResourceManager,
    ) -> Result<(), QueryError>;
}

#[derive(Default)]
pub struct SystemManager<const N: usize> {
    stages: Vec<Vec<Box<dyn SystemBridge<N>>>>,
    names: BiMap<TypeId, &'static str>,
}

impl<const N: usize> SystemManager<N> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<S: System<N>>(&mut self, order: usize, name: &'static str, system: S) {
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
        fn add<const N: usize>(a1: &mut ([Access; N], Access), a2: &([Access; N], Access)) -> bool {
            for (a1, a2) in a1.0.iter().zip(a2.0.iter()) {
                if !a1.compats_with(a2) {
                    return false;
                }
            }
            if !a1.1.compats_with(&a2.1) {
                return false;
            }

            for (a1, a2) in a1.0.iter_mut().zip(a2.0.iter()) {
                a1.add(a2);
            }
            a1.1.add(&a2.1);

            true
        }

        let mut stages = Vec::new();

        for stage in self.stages.iter_mut() {
            while !stage.is_empty() {
                let mut stage1 = Vec::new();
                let mut accesses = (array::from_fn(|_| Access::new()), Access::new());
                let mut remaining = Vec::new();

                for system in stage.drain(..) {
                    if add(&mut accesses, &system.accesses()) {
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

    /// A system that fails is logged, and the systems that could still run do.
    pub fn update(&mut self, entities: [&EntityManager; N], resources: &ResourceManager) {
        for stage in self.stages.iter_mut() {
            stage.par_iter_mut().for_each(|system| {
                if let Err(e) = system.update(entities, resources) {
                    let name = *self
                        .names
                        .get_by_left(&system.id())
                        .expect("failed System is not registered");
                    error!("failed to run System {}: {}", name, e);
                }
            });
        }
    }
}

impl<const N: usize, S: System<N>> SystemBridge<N> for S {
    fn id(&self) -> TypeId {
        TypeId::of::<S>()
    }

    fn accesses(&self) -> ([Access; N], Access) {
        (S::CompQueries::accesses(), S::ResQuery::access())
    }

    fn update(
        &mut self,
        entities: [&EntityManager; N],
        resources: &ResourceManager,
    ) -> Result<(), QueryError> {
        let comp = <S as System<N>>::CompQueries::new(entities.map(|e| &e.components))?;
        let res = <S as System<N>>::ResQuery::new(resources)?;
        let com = entities.map(|e| e.commander());
        <S as System<N>>::update(self, comp, res.get(), com);
        Ok(())
    }
}
