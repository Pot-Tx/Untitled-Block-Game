//! Queries describe, at the type level, what a system reads and writes.
//!
//! A [`CompFetch`] is one component access, such as [`CompRead`] or
//! [`OptionalWrite`], and a [`CompQuery`] is a tuple of fetches; resources work
//! the same way through [`ResFetch`] and [`ResQuery`]. Every fetch reports the
//! types it touches through [`Access`], which the scheduler uses to tell
//! whether two systems may run in parallel, and validation checks that the
//! storage a fetch needs has been registered.
//!
//! The fetches hold raw pointers to the storage instead of references, because
//! a system hands out one mutable borrow per entity while the other fetches of
//! the same system keep borrowing the same storage.

use crate::ecs::component::ComponentManager;
use crate::ecs::resource::ResourceManager;
use crate::ecs::{Component, ErasedComponent, Resource};
use crate::util::Id;
use std::any::{type_name, TypeId};
use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::iter;
use std::marker::PhantomData;

/// The error of validating a query, naming the type that is not registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryError {
    MissingComponent(&'static str),
    MissingComponentId(TypeId),
    MissingResource(&'static str),
}

impl QueryError {
    /// The error for a component type that is not registered.
    pub fn component<C: 'static>() -> Self {
        Self::MissingComponent(type_name::<C>())
    }

    /// The error for a resource type that is not registered.
    pub fn resource<R: 'static>() -> Self {
        Self::MissingResource(type_name::<R>())
    }
}

impl Display for QueryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingComponent(name) => write!(f, "component {} is not registered", name),

            Self::MissingComponentId(id) => write!(f, "component {:?} is not registered", id),

            Self::MissingResource(name) => write!(f, "resource {} is not registered", name),
        }
    }
}

impl std::error::Error for QueryError {}

/// A query over the components of the entities a system runs on.
///
/// The first fetch of the query drives the iteration, so it has to be a fetch
/// that iterates entities ([`CompRead`] or [`CompWrite`]); the fetches after it
/// filter or widen the entity that is currently being visited.
pub trait CompQuery {
    type Item<'a>;

    fn access() -> Access;

    fn validate(components: &ComponentManager) -> Result<(), QueryError>;

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, f: F);
}

/// A query over the resources a system runs on.
pub trait ResQuery {
    type Item<'a>;

    fn access() -> Access;

    fn validate(resources: &ResourceManager) -> Result<(), QueryError>;

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F);
}

/// One component access inside a [`CompQuery`].
pub trait CompFetch {
    type Item<'a>;

    fn add_to(access: &mut Access);

    fn validate(components: &ComponentManager) -> Result<(), QueryError>;

    fn new(components: &ComponentManager) -> Self;

    fn get<'a>(&self, entity: Id) -> Option<Self::Item<'a>>;

    fn iter(components: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)>;
}

/// One resource access inside a [`ResQuery`].
pub trait ResFetch {
    type Item<'a>;

    fn add_to(access: &mut Access);

    fn validate(resources: &ResourceManager) -> Result<(), QueryError>;

    fn get(resources: &ResourceManager) -> Self::Item<'_>;
}

/// The component and resource types a fetch reads and writes.
///
/// The scheduler adds the accesses of the systems of a stage together, and
/// starts a new sub-stage whenever they overlap.
#[derive(Default)]
pub struct Access {
    read: HashSet<TypeId>,
    write: HashSet<TypeId>,
}

impl Access {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `other` to this access, returning whether the two overlap.
    pub fn add(&mut self, other: &Access) -> bool {
        if other.read.intersection(&self.write).next().is_none()
            && other.write.intersection(&self.read).next().is_none()
            && other.write.intersection(&self.write).next().is_none()
        {
            self.read.extend(other.read.iter());
            self.write.extend(other.write.iter());
            true
        } else {
            false
        }
    }
}

/// Reads one component of the visited entity.
pub struct CompRead<C: Component> {
    comp: *const ErasedComponent,
    marker: PhantomData<C>,
}

/// Reads and writes one component of the visited entity.
pub struct CompWrite<C: Component> {
    comp: *const ErasedComponent,
    marker: PhantomData<C>,
}

/// Reads a component that the visited entity may not have.
pub struct OptionalRead<C: Component> {
    comp: *const ErasedComponent,
    marker: PhantomData<C>,
}

/// Reads and writes a component that the visited entity may not have.
pub struct OptionalWrite<C: Component> {
    comp: *const ErasedComponent,
    marker: PhantomData<C>,
}

/// Restricts the query to the entities that do not have this component.
pub struct Without<C: Component> {
    comp: *const ErasedComponent,
    marker: PhantomData<C>,
}

/// Reads one resource.
pub struct ResRead<R: Resource>(PhantomData<R>);

/// Reads and writes one resource.
pub struct ResWrite<R: Resource>(PhantomData<R>);

impl<C: Component> CompFetch for CompRead<C> {
    type Item<'a> = &'a C;

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        components.try_get::<C>().map(|_| ())
    }

    fn add_to(access: &mut Access) {
        access.read.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Self {
        Self {
            comp: components.get::<C>(),
            marker: PhantomData,
        }
    }

    fn get<'a>(&self, entity: Id) -> Option<Self::Item<'a>> {
        unsafe { (&*self.comp).get::<C>(entity) }
    }

    fn iter(components: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        components.get::<C>().iter()
    }
}

impl<C: Component> CompFetch for CompWrite<C> {
    type Item<'a> = &'a mut C;

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        components.try_get::<C>().map(|_| ())
    }

    fn add_to(access: &mut Access) {
        access.write.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Self {
        Self {
            comp: components.get::<C>(),
            marker: PhantomData,
        }
    }

    fn get<'a>(&self, entity: Id) -> Option<Self::Item<'a>> {
        unsafe { (&*self.comp).get_mut::<C>(entity) }
    }

    fn iter(components: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        components.get::<C>().iter_mut()
    }
}

impl<C: Component> CompFetch for OptionalRead<C> {
    type Item<'a> = Option<&'a C>;

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        components.try_get::<C>().map(|_| ())
    }

    fn add_to(access: &mut Access) {
        access.read.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Self {
        Self {
            comp: components.get::<C>(),
            marker: PhantomData,
        }
    }

    fn get<'a>(&self, entity: Id) -> Option<Self::Item<'a>> {
        unsafe { Some((&*self.comp).get::<C>(entity)) }
    }

    fn iter(_: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        iter::empty()
    }
}

impl<C: Component> CompFetch for OptionalWrite<C> {
    type Item<'a> = Option<&'a mut C>;

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        components.try_get::<C>().map(|_| ())
    }

    fn add_to(access: &mut Access) {
        access.write.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Self {
        Self {
            comp: components.get::<C>(),
            marker: PhantomData,
        }
    }

    fn get<'a>(&self, entity: Id) -> Option<Self::Item<'a>> {
        unsafe { Some((&*self.comp).get_mut::<C>(entity)) }
    }

    fn iter(_: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        iter::empty()
    }
}

impl<C: Component> CompFetch for Without<C> {
    type Item<'a> = ();

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        components.try_get::<C>().map(|_| ())
    }

    fn add_to(_: &mut Access) {}

    fn new(components: &ComponentManager) -> Self {
        Self {
            comp: components.get::<C>(),
            marker: PhantomData,
        }
    }

    fn get<'a>(&self, entity: Id) -> Option<Self::Item<'a>> {
        unsafe {
            if (&*self.comp).contains(entity) {
                None
            } else {
                Some(())
            }
        }
    }

    fn iter(_: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        iter::empty()
    }
}

impl<R: Resource> ResFetch for ResRead<R> {
    type Item<'a> = &'a R;

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        resources.try_get::<R>().map(|_| ())
    }

    fn add_to(access: &mut Access) {
        access.read.insert(TypeId::of::<R>());
    }

    fn get(resources: &ResourceManager) -> Self::Item<'_> {
        resources.get::<R>()
    }
}

impl<R: Resource> ResFetch for ResWrite<R> {
    type Item<'a> = &'a mut R;

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        resources.try_get::<R>().map(|_| ())
    }

    fn add_to(access: &mut Access) {
        access.write.insert(TypeId::of::<R>());
    }

    fn get(resources: &ResourceManager) -> Self::Item<'_> {
        resources.get_mut::<R>()
    }
}

/// A query without components, which visits the world exactly once.
impl CompQuery for () {
    type Item<'a> = ();

    fn access() -> Access {
        Access::new()
    }

    fn validate(_: &ComponentManager) -> Result<(), QueryError> {
        Ok(())
    }

    fn for_each<F: FnMut(Self::Item<'_>)>(_: &ComponentManager, f: F) {
        iter::once(()).for_each(f);
    }
}

/// A query without resources.
impl ResQuery for () {
    type Item<'a> = ();

    fn access() -> Access {
        Access::new()
    }

    fn validate(_: &ResourceManager) -> Result<(), QueryError> {
        Ok(())
    }

    fn run<F: FnOnce(Self::Item<'_>)>(_: &ResourceManager, f: F) {
        f(());
    }
}

/// The single fetch of a one element query.
impl<C: CompFetch> CompQuery for C {
    type Item<'a> = (Id, C::Item<'a>);

    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        access
    }

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        C::validate(components)
    }

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, f: F) {
        C::iter(components).for_each(f);
    }
}

/// The single fetch of a one element resource query.
impl<R: ResFetch> ResQuery for R {
    type Item<'a> = R::Item<'a>;

    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        access
    }

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        R::validate(resources)
    }

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f(R::get(resources));
    }
}

/// A query of two fetches: the first one drives the iteration and the second is
/// looked up for every entity it yields.
impl<C: CompFetch, D: CompFetch> CompQuery for (C, D) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>);

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        C::validate(components)?;
        D::validate(components)
    }

    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        D::add_to(&mut access);
        access
    }

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = D::new(components);
        for (i, c) in C::iter(components) {
            if let Some(d) = d.get(i) {
                f((i, c, d));
            }
        }
    }
}

/// A resource query of two fetches.
impl<R: ResFetch, S: ResFetch> ResQuery for (R, S) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>);

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        R::validate(resources)?;
        S::validate(resources)
    }

    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        S::add_to(&mut access);
        access
    }

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((R::get(resources), S::get(resources)));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch> CompQuery for (C, D, E) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>, E::Item<'a>);

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        C::validate(components)?;
        D::validate(components)?;
        E::validate(components)
    }

    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        D::add_to(&mut access);
        E::add_to(&mut access);
        access
    }

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = D::new(components);
        let e = E::new(components);
        for (i, c) in C::iter(components) {
            if let Some(d) = d.get(i)
                && let Some(e) = e.get(i)
            {
                f((i, c, d, e));
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch, T: ResFetch> ResQuery for (R, S, T) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>, T::Item<'a>);

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        R::validate(resources)?;
        S::validate(resources)?;
        T::validate(resources)
    }

    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        S::add_to(&mut access);
        T::add_to(&mut access);
        access
    }

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((R::get(resources), S::get(resources), T::get(resources)));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch, G: CompFetch> CompQuery for (C, D, E, G) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>, E::Item<'a>, G::Item<'a>);

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        C::validate(components)?;
        D::validate(components)?;
        E::validate(components)?;
        G::validate(components)
    }

    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        D::add_to(&mut access);
        E::add_to(&mut access);
        G::add_to(&mut access);
        access
    }

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = D::new(components);
        let e = E::new(components);
        let g = G::new(components);
        for (i, c) in C::iter(components) {
            if let Some(d) = d.get(i)
                && let Some(e) = e.get(i)
                && let Some(g) = g.get(i)
            {
                f((i, c, d, e, g));
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch, T: ResFetch, U: ResFetch> ResQuery for (R, S, T, U) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>, T::Item<'a>, U::Item<'a>);

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        R::validate(resources)?;
        S::validate(resources)?;
        T::validate(resources)?;
        U::validate(resources)
    }

    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        S::add_to(&mut access);
        T::add_to(&mut access);
        U::add_to(&mut access);
        access
    }

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(resources),
            S::get(resources),
            T::get(resources),
            U::get(resources),
        ));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch, G: CompFetch, H: CompFetch> CompQuery
    for (C, D, E, G, H)
{
    type Item<'a> = (
        Id,
        C::Item<'a>,
        D::Item<'a>,
        E::Item<'a>,
        G::Item<'a>,
        H::Item<'a>,
    );

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        C::validate(components)?;
        D::validate(components)?;
        E::validate(components)?;
        G::validate(components)?;
        H::validate(components)
    }

    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        D::add_to(&mut access);
        E::add_to(&mut access);
        G::add_to(&mut access);
        H::add_to(&mut access);
        access
    }

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = D::new(components);
        let e = E::new(components);
        let g = G::new(components);
        let h = H::new(components);
        for (i, c) in C::iter(components) {
            if let Some(d) = d.get(i)
                && let Some(e) = e.get(i)
                && let Some(g) = g.get(i)
                && let Some(h) = h.get(i)
            {
                f((i, c, d, e, g, h));
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch, T: ResFetch, U: ResFetch, V: ResFetch> ResQuery for (R, S, T, U, V) {
    type Item<'a> = (
        R::Item<'a>,
        S::Item<'a>,
        T::Item<'a>,
        U::Item<'a>,
        V::Item<'a>,
    );

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        R::validate(resources)?;
        S::validate(resources)?;
        T::validate(resources)?;
        U::validate(resources)?;
        V::validate(resources)
    }

    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        S::add_to(&mut access);
        T::add_to(&mut access);
        U::add_to(&mut access);
        V::add_to(&mut access);
        access
    }

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(resources),
            S::get(resources),
            T::get(resources),
            U::get(resources),
            V::get(resources),
        ));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch, G: CompFetch, H: CompFetch, J: CompFetch> CompQuery
    for (C, D, E, G, H, J)
{
    type Item<'a> = (
        Id,
        C::Item<'a>,
        D::Item<'a>,
        E::Item<'a>,
        G::Item<'a>,
        H::Item<'a>,
        J::Item<'a>,
    );

    fn validate(components: &ComponentManager) -> Result<(), QueryError> {
        C::validate(components)?;
        D::validate(components)?;
        E::validate(components)?;
        G::validate(components)?;
        H::validate(components)?;
        J::validate(components)
    }

    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        D::add_to(&mut access);
        E::add_to(&mut access);
        G::add_to(&mut access);
        H::add_to(&mut access);
        J::add_to(&mut access);
        access
    }

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = D::new(components);
        let e = E::new(components);
        let g = G::new(components);
        let h = H::new(components);
        let j = J::new(components);
        for (i, c) in C::iter(components) {
            if let Some(d) = d.get(i)
                && let Some(e) = e.get(i)
                && let Some(g) = g.get(i)
                && let Some(h) = h.get(i)
                && let Some(j) = j.get(i)
            {
                f((i, c, d, e, g, h, j));
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch, T: ResFetch, U: ResFetch, V: ResFetch, W: ResFetch> ResQuery
    for (R, S, T, U, V, W)
{
    type Item<'a> = (
        R::Item<'a>,
        S::Item<'a>,
        T::Item<'a>,
        U::Item<'a>,
        V::Item<'a>,
        W::Item<'a>,
    );

    fn validate(resources: &ResourceManager) -> Result<(), QueryError> {
        R::validate(resources)?;
        S::validate(resources)?;
        T::validate(resources)?;
        U::validate(resources)?;
        V::validate(resources)?;
        W::validate(resources)
    }

    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        S::add_to(&mut access);
        T::add_to(&mut access);
        U::add_to(&mut access);
        V::add_to(&mut access);
        W::add_to(&mut access);
        access
    }

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(resources),
            S::get(resources),
            T::get(resources),
            U::get(resources),
            V::get(resources),
            W::get(resources),
        ));
    }
}
