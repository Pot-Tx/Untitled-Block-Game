//! Queries describe, at the type level, what a system reads and writes.
//!
//! A [`CompFetch`] is one component access, such as [`CompRead`] or
//! [`OptionalWrite`], and a [`CompQuery`] is a tuple of fetches; resources work
//! the same way through [`ResFetch`] and [`ResQuery`]. Every fetch reports the
//! types it touches through [`Access`], which the scheduler uses to tell
//! whether two systems may run in parallel, and every lookup fails with a
//! [`QueryError`] when the storage a fetch needs has not been registered.
//!
//! The fetches hold raw pointers to the storage instead of references, because
//! a system hands out one mutable borrow per entity while the other fetches of
//! the same system keep borrowing the same storage.

use crate::ecs::component::ComponentManager;
use crate::ecs::resource::ResourceManager;
use crate::ecs::{Component, ErasedComponent, ErasedResource, Resource};
use crate::util::Id;
use std::any::TypeId;
use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::iter;
use std::marker::PhantomData;

#[derive(Debug, PartialEq, Eq)]
pub enum QueryError {
    MissingComponent(&'static str),
    MissingComponentId(TypeId),
    MissingComponentName(String),
    MissingResource(&'static str),
    MissingResourceId(TypeId),
    MissingResourceName(String),
}

impl Display for QueryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingComponent(name) => write!(f, "component {} is not registered", name),

            Self::MissingComponentId(id) => write!(f, "component {:?} is not registered", id),
            
            Self::MissingComponentName(name) => write!(f, "component name \"{}\" is not registered", name),

            Self::MissingResource(name) => write!(f, "resource {} is not registered", name),
            
            Self::MissingResourceId(id) => write!(f, "resource {:?} is not registered", id),
            
            Self::MissingResourceName(name) => write!(f, "resource name \"{}\" is not registered", name),
        }
    }
}

impl std::error::Error for QueryError {}

/// The first fetch of the query drives the iteration, so it has to be a fetch
/// that iterates entities ([`CompRead`] or [`CompWrite`]); the fetches after it
/// filter or widen the entity that is currently being visited.
pub trait CompQuery {
    type Item<'a> where Self: 'a;

    fn access() -> Access;
    
    fn new(components: &ComponentManager) -> Result<Self, QueryError>
    where
        Self: Sized; 
    
    fn iter(&self) -> impl Iterator<Item = Self::Item<'_>>;
}

pub trait ResQuery {
    type Item<'a> where Self: 'a;

    fn access() -> Access;
    
    fn new(resources: &ResourceManager) -> Result<Self, QueryError>
    where
        Self: Sized;
    
    fn get(&self) -> Self::Item<'_>;
}

pub trait CompFetch {
    type Item<'a> where Self: 'a;

    fn add_to(access: &mut Access);

    fn new(components: &ComponentManager) -> Result<Self, QueryError>
    where
        Self: Sized;

    fn get(&self, entity: Id) -> Option<Self::Item<'_>>;

    fn iter(&self) -> impl Iterator<Item = (Id, Self::Item<'_>)>;
}

pub trait ResFetch {
    type Item<'a> where Self: 'a;

    fn add_to(access: &mut Access);
    
    fn new(resources: &ResourceManager) -> Result<Self, QueryError>
    where
        Self: Sized;

    fn get(&self) -> Self::Item<'_>;
}

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

pub struct CompRead<C: Component> {
    comp: *const ErasedComponent,
    _marker: PhantomData<C>,
}

pub struct CompWrite<C: Component> {
    comp: *const ErasedComponent,
    _marker: PhantomData<C>,
}

pub struct OptionalRead<C: Component> {
    comp: *const ErasedComponent,
    _marker: PhantomData<C>,
}

pub struct OptionalWrite<C: Component> {
    comp: *const ErasedComponent,
    _marker: PhantomData<C>,
}

pub struct Without<C: Component> {
    comp: *const ErasedComponent,
    _marker: PhantomData<C>,
}

pub struct ResRead<R: Resource> {
    res: *const ErasedResource,
    _marker: PhantomData<R>,
}

pub struct ResWrite<R: Resource> {
    res: *const ErasedResource,
    _marker: PhantomData<R>,
}

impl<C: Component> CompFetch for CompRead<C> {
    type Item<'a> = &'a C;

    fn add_to(access: &mut Access) {
        access.read.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Result<Self, QueryError> {
        Ok(Self {
            comp: components.get::<C>()?,
            _marker: PhantomData,
        })
    }

    fn get(&self, entity: Id) -> Option<Self::Item<'_>> {
        unsafe { (&*self.comp).get::<C>(entity) }
    }

    fn iter(&self) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        unsafe { (&*self.comp).iter() }
    }
}

impl<C: Component> CompFetch for CompWrite<C> {
    type Item<'a> = &'a mut C;

    fn add_to(access: &mut Access) {
        access.write.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Result<Self, QueryError> {
        Ok(Self {
            comp: components.get::<C>()?,
            _marker: PhantomData,
        })
    }

    fn get(&self, entity: Id) -> Option<Self::Item<'_>> {
        unsafe { (&*self.comp).get_mut::<C>(entity) }
    }
    
    fn iter(&self) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        unsafe { (&*self.comp).iter_mut() }
    }
}

impl<C: Component> CompFetch for OptionalRead<C> {
    type Item<'a> = Option<&'a C>;

    fn add_to(access: &mut Access) {
        access.read.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Result<Self, QueryError> {
        Ok(Self {
            comp: components.get::<C>()?,
            _marker: PhantomData,
        })
    }

    fn get(&self, entity: Id) -> Option<Self::Item<'_>> {
        unsafe { Some((&*self.comp).get::<C>(entity)) }
    }
    
    fn iter(&self) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        iter::empty()
    }
}

impl<C: Component> CompFetch for OptionalWrite<C> {
    type Item<'a> = Option<&'a mut C>;

    fn add_to(access: &mut Access) {
        access.write.insert(TypeId::of::<C>());
    }

    fn new(components: &ComponentManager) -> Result<Self, QueryError> {
        Ok(Self {
            comp: components.get::<C>()?,
            _marker: PhantomData,
        })
    }

    fn get(&self, entity: Id) -> Option<Self::Item<'_>> {
        unsafe { Some((&*self.comp).get_mut::<C>(entity)) }
    }
    
    fn iter(&self) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        iter::empty()
    }
}

impl<C: Component> CompFetch for Without<C> {
    type Item<'a> = ();

    fn add_to(_: &mut Access) {}

    fn new(components: &ComponentManager) -> Result<Self, QueryError> {
        Ok(Self {
            comp: components.get::<C>()?,
            _marker: PhantomData,
        })
    }

    fn get(&self, entity: Id) -> Option<Self::Item<'_>> {
        unsafe {
            if (&*self.comp).contains(entity) {
                None
            } else {
                Some(())
            }
        }
    }
    
    fn iter(&self) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        iter::empty()
    }
}

impl<R: Resource> ResFetch for ResRead<R> {
    type Item<'a> = &'a R;

    fn add_to(access: &mut Access) {
        access.read.insert(TypeId::of::<R>());
    }
    
    fn new(resources: &ResourceManager) -> Result<Self, QueryError> {
        Ok(Self {
            res: resources.get::<R>()?,
            _marker: PhantomData,
        })
    }

    fn get(&self) -> Self::Item<'_> {
        unsafe { (&*self.res).get() }
    }
}

impl<R: Resource> ResFetch for ResWrite<R> {
    type Item<'a> = &'a mut R;

    fn add_to(access: &mut Access) {
        access.write.insert(TypeId::of::<R>());
    }
    
    fn new(resources: &ResourceManager) -> Result<Self, QueryError> {
        Ok(Self {
            res: resources.get::<R>()?,
            _marker: PhantomData,
        })
    }
    
    fn get(&self) -> Self::Item<'_> {
        unsafe { (&*self.res).get_mut() }
    }
}

/// A query without components, which visits the world exactly once.
impl CompQuery for () {
    type Item<'a> = ();

    fn access() -> Access {
        Access::new()
    }
    
    fn new(_: &ComponentManager) -> Result<Self, QueryError> {
        Ok(())
    }
    
    fn iter(&self) -> impl Iterator<Item = Self::Item<'_>> {
        iter::once(())
    }
}

impl ResQuery for () {
    type Item<'a> = ();

    fn access() -> Access {
        Access::new()
    }
    
    fn new(_: &ResourceManager) -> Result<Self, QueryError> {
        Ok(())
    }
    
    fn get(&self) -> Self::Item<'_> {
        ()
    }
}

impl<C: CompFetch> CompQuery for C {
    type Item<'a> = (Id, C::Item<'a>) where C: 'a;

    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        access
    }
    
    fn new(components: &ComponentManager) -> Result<Self, QueryError> {
        C::new(components)
    }
    
    fn iter(&self) -> impl Iterator<Item = Self::Item<'_>> {
        self.iter()
    }

}

impl<R: ResFetch> ResQuery for R {
    type Item<'a> = R::Item<'a> where R: 'a;

    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        access
    }
    
    fn new(resources: &ResourceManager) -> Result<Self, QueryError> {
        R::new(resources)
    }
    
    fn get(&self) -> Self::Item<'_> {
        self.get()
    }
}

/// The component arm lets the first fetch drive the iteration and looks the
/// later ones up for every entity it yields; the resource arm fetches every
/// value before it hands them over.
///
/// Every fetch is written as the pair of its type and the names its fetch and
/// its item are bound to, because a tuple element cannot be addressed by an
/// index from within a `macro_rules!` body.
macro_rules! impl_query {
    (comp $head:ident($head_fetch:ident, $head_item:ident)
        $(, $tail:ident($tail_fetch:ident, $tail_item:ident))*) => {
        impl<$head: CompFetch, $($tail: CompFetch),*> CompQuery for ($head, $($tail,)*) {
            type Item<'a> = (Id, $head::Item<'a>, $($tail::Item<'a>,)*) where Self: 'a;

            fn access() -> Access {
                let mut access = Access::new();
                $head::add_to(&mut access);
                $($tail::add_to(&mut access);)*
                access
            }

            fn new(components: &ComponentManager) -> Result<Self, QueryError> {
                Ok(($head::new(components)?, $($tail::new(components)?,)*))
            }

            fn iter(&self) -> impl Iterator<Item = Self::Item<'_>> {
                let ($head_fetch, $($tail_fetch,)*) = self;

                $head_fetch.iter().filter_map(move |(id, $head_item)| {
                    $(let $tail_item = $tail_fetch.get(id)?;)*

                    Some((id, $head_item, $($tail_item,)*))
                })
            }
        }
    };

    (res $head:ident($head_fetch:ident) $(, $tail:ident($tail_fetch:ident))*) => {
        impl<$head: ResFetch, $($tail: ResFetch),*> ResQuery for ($head, $($tail,)*) {
            type Item<'a> = ($head::Item<'a>, $($tail::Item<'a>,)*) where Self: 'a;

            fn access() -> Access {
                let mut access = Access::new();
                $head::add_to(&mut access);
                $($tail::add_to(&mut access);)*
                access
            }

            fn new(resources: &ResourceManager) -> Result<Self, QueryError> {
                Ok(($head::new(resources)?, $($tail::new(resources)?,)*))
            }

            fn get(&self) -> Self::Item<'_> {
                let ($head_fetch, $($tail_fetch,)*) = self;

                ($head_fetch.get(), $($tail_fetch.get(),)*)
            }
        }
    };
}

impl_query!(comp C(cf, cv));
impl_query!(comp C(cf, cv), D(df, dv));
impl_query!(comp C(cf, cv), D(df, dv), E(ef, ev));
impl_query!(comp C(cf, cv), D(df, dv), E(ef, ev), G(gf, gv));
impl_query!(comp C(cf, cv), D(df, dv), E(ef, ev), G(gf, gv), H(hf, hv));
impl_query!(comp C(cf, cv), D(df, dv), E(ef, ev), G(gf, gv), H(hf, hv), J(jf, jv));

impl_query!(res R(rf));
impl_query!(res R(rf), S(sf));
impl_query!(res R(rf), S(sf), T(tf));
impl_query!(res R(rf), S(sf), T(tf), U(uf));
impl_query!(res R(rf), S(sf), T(tf), U(uf), V(vf));
impl_query!(res R(rf), S(sf), T(tf), U(uf), V(vf), W(wf));
