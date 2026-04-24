use crate::ecs::component::ComponentManager;
use crate::ecs::resource::ResourceManager;
use crate::ecs::{Component, ErasedComponent, Resource};
use crate::util::Id;
use std::any::TypeId;
use std::collections::HashSet;
use std::iter;
use std::marker::PhantomData;

pub trait CompQuery {
    type Item<'a>;
    
    fn access() -> Access;
    
    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, f: F);
}

pub trait ResQuery {
    type Item<'a>;
    
    fn access() -> Access;
    
    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F);
}

pub trait CompFetch {
    type Item<'a>;
    
    fn add_to(access: &mut Access);
    
    fn new(components: &ComponentManager) -> Self;
    
    fn get<'a>(&self, entity: Id) -> Option<Self::Item<'a>>;
    
    fn iter(components: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)>;
}

pub trait ResFetch {
    type Item<'a>;
    
    fn add_to(access: &mut Access);
    
    fn get(resources: &ResourceManager) -> Self::Item<'_>;
}

#[derive(Default)]
pub struct Access {
    read: HashSet<TypeId>,
    write: HashSet<TypeId>,
}

impl Access {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add(&mut self, other: &Access) -> bool {
        if other.read.intersection(&self.write).next().is_none()
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
    marker: PhantomData<C>,
}

pub struct CompWrite<C: Component> {
    comp: *const ErasedComponent,
    marker: PhantomData<C>,
}

pub struct Without<C: Component> {
    comp: *const ErasedComponent,
    marker: PhantomData<C>,
}

pub struct ResRead<R: Resource>(PhantomData<R>);

pub struct ResWrite<R: Resource>(PhantomData<R>);

impl<C: Component> CompFetch for CompRead<C> {
    type Item<'a> = &'a C;
    
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
        unsafe {
            (&*self.comp).get::<C>(entity)
        }
    }
    
    fn iter(components: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        components.get::<C>().iter()
    }
}

impl<C: Component> CompFetch for CompWrite<C> {
    type Item<'a> = &'a mut C;
    
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
        unsafe {
            (&*self.comp).get_mut::<C>(entity)
        }
    }
    
    fn iter(components: &ComponentManager) -> impl Iterator<Item = (Id, Self::Item<'_>)> {
        components.get::<C>().iter_mut()
    }
}

impl<C: Component> CompFetch for Without<C> {
    type Item<'a> = ();
    
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
    
    fn add_to(access: &mut Access) {
        access.read.insert(TypeId::of::<R>());
    }
    
    fn get(resources: &ResourceManager) -> Self::Item<'_> {
        resources.get::<R>()
    }
}

impl<R: Resource> ResFetch for ResWrite<R> {
    type Item<'a> = &'a mut R;
    
    fn add_to(access: &mut Access) {
        access.write.insert(TypeId::of::<R>());
    }
    
    fn get(resources: &ResourceManager) -> Self::Item<'_> {
        resources.get_mut::<R>()
    }
}

impl CompQuery for () {
    type Item<'a> = ();
    
    fn access() -> Access {
        Access::new()
    }
    
    fn for_each<F: FnMut(Self::Item<'_>)>(_: &ComponentManager, f: F) {
        iter::once(()).for_each(f);
    }
}

impl ResQuery for () {
    type Item<'a> = ();
    
    fn access() -> Access {
        Access::new()
    }
    
    fn run<F: FnOnce(Self::Item<'_>)>(_: &ResourceManager, f: F) {
        f(());
    }
}

impl<C: CompFetch> CompQuery for C {
    type Item<'a> = (Id, C::Item<'a>);
    
    fn access() -> Access {
        let mut access = Access::new();
        C::add_to(&mut access);
        access
    }
    
    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, f: F) {
        C::iter(components).for_each(f);
    }
}

impl<R: ResFetch> ResQuery for R {
    type Item<'a> = R::Item<'a>;
    
    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        access
    }
    
    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f(R::get(resources));
    }
}

impl<C: CompFetch, D: CompFetch> CompQuery for (C, D) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>);
    
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

impl<R: ResFetch, S: ResFetch> ResQuery for (R, S) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>);
    
    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        S::add_to(&mut access);
        access
    }
    
    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(resources),
            S::get(resources),
        ));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch> CompQuery for (C, D, E) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>, E::Item<'a>);
    
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
        unsafe {
            for (i, c) in C::iter(components) {
                if let Some(d) = d.get(i)
                    && let Some(e) = e.get(i)
                {
                    f((i, c, d, e));
                }
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch, T: ResFetch> ResQuery for (R, S, T) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>, T::Item<'a>);
    
    fn access() -> Access {
        let mut access = Access::new();
        R::add_to(&mut access);
        S::add_to(&mut access);
        T::add_to(&mut access);
        access
    }
    
    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(resources),
            S::get(resources),
            T::get(resources),
        ));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch, G: CompFetch> CompQuery for (C, D, E, G) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>, E::Item<'a>, G::Item<'a>);
    
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
        unsafe {
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
}

impl<R: ResFetch, S: ResFetch, T: ResFetch, U: ResFetch> ResQuery for (R, S, T, U) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>, T::Item<'a>, U::Item<'a>);
    
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

impl<C: CompFetch, D: CompFetch, E: CompFetch, G: CompFetch, H: CompFetch> CompQuery for (C, D, E, G, H) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>, E::Item<'a>, G::Item<'a>, H::Item<'a>);
    
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
        unsafe {
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
}

impl<R: ResFetch, S: ResFetch, T: ResFetch, U: ResFetch, V: ResFetch> ResQuery for (R, S, T, U, V) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>, T::Item<'a>, U::Item<'a>, V::Item<'a>);
    
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
