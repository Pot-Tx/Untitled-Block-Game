use crate::ecs::component::ComponentManager;
use crate::ecs::resource::ResourceManager;
use crate::ecs::{Component, ErasedComponent, Resource};
use crate::util::erasure::ErasedBox;
use crate::util::Id;
use std::iter;
use std::marker::PhantomData;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

pub trait CompQuery {
    type Item<'a>;

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, f: F);
}

pub trait ResQuery {
    type Item<'a>;

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F);
}

pub trait CompFetch {
    type Item<'a>;
    type Comp<'a>;

    fn bind(components: &ComponentManager) -> Self::Comp<'_>;

    fn get<'a>(component: &'a mut Self::Comp<'_>, entity: Id) -> Option<Self::Item<'a>>;

    fn iter<'a>(component: &'a mut Self::Comp<'_>) -> impl Iterator<Item = (Id, Self::Item<'a>)>;
}

pub trait ResFetch {
    type Item<'a>;
    type Res<'a>;

    fn bind(resources: &ResourceManager) -> Self::Res<'_>;

    fn get<'a>(resource: &'a mut Self::Res<'_>) -> Self::Item<'a>;
}

pub struct CompRead<C: Component>(PhantomData<C>);

pub struct CompWrite<C: Component>(PhantomData<C>);

pub struct ResRead<R: Resource>(PhantomData<R>);

pub struct ResWrite<R: Resource>(PhantomData<R>);

impl<C: Component> CompFetch for CompRead<C> {
    type Item<'a> = &'a C;
    type Comp<'a> = RwLockReadGuard<'a, ErasedComponent>;

    #[inline]
    fn bind(components: &ComponentManager) -> Self::Comp<'_> {
        components.by_type::<C>()
    }

    #[inline]
    fn get<'a>(component: &'a mut Self::Comp<'_>, entity: Id) -> Option<Self::Item<'a>> {
        component.get(entity)
    }

    #[inline]
    fn iter<'a>(component: &'a mut Self::Comp<'_>) -> impl Iterator<Item = (Id, Self::Item<'a>)> {
        component.iter()
    }
}

impl<C: Component> CompFetch for CompWrite<C> {
    type Item<'a> = &'a mut C;
    type Comp<'a> = RwLockWriteGuard<'a, ErasedComponent>;

    #[inline]
    fn bind(components: &ComponentManager) -> Self::Comp<'_> {
        components.by_type_mut::<C>()
    }

    #[inline]
    fn get<'a>(component: &'a mut Self::Comp<'_>, entity: Id) -> Option<Self::Item<'a>> {
        component.get_mut(entity)
    }

    #[inline]
    fn iter<'a>(component: &'a mut Self::Comp<'_>) -> impl Iterator<Item = (Id, Self::Item<'a>)> {
        component.iter_mut()
    }
}

impl<R: Resource> ResFetch for ResRead<R> {
    type Item<'a> = &'a R;
    type Res<'a> = RwLockReadGuard<'a, ErasedBox>;

    #[inline]
    fn bind(resources: &ResourceManager) -> Self::Res<'_> {
        resources.get::<R>()
    }

    #[inline]
    fn get<'a>(resource: &'a mut Self::Res<'_>) -> Self::Item<'a> {
        resource.cast()
    }
}

impl<R: Resource> ResFetch for ResWrite<R> {
    type Item<'a> = &'a mut R;
    type Res<'a> = RwLockWriteGuard<'a, ErasedBox>;

    #[inline]
    fn bind(resources: &ResourceManager) -> Self::Res<'_> {
        resources.get_mut::<R>()
    }

    #[inline]
    fn get<'a>(resource: &'a mut Self::Res<'_>) -> Self::Item<'a> {
        resource.cast_mut()
    }
}

impl CompQuery for () {
    type Item<'a> = ();

    fn for_each<F: FnMut(Self::Item<'_>)>(_: &ComponentManager, f: F) {
        iter::once(()).for_each(f);
    }
}

impl ResQuery for () {
    type Item<'a> = ();

    fn run<F: FnOnce(Self::Item<'_>)>(_: &ResourceManager, f: F) {
        f(());
    }
}

impl<C: CompFetch> CompQuery for C {
    type Item<'a> = (Id, C::Item<'a>);

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, f: F) {
        C::iter(&mut C::bind(components)).for_each(f);
    }
}

impl<R: ResFetch> ResQuery for R {
    type Item<'a> = R::Item<'a>;

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f(R::get(&mut R::bind(resources)));
    }
}

impl<C: CompFetch, D: CompFetch> CompQuery for (C, D) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>);

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = &mut D::bind(components) as *mut _;
        unsafe {
            for (i, c) in C::iter(&mut C::bind(components)) {
                if let Some(d) = D::get(&mut *d, i) {
                    f((i, c, d));
                }
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch> ResQuery for (R, S) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>);

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(&mut R::bind(resources)),
            S::get(&mut S::bind(resources)),
        ));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch> CompQuery for (C, D, E) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>, E::Item<'a>);

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = &mut D::bind(components) as *mut _;
        let e = &mut E::bind(components) as *mut _;
        unsafe {
            for (i, c) in C::iter(&mut C::bind(components)) {
                if let Some(d) = D::get(&mut *d, i)
                    && let Some(e) = E::get(&mut *e, i)
                {
                    f((i, c, d, e));
                }
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch, T: ResFetch> ResQuery for (R, S, T) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>, T::Item<'a>);

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(&mut R::bind(resources)),
            S::get(&mut S::bind(resources)),
            T::get(&mut T::bind(resources)),
        ));
    }
}

impl<C: CompFetch, D: CompFetch, E: CompFetch, G: CompFetch> CompQuery for (C, D, E, G) {
    type Item<'a> = (Id, C::Item<'a>, D::Item<'a>, E::Item<'a>, G::Item<'a>);

    fn for_each<F: FnMut(Self::Item<'_>)>(components: &ComponentManager, mut f: F) {
        let d = &mut D::bind(components) as *mut _;
        let e = &mut E::bind(components) as *mut _;
        let g = &mut G::bind(components) as *mut _;
        unsafe {
            for (i, c) in C::iter(&mut C::bind(components)) {
                if let Some(d) = D::get(&mut *d, i)
                    && let Some(e) = E::get(&mut *e, i)
                    && let Some(g) = G::get(&mut *g, i)
                {
                    f((i, c, d, e, g));
                }
            }
        }
    }
}

impl<R: ResFetch, S: ResFetch, T: ResFetch, U: ResFetch> ResQuery for (R, S, T, U) {
    type Item<'a> = (R::Item<'a>, S::Item<'a>, T::Item<'a>, U::Item<'a>);

    fn run<F: FnOnce(Self::Item<'_>)>(resources: &ResourceManager, f: F) {
        f((
            R::get(&mut R::bind(resources)),
            S::get(&mut S::bind(resources)),
            T::get(&mut T::bind(resources)),
            U::get(&mut U::bind(resources)),
        ));
    }
}
