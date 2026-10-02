use crate::render::{Canvas, FromConfig};
use std::marker::PhantomData;
use wgpu::*;

/// A `wgpu` object that can be bound to a shader binding.
pub trait BindRes {
    fn as_resource(&self) -> BindingResource<'_>;
}

/// The bind group entries of a bind set, in binding order.
pub trait BindContent<'a> {
    fn to_bindings(&self) -> Vec<BindGroupEntry<'a>>;
}

/// The layout and the content of one bind group.
///
/// Implemented for single resources and for tuples of them, so that a bind
/// group can hold any number of bindings; the position in the tuple is the
/// shader binding index.
pub trait BindSignature: 'static {
    const NAME: &'static str;
    /// The entries of the bind group layout, in the same order as the content.
    const LAYOUTS: &'static [BindGroupLayoutEntry];
    type Content<'a>: BindContent<'a>;

    /// Creates the bind group layout on `canvas`.
    fn layout(canvas: &Canvas) -> BindGroupLayout {
        canvas
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Label::from(format!("{}_bind_group_layout", Self::NAME).as_str()),
                entries: Self::LAYOUTS,
            })
    }
}

/// An allocated bind group, created from the content of a [`BindSignature`].
pub struct BindSet<S: BindSignature> {
    pub bind_group: BindGroup,
    _marker: PhantomData<S>,
}

/// The name and content of a bind group to create.
pub struct BindSetConfig<'a, S: BindSignature> {
    pub name: &'a str,
    pub content: S::Content<'a>,
}

impl<S: BindSignature> FromConfig<BindSetConfig<'_, S>> for BindSet<S> {
    type Base = Canvas;

    fn new(base: &Self::Base, config: &BindSetConfig<S>) -> Self {
        Self {
            bind_group: base.device.create_bind_group(&BindGroupDescriptor {
                label: Label::from(format!("{}_{}_bind_group", config.name, S::NAME).as_str()),
                layout: &S::layout(base),
                entries: &config.content.to_bindings(),
            }),
            _marker: Default::default(),
        }
    }
}

impl BindRes for TextureView {
    fn as_resource(&self) -> BindingResource<'_> {
        BindingResource::TextureView(self)
    }
}

impl BindRes for Sampler {
    fn as_resource(&self) -> BindingResource<'_> {
        BindingResource::Sampler(self)
    }
}

impl BindRes for Buffer {
    fn as_resource(&self) -> BindingResource<'_> {
        self.as_entire_binding()
    }
}

impl<'a, R: BindRes> BindContent<'a> for &'a R {
    fn to_bindings(&self) -> Vec<BindGroupEntry<'a>> {
        vec![BindGroupEntry {
            binding: 0,
            resource: self.as_resource(),
        }]
    }
}

/// Implements [`BindContent`] for a tuple of resources, which become the
/// bindings `0` to `n - 1` of the bind group.
///
/// Every resource is written as the pair of its type and the name its value is
/// bound to.
macro_rules! impl_bind_content {
    ($($res:ident($var:ident)),+ $(,)?) => {
        impl<'a, $($res: BindRes),+> BindContent<'a> for ($(&'a $res,)+) {
            fn to_bindings(&self) -> Vec<BindGroupEntry<'a>> {
                let ($($var,)+) = *self;

                [$($var.as_resource(),)+]
                    .into_iter()
                    .enumerate()
                    .map(|(binding, resource)| BindGroupEntry {
                        binding: binding as u32,
                        resource,
                    })
                    .collect()
            }
        }
    };
}

impl_bind_content!(R(r), S(s));
impl_bind_content!(R(r), S(s), T(t));
impl_bind_content!(R(r), S(s), T(t), U(u));
