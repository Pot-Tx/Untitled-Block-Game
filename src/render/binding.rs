use crate::render::{Canvas, FromConfig};
use std::marker::PhantomData;
use wgpu::*;

pub trait BindRes {
    fn as_resource(&self) -> BindingResource<'_>;
}

pub trait BindContent<'a> {
    fn to_bindings(&self) -> Vec<BindGroupEntry<'a>>;
}

pub trait BindSignature: 'static {
    const NAME: &'static str;
    const LAYOUTS: &'static [BindGroupLayoutEntry];
    type Content<'a>: BindContent<'a>;

    fn layout(canvas: &Canvas) -> BindGroupLayout {
        canvas
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Label::from(format!("{}_bind_group_layout", Self::NAME).as_str()),
                entries: Self::LAYOUTS,
            })
    }
}

pub struct BindSet<S: BindSignature> {
    pub bind_group: BindGroup,
    _marker: PhantomData<S>,
}

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

impl<'a, R: BindRes, S: BindRes> BindContent<'a> for (&'a R, &'a S) {
    fn to_bindings(&self) -> Vec<BindGroupEntry<'a>> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.0.as_resource(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.1.as_resource(),
            },
        ]
    }
}

impl<'a, R: BindRes, S: BindRes, T: BindRes> BindContent<'a> for (&'a R, &'a S, &'a T) {
    fn to_bindings(&self) -> Vec<BindGroupEntry<'a>> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.0.as_resource(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.1.as_resource(),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.2.as_resource(),
            },
        ]
    }
}

impl<'a, R: BindRes, S: BindRes, T: BindRes, U: BindRes> BindContent<'a>
    for (&'a R, &'a S, &'a T, &'a U)
{
    fn to_bindings(&self) -> Vec<BindGroupEntry<'a>> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.0.as_resource(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.1.as_resource(),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.2.as_resource(),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.3.as_resource(),
            },
        ]
    }
}
