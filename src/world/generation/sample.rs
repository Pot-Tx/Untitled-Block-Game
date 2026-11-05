use crate::util::collection::Volume;
use crate::util::coord::Coord3;
use crate::world::region::{LocalPos, REGION_SIZE, Region, RelPos};
use glam::{U8Vec3, Vec3};
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;

/// The number of floats one sample may hold. The manifest rejects a layout that
/// asks for more, so a sample always fits on the stack.
pub const MAX_SAMPLE_FLOATS: usize = 32;
/// Number of blocks between two samples of a grid.
pub const SAMPLE_INTERVAL: u8 = 8;
/// Number of samples along one axis of a grid, without its halo.
pub const GRID_SIZE: u8 = REGION_SIZE / SAMPLE_INTERVAL;
/// Number of samples along one axis of a grid, including the halo that the
/// gradient needs.
pub const GRID_STRIDE: u8 = GRID_SIZE + 1;

/// One named run of components inside a sample.
#[derive(Clone, Copy, Debug)]
pub struct Channel {
    pub offset: usize,
    pub components: u8,
}

/// The channels of a sample, in the order their components are stored.
pub struct SampleLayout {
    channels: Vec<(String, Channel)>,
    stride: usize,
    halo: Option<HaloChannels>,
    gradients: Vec<(Channel, Channel)>,
}

/// The two channels the halo redirection reads.
#[derive(Clone, Copy, Debug)]
pub struct HaloChannels {
    pub density: Channel,
    pub erosion: Channel,
}

impl SampleLayout {
    /// The layout of the generator before the world program overrides it.
    pub fn fixed() -> Self {
        let mut layout = Self {
            channels: Vec::new(),
            stride: 0,
            halo: None,
            gradients: Vec::new(),
        };

        let climate = layout.push("climate", 3);
        let density = layout.push("density", 1);
        let erosion = layout.push("erosion", 1);
        let gradient = layout.push("density_gradient", 3);
        let _ = (climate, gradient);
        layout.halo = Some(HaloChannels { density, erosion });
        layout.gradients.push((gradient, density));
        layout
    }

    /// Builds a layout from a schema, rejecting channels that do not fit the
    /// sample or gradients of channels that are not scalars.
    pub fn new(
        channels: impl IntoIterator<Item = (String, u8)>,
        halo: Option<(String, String)>,
        gradients: impl IntoIterator<Item = (String, String)>,
    ) -> anyhow::Result<Self> {
        let mut layout = Self {
            channels: Vec::new(),
            stride: 0,
            halo: None,
            gradients: Vec::new(),
        };
        let gradients = gradients.into_iter().collect::<Vec<_>>();

        for (name, components) in channels {
            if components == 0 {
                anyhow::bail!("channel {name} has no components");
            }

            layout.push(&name, components);
        }

        for (name, source_name) in &gradients {
            let source = layout
                .channel(source_name)
                .ok_or_else(|| anyhow::anyhow!("gradient channel {name} has no source"))?;
            if source.components != 1 {
                anyhow::bail!("gradient channel {name} needs a scalar source");
            }
            let derived = layout.push(name, 3);
            layout.gradients.push((derived, source));
        }

        if layout.stride > MAX_SAMPLE_FLOATS {
            anyhow::bail!(
                "a sample holds {} floats, but at most {MAX_SAMPLE_FLOATS} are supported",
                layout.stride
            );
        }

        layout.halo = match halo {
            Some((density, erosion)) => Some(HaloChannels {
                density: layout
                    .channel(&density)
                    .ok_or_else(|| anyhow::anyhow!("halo channel {density} is missing"))?,
                erosion: layout
                    .channel(&erosion)
                    .ok_or_else(|| anyhow::anyhow!("halo channel {erosion} is missing"))?,
            }),
            None => None,
        };

        Ok(layout)
    }

    fn push(&mut self, name: &str, components: u8) -> Channel {
        let channel = Channel {
            offset: self.stride,
            components,
        };
        self.stride += components as usize;
        self.channels.push((name.to_owned(), channel));
        channel
    }

    #[inline]
    pub fn channel(&self, name: &str) -> Option<Channel> {
        self.channels
            .iter()
            .find(|(entry, _)| entry == name)
            .map(|(_, channel)| *channel)
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    pub fn halo(&self) -> Option<HaloChannels> {
        self.halo
    }

    pub fn channels(&self) -> &[(String, Channel)] {
        &self.channels
    }

    pub fn gradients(&self) -> &[(Channel, Channel)] {
        &self.gradients
    }
}

/// The samples of one region, stored channel interleaved.
pub struct SampleGrid {
    pub size: U8Vec3,
    pub data: Vec<f32>,
}

impl SampleGrid {
    pub fn new(size: U8Vec3, stride: usize) -> Self {
        Self {
            size,
            data: vec![0.0; size.as_usizevec3().element_product() * stride],
        }
    }

    #[inline]
    pub fn values(&self, pos: U8Vec3, stride: usize) -> &[f32] {
        let start = self.idx_of_pos(pos) * stride;
        &self.data[start..start + stride]
    }

    #[inline]
    pub fn values_mut(&mut self, pos: U8Vec3, stride: usize) -> &mut [f32] {
        let start = self.idx_of_pos(pos) * stride;
        &mut self.data[start..start + stride]
    }

    #[inline]
    fn idx_of_pos(&self, pos: U8Vec3) -> usize {
        let h = self.size.y as usize;
        let d = self.size.z as usize;
        pos.x as usize * h * d + pos.y as usize * d + pos.z as usize
    }

    pub fn part(&self, min: U8Vec3, size: U8Vec3, stride: usize) -> Self {
        let mut part = Self::new(size, stride);
        let [_, h, d] = self.size.as_usizevec3().to_array();
        let [sw, sh, sd] = size.as_usizevec3().to_array();
        let [dx, dy, dz] = min.as_usizevec3().to_array();

        for x in 0..sw {
            for y in 0..sh {
                let src = (x + dx) * h * d + (y + dy) * d + dz;
                let dst = x * sh * sd + y * sd;
                let len = sd * stride;
                part.data[dst * stride..(dst + sd) * stride]
                    .copy_from_slice(&self.data[src * stride..src * stride + len]);
            }
        }

        part
    }
}

/// A sample of the field, with the layout that gives its components meaning.
#[derive(Clone)]
pub struct Sample<'a> {
    values: SmallVec<[f32; MAX_SAMPLE_FLOATS]>,
    layout: &'a SampleLayout,
}

impl<'a> Sample<'a> {
    pub fn new(layout: &'a SampleLayout) -> Self {
        Self {
            values: smallvec![0.0; layout.stride()],
            layout,
        }
    }

    pub fn empty(layout: &'a SampleLayout) -> Self {
        Self::new(layout)
    }

    #[inline]
    pub fn channel(&self, name: &str) -> Option<Channel> {
        self.layout.channel(name)
    }

    #[inline]
    pub fn scalar(&self, channel: Channel) -> f32 {
        self.values[channel.offset]
    }

    #[inline]
    pub fn component(&self, channel: Channel, component: usize) -> f32 {
        self.values[channel.offset + component]
    }

    #[inline]
    pub fn vector(&self, channel: Channel) -> Vec3 {
        Vec3::new(
            self.values[channel.offset],
            self.values[channel.offset + 1],
            self.values[channel.offset + 2],
        )
    }

    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    #[inline]
    pub fn layout(&self) -> &'a SampleLayout {
        self.layout
    }
}

/// The grid neighborhood a task reads, together with the task geometry that
/// turns a chunk-local position into a grid position.
pub struct SampleGridView {
    pub(crate) grids: Volume<Arc<SampleGrid>>,
    pub(crate) layout: Arc<SampleLayout>,
    pub(crate) lod: u8,
    /// The region of the first grid in the view.
    pub(crate) base: crate::world::RegionPos,
    /// The block offset of chunk-local (0, 0, 0) from the origin of the first
    /// region of the view.
    pub(crate) chunk_origin: RelPos,
    /// The number of blocks along one axis of the chunk. Zero means the view is
    /// not bound to a chunk and never asks for a halo sample.
    pub(crate) stride: u8,
}

impl SampleGridView {
    /// Samples the field at `local`, which is chunk-local, by interpolating the
    /// surrounding grid.
    ///
    /// `lod` widens the distance between the corners that are read, and a local
    /// position on the border of the chunk redirects the sample towards the
    /// corner of the cell with the lowest density, so that the surface of a
    /// coarse block keeps following the fine geometry it replaces.
    pub fn interpolate(&self, local: LocalPos) -> Sample<'_> {
        let block_size = Region::block_size_on_lod(self.lod);
        let pos = self.chunk_origin + local * block_size;
        let halo = self.stride != 0
            && (local.min_element() == 0 || local.max_element() == self.stride - 1);
        let stride = self.layout.stride();

        let origin = pos / SAMPLE_INTERVAL;
        let offset = (block_size / SAMPLE_INTERVAL).max(1);
        let corners = U8Vec3::corners(U8Vec3::ZERO, U8Vec3::ONE);
        let samples = corners.map(|corner| {
            let pos = origin + corner * offset;
            let grid_pos = (pos / GRID_SIZE).min(self.grids.size - 1);
            let pos = pos - grid_pos * GRID_SIZE;
            self.grids.get(grid_pos).values(pos, stride)
        });

        let redirection = match (halo, self.layout.halo()) {
            (true, Some(halo)) => {
                let mut min = 1.0;
                let mut min_corner = U8Vec3::ZERO;

                for (&corner, sample) in corners.iter().zip(samples.iter()) {
                    let d = sample[halo.density.offset].min(-sample[halo.erosion.offset]);
                    if d < min {
                        min = d;
                        min_corner = corner;
                    }
                }

                min_corner * (block_size - 1)
            }
            _ => U8Vec3::splat(block_size / 2),
        };
        let rate =
            (pos % SAMPLE_INTERVAL + redirection).as_vec3() / (offset * SAMPLE_INTERVAL) as f32;

        let mut value = Sample::new(&self.layout);

        // `rate` is the position of `pos` inside the cell, so the per axis
        // weights are `rate` and `1 - rate`. Their products sum to one, which is
        // why the accumulated value needs no normalization.
        for (&corner, sample) in corners.iter().zip(samples.iter()) {
            let weight = (rate - (1 - corner).as_vec3()).abs().element_product();

            for i in 0..stride {
                value.values_mut()[i] += sample[i] * weight;
            }
        }

        value
    }

    /// The channels of the layout, which a caller resolves once before a hot
    /// loop.
    pub fn layout(&self) -> &SampleLayout {
        &self.layout
    }
}
