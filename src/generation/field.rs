//! The fields of a generator: the graphs that compute their samples, the
//! samples of a region, and the sampling of a field at a world position.

use crate::util::collection::Volume;
use crate::util::coord::{Axis, Coord3, ICoord3};
use crate::world::RegionPos;
use crate::world::region::REGION_SIZE;
use anyhow::anyhow;
use glam::{IVec3, U8Vec3, Vec3};
use noise_functions::{CellDistanceSq, Noise, Perlin};
use rand_seeder::Seeder;
use serde::Deserialize;
use std::sync::Arc;

/// The number of floats one sample may hold.
pub const MAX_SAMPLE_FLOATS: usize = 32;

/// The cells a graph computes beyond the samples of its region, because a
/// gradient reads the cell before and after.
const MARGIN: i32 = 1;

/// One node of a field graph. Every node computes one scalar per sample.
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum FieldNode {
    /// A constant.
    Constant {
        value: f32,
    },
    /// The world position along `axis`, in blocks.
    Position {
        axis: Axis,
    },
    /// A perlin noise of the world position, in blocks.
    Perlin {
        frequency: f32,
        /// Keeps two noises of one graph apart, together with the seed of the world.
        salt: u32,
        #[serde(default = "one")]
        scale: [f32; 3],
    },
    /// The distance to the nearest point of a jittered cell pattern.
    Cells {
        frequency: f32,
        jitter: f32,
        salt: u32,
        #[serde(default = "one")]
        scale: [f32; 3],
    },
    /// The slope of `source` along `axis`, per block.
    Gradient {
        source: u32,
        axis: Axis,
    },
    Add {
        lhs: u32,
        rhs: u32,
    },
    Sub {
        lhs: u32,
        rhs: u32,
    },
    Mul {
        lhs: u32,
        rhs: u32,
    },
    Div {
        lhs: u32,
        rhs: u32,
    },
    Abs {
        source: u32,
    },
    Tanh {
        source: u32,
    },
    Clamp {
        source: u32,
        min: f32,
        max: f32,
    },
    /// `then` where `condition` is greater than zero, `otherwise` everywhere else.
    Select {
        condition: u32,
        then: u32,
        otherwise: u32,
    },
}

/// The graph of one field: what its samples are made of.
pub struct FieldGraph {
    /// The name the rules of the generator call the field by.
    pub name: String,
    /// The number of blocks between two samples.
    pub interval: u8,
    /// The cells one region holds, without the cell the interpolation reads.
    pub cells: U8Vec3,
    /// The number of floats one sample holds.
    pub stride: usize,
    /// The offset of the channel that says how open a sample is, which the
    /// sampler biases the border samples of a coarse block towards.
    pub bias: Option<usize>,
    nodes: Vec<FieldNode>,
    channels: Vec<(String, usize)>,
}

/// The content of one field over one region, channel interleaved.
pub struct FieldGrid {
    pub interval: u8,
    /// The samples along each axis: the cells of the region plus one, so that a
    /// position inside the last cell still reads the cell after it.
    pub size: U8Vec3,
    pub stride: usize,
    pub data: Vec<f32>,
}

impl FieldGraph {
    /// Reads the graph of one field, checking the interval, the nodes, the
    /// channels and the number of floats a sample holds.
    pub(crate) fn new(
        name: &str,
        interval: u8,
        channels: &[(String, u32)],
        bias: Option<&str>,
        nodes: Vec<FieldNode>,
    ) -> anyhow::Result<Self> {
        if interval < 2 || !REGION_SIZE.is_multiple_of(interval) {
            return Err(anyhow!(
                "the interval {interval} of the field {name} does not divide the {REGION_SIZE} \
                 blocks of a region"
            ));
        }

        let mut graph = Self {
            name: name.to_owned(),
            interval,
            cells: U8Vec3::splat(REGION_SIZE / interval),
            stride: 0,
            nodes: Vec::with_capacity(nodes.len()),
            channels: Vec::new(),
            bias: None,
        };

        for node in nodes {
            graph.push(node)?;
        }

        for (channel, node) in channels {
            graph.push_channel(channel, *node)?;
        }

        if graph.stride > MAX_SAMPLE_FLOATS {
            return Err(anyhow!(
                "the field {name} holds {} floats per sample, but at most {MAX_SAMPLE_FLOATS} are \
                 supported",
                graph.stride
            ));
        }

        graph.bias = match bias {
            Some(bias) => Some(graph.channel(bias).ok_or_else(|| {
                anyhow!("the field {name} biases its samples towards the missing channel {bias}")
            })?),
            None => None,
        };

        Ok(graph)
    }

    /// Adds a node, which may only read the nodes before it.
    fn push(&mut self, node: FieldNode) -> anyhow::Result<()> {
        let index = self.nodes.len() as u32;

        match &node {
            FieldNode::Constant { .. }
            | FieldNode::Position { .. }
            | FieldNode::Perlin { .. }
            | FieldNode::Cells { .. } => (),
            FieldNode::Gradient { source, .. }
            | FieldNode::Abs { source }
            | FieldNode::Tanh { source }
            | FieldNode::Clamp { source, .. } => check(index, *source, "source")?,
            FieldNode::Add { lhs, rhs }
            | FieldNode::Sub { lhs, rhs }
            | FieldNode::Mul { lhs, rhs }
            | FieldNode::Div { lhs, rhs } => {
                check(index, *lhs, "left side")?;
                check(index, *rhs, "right side")?;
            }
            FieldNode::Select {
                condition,
                then,
                otherwise,
            } => {
                check(index, *condition, "condition")?;
                check(index, *then, "then")?;
                check(index, *otherwise, "otherwise")?;
            }
        }

        self.nodes.push(node);
        Ok(())
    }

    fn push_channel(&mut self, name: &str, node: u32) -> anyhow::Result<()> {
        if self.channel(name).is_some() {
            return Err(anyhow!(
                "the field {} names the channel {name} twice",
                self.name
            ));
        }

        check(self.nodes.len() as u32, node, "channel")?;

        self.channels.push((name.to_owned(), node as usize));
        self.stride += 1;
        Ok(())
    }

    /// The offset of the channel `name` inside a sample.
    pub fn channel(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|(entry, _)| entry == name)
    }

    /// Computes the field of the region `region` for the seed of the world.
    pub fn evaluate(&self, region: RegionPos, seed: u64) -> FieldGrid {
        let cells = self.cells.as_ivec3();
        let dims = (cells + IVec3::splat(1 + 2 * MARGIN)).as_u8vec3();
        let edge = dims.z as usize;
        let plane = edge * dims.y as usize;
        let origin = region * REGION_SIZE as i32;
        let interval = self.interval as i32;

        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let mut values = vec![0.0f32; plane * dims.x as usize];

            for x in 0..dims.x as i32 {
                for y in 0..dims.y as i32 {
                    for z in 0..dims.z as i32 {
                        let index = x as usize * plane + y as usize * edge + z as usize;
                        let pos = origin
                            + IVec3::new(
                                (x - MARGIN) * interval,
                                (y - MARGIN) * interval,
                                (z - MARGIN) * interval,
                            );
                        let at = |step: IVec3| {
                            let x = (x + step.x).clamp(0, dims.x as i32 - 1) as usize;
                            let y = (y + step.y).clamp(0, dims.y as i32 - 1) as usize;
                            let z = (z + step.z).clamp(0, dims.z as i32 - 1) as usize;

                            x * plane + y * edge + z
                        };
                        let value = |node: u32| planes[node as usize][index];

                        values[index] = match node {
                            FieldNode::Constant { value } => *value,
                            FieldNode::Position { axis } => pos.get(*axis) as f32,
                            FieldNode::Perlin {
                                frequency,
                                salt,
                                scale,
                            } => Perlin
                                .frequency(*frequency)
                                .seed(seed_of(seed, *salt))
                                .sample3(pos.as_vec3() * Vec3::from_array(*scale)),
                            FieldNode::Cells {
                                frequency,
                                jitter,
                                salt,
                                scale,
                            } => CellDistanceSq::default()
                                .jitter(*jitter)
                                .frequency(*frequency)
                                .seed(seed_of(seed, *salt))
                                .sample3(pos.as_vec3() * Vec3::from_array(*scale)),
                            FieldNode::Gradient { source, axis } => {
                                let source = &planes[*source as usize];
                                let after = source[at(IVec3::ZERO.step(axis.direction(true)))];
                                let before = source[at(IVec3::ZERO.step(axis.direction(false)))];

                                (after - before) / (2 * interval) as f32
                            }
                            FieldNode::Add { lhs, rhs } => value(*lhs) + value(*rhs),
                            FieldNode::Sub { lhs, rhs } => value(*lhs) - value(*rhs),
                            FieldNode::Mul { lhs, rhs } => value(*lhs) * value(*rhs),
                            FieldNode::Div { lhs, rhs } => value(*lhs) / value(*rhs),
                            FieldNode::Abs { source } => value(*source).abs(),
                            FieldNode::Tanh { source } => value(*source).tanh(),
                            FieldNode::Clamp { source, min, max } => {
                                value(*source).clamp(*min, *max)
                            }
                            FieldNode::Select {
                                condition,
                                then,
                                otherwise,
                            } => {
                                if value(*condition) > 0.0 {
                                    value(*then)
                                } else {
                                    value(*otherwise)
                                }
                            }
                        };
                    }
                }
            }

            planes.push(values);
        }

        // Keep the cells the interpolation reads: the ones of the region plus the
        // cell after them.
        let size = (cells + IVec3::ONE).as_u8vec3();
        let kept_edge = size.z as usize;
        let kept_plane = kept_edge * size.y as usize;
        let mut data = vec![0.0f32; kept_plane * size.x as usize * self.stride];

        for (offset, (_, node)) in self.channels.iter().enumerate() {
            let values = &planes[*node];

            for x in 0..size.x as usize {
                for y in 0..size.y as usize {
                    for z in 0..size.z as usize {
                        let from = (x as i32 + MARGIN) as usize * plane
                            + (y as i32 + MARGIN) as usize * edge
                            + (z as i32 + MARGIN) as usize;
                        let to = (x * kept_plane + y * kept_edge + z) * self.stride + offset;

                        data[to] = values[from];
                    }
                }
            }
        }

        FieldGrid {
            interval: self.interval,
            size,
            stride: self.stride,
            data,
        }
    }
}

impl FieldGrid {
    /// The samples of one cell, channel interleaved.
    #[inline]
    pub fn values(&self, cell: U8Vec3) -> &[f32] {
        let index = cell.x as usize * self.size.y as usize * self.size.z as usize
            + cell.y as usize * self.size.z as usize
            + cell.z as usize;

        &self.data[index * self.stride..(index + 1) * self.stride]
    }
}

/// The fields of the regions a task covers, which a rule samples.
pub struct FieldView<'a> {
    pub graphs: &'a [FieldGraph],
    /// One volume of region grids per field, over the regions of the task.
    pub regions: &'a [Volume<Arc<FieldGrid>>],
    /// The region the first grid of every volume covers.
    pub base: RegionPos,
}

impl FieldView<'_> {
    /// Samples the field `field` at the world position `pos`, which is inside a
    /// block `lod` levels coarser than a block.
    ///
    /// A block is sampled at its center, so that a coarse block stands for the
    /// surface of the finer blocks it covers.
    pub fn sample(&self, field: usize, pos: IVec3, lod: u8, halo: bool, out: &mut [f32]) {
        let graph = &self.graphs[field];
        let grid = self.grid(field, pos);
        let interval = graph.interval as f32;
        let block = 1i32 << lod;
        let region = pos.div_euclid(IVec3::splat(REGION_SIZE as i32));
        let local = pos - region * REGION_SIZE as i32;
        let point = local.as_vec3() + Vec3::splat(block as f32 * 0.5);
        let cell = (point / interval).floor().as_u8vec3();
        let stride = grid.stride;
        let dims = grid.size;
        let corners = U8Vec3::corners(U8Vec3::ZERO, U8Vec3::ONE);

        let mut values = [[0.0f32; MAX_SAMPLE_FLOATS]; 8];

        for (values, corner) in values.iter_mut().zip(corners.iter()) {
            let at = (cell.as_ivec3() + corner.as_ivec3())
                .clamp(IVec3::ZERO, dims.as_ivec3() - 1)
                .as_u8vec3();

            values[..stride].copy_from_slice(grid.values(at));
        }

        let mut frac = (point - cell.as_vec3() * interval) / interval;

        if let Some(bias) = graph.bias.filter(|_| lod > 0 && halo) {
            let mut best = 0;

            for index in 1..values.len() {
                if values[index][bias] < values[best][bias] {
                    best = index;
                }
            }

            let end = cell.as_vec3() * interval + interval - 1.0;
            let nudged =
                (local.as_vec3() + corners[best].as_vec3() * (block as f32 - 1.0)).min(end);

            frac = (nudged - cell.as_vec3() * interval) / interval;
        }

        out[..stride].fill(0.0);

        for (index, corner) in corners.iter().enumerate() {
            let weight = (frac - (1 - corner).as_vec3()).abs().element_product();

            for (out, value) in out[..stride].iter_mut().zip(values[index].iter()) {
                *out += value * weight;
            }
        }
    }

    fn grid(&self, field: usize, pos: IVec3) -> &FieldGrid {
        let region = pos.div_euclid(IVec3::splat(REGION_SIZE as i32));
        let volume = &self.regions[field];
        let index = (region - self.base)
            .clamp(IVec3::ZERO, volume.size.as_ivec3() - 1)
            .as_u8vec3();

        volume.get(index)
    }
}

/// Checks that a node reads one of the nodes before it.
fn check(from: u32, source: u32, what: &str) -> anyhow::Result<()> {
    if source >= from {
        return Err(anyhow!(
            "the field node {from} reads its {what} {source}, which is not an earlier node"
        ));
    }

    Ok(())
}

/// The seed one noise of a graph is drawn with, which keeps the noises of a graph
/// apart without depending on the order of their nodes.
fn seed_of(seed: u64, salt: u32) -> i32 {
    let mut seeder = Seeder::from((seed, salt));
    let bytes: [u8; 4] = seeder.make_seed();

    i32::from_le_bytes(bytes)
}

/// The scale a noise samples the world position at unless it says otherwise.
const fn one() -> [f32; 3] {
    [1.0; 3]
}
