//! The rules that turn the fields of a region into blocks, and the conditions a
//! structure is placed with.
//!
//! A condition reads samples of the fields at fixed offsets. The samples a
//! condition or a rule needs are resolved when it is read, so that the hot path
//! only has to sample the field and walk the tree.

use crate::generation::field::{FieldGraph, FieldView, MAX_SAMPLE_FLOATS};
use crate::world::block::{BLOCK_TYPES, BlockMeta};
use anyhow::anyhow;
use glam::{I8Vec3, IVec3};
use serde::Deserialize;

/// The largest offset a sample of a rule may be read at, in blocks.
const MAX_SAMPLE_OFFSET: i8 = 8;

/// One sample a rule or a condition reads: a channel of a field at an offset
/// from the position the rule is evaluated at.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FieldSample {
    pub field: u8,
    pub channel: usize,
    pub at: I8Vec3,
}

/// How two values of a condition are compared.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// The values a condition compares.
#[derive(Clone, Copy)]
enum Operand {
    Sample(u16),
    Constant(f32),
}

/// A condition over the fields of a generator.
pub struct FieldCondition {
    samples: Vec<FieldSample>,
    tree: ConditionNode,
}

enum ConditionNode {
    All(Vec<ConditionNode>),
    Any(Vec<ConditionNode>),
    Not(Box<ConditionNode>),
    Compare {
        lhs: Operand,
        op: CompareOp,
        rhs: Operand,
    },
}

/// The rule that decides the block of one position.
pub struct BlockRule {
    samples: Vec<FieldSample>,
    tree: RuleNode,
}

enum RuleNode {
    Block(BlockMeta),
    When {
        condition: ConditionNode,
        then: Box<RuleNode>,
        otherwise: Box<RuleNode>,
    },
}

impl FieldCondition {
    /// Reads the condition, resolving the fields and the channels it names.
    pub fn new(raw: &RawCondition, graphs: &[FieldGraph]) -> anyhow::Result<Self> {
        let mut builder = Builder::new(graphs);
        let tree = builder.condition(raw)?;

        Ok(Self {
            samples: builder.samples,
            tree,
        })
    }

    /// The number of samples the condition reads, which its caller has to pass
    /// as the scratch space of one evaluation.
    pub fn samples(&self) -> usize {
        self.samples.len()
    }

    /// Whether the condition holds at the position `origin`.
    pub fn holds(&self, view: &FieldView, origin: IVec3, lod: u8, scratch: &mut [f32]) -> bool {
        sample_all(view, &self.samples, origin, lod, false, scratch);
        walk(&self.tree, scratch)
    }
}

impl BlockRule {
    /// Reads the rule, resolving the fields, the channels and the blocks it names.
    pub fn new(raw: &RawBlockRule, graphs: &[FieldGraph]) -> anyhow::Result<Self> {
        let mut builder = Builder::new(graphs);
        let tree = builder.rule(raw)?;

        Ok(Self {
            samples: builder.samples,
            tree,
        })
    }

    /// The number of samples the rule reads, which its caller has to pass as the
    /// scratch space of one evaluation.
    pub fn samples(&self) -> usize {
        self.samples.len()
    }

    /// The block of the position `origin`.
    pub fn block(
        &self,
        view: &FieldView,
        origin: IVec3,
        lod: u8,
        halo: bool,
        scratch: &mut [f32],
    ) -> BlockMeta {
        sample_all(view, &self.samples, origin, lod, halo, scratch);
        block(&self.tree, scratch)
    }
}

/// Samples every sample of a plan at once. The offsets of a sample count blocks,
/// so that a rule reads the same voxel above a coarse block as above a fine one.
fn sample_all(
    view: &FieldView,
    samples: &[FieldSample],
    origin: IVec3,
    lod: u8,
    halo: bool,
    scratch: &mut [f32],
) {
    let mut values = [0.0f32; MAX_SAMPLE_FLOATS];
    let block = 1i32 << lod;

    for (scratch, sample) in scratch.iter_mut().zip(samples) {
        let at = sample.at;
        let pos = origin + IVec3::new(at.x as i32, at.y as i32, at.z as i32) * block;

        view.sample(sample.field as usize, pos, lod, halo, &mut values);
        *scratch = values[sample.channel];
    }
}

fn walk(node: &ConditionNode, samples: &[f32]) -> bool {
    match node {
        ConditionNode::All(nodes) => nodes.iter().all(|node| walk(node, samples)),
        ConditionNode::Any(nodes) => nodes.iter().any(|node| walk(node, samples)),
        ConditionNode::Not(node) => !walk(node, samples),
        ConditionNode::Compare { lhs, op, rhs } => {
            let lhs = value(*lhs, samples);
            let rhs = value(*rhs, samples);

            match op {
                CompareOp::Gt => lhs > rhs,
                CompareOp::Ge => lhs >= rhs,
                CompareOp::Lt => lhs < rhs,
                CompareOp::Le => lhs <= rhs,
                CompareOp::Eq => lhs == rhs,
                CompareOp::Ne => lhs != rhs,
            }
        }
    }
}

fn block(node: &RuleNode, samples: &[f32]) -> BlockMeta {
    match node {
        RuleNode::Block(block) => *block,
        RuleNode::When {
            condition,
            then,
            otherwise,
        } => {
            if walk(condition, samples) {
                block(then, samples)
            } else {
                block(otherwise, samples)
            }
        }
    }
}

fn value(operand: Operand, samples: &[f32]) -> f32 {
    match operand {
        Operand::Sample(index) => samples[index as usize],
        Operand::Constant(value) => value,
    }
}

/// Reads the raw form of a rule or a condition, collecting the samples both of
/// them need.
struct Builder<'a> {
    graphs: &'a [FieldGraph],
    samples: Vec<FieldSample>,
}

impl<'a> Builder<'a> {
    fn new(graphs: &'a [FieldGraph]) -> Self {
        Self {
            graphs,
            samples: Vec::new(),
        }
    }

    fn condition(&mut self, raw: &RawCondition) -> anyhow::Result<ConditionNode> {
        Ok(match raw {
            RawCondition::All { all: conditions } => ConditionNode::All(
                conditions
                    .iter()
                    .map(|condition| self.condition(condition))
                    .collect::<anyhow::Result<_>>()?,
            ),
            RawCondition::Any { any: conditions } => ConditionNode::Any(
                conditions
                    .iter()
                    .map(|condition| self.condition(condition))
                    .collect::<anyhow::Result<_>>()?,
            ),
            RawCondition::Not { not } => ConditionNode::Not(Box::new(self.condition(not)?)),
            RawCondition::Compare { lhs, op, rhs } => ConditionNode::Compare {
                lhs: self.operand(lhs)?,
                op: *op,
                rhs: self.operand(rhs)?,
            },
        })
    }

    fn rule(&mut self, raw: &RawBlockRule) -> anyhow::Result<RuleNode> {
        Ok(match raw {
            RawBlockRule::Block { block } => {
                let id = BLOCK_TYPES
                    .names
                    .get_by_right(block)
                    .ok_or_else(|| anyhow!("the terrain rule names the unknown block {block}"))?;

                RuleNode::Block(*id as BlockMeta)
            }
            RawBlockRule::When {
                when,
                then,
                otherwise,
            } => RuleNode::When {
                condition: self.condition(when)?,
                then: Box::new(self.rule(then)?),
                otherwise: Box::new(self.rule(otherwise)?),
            },
        })
    }

    fn operand(&mut self, raw: &RawValue) -> anyhow::Result<Operand> {
        Ok(match raw {
            RawValue::Constant(value) => Operand::Constant(*value),
            RawValue::Sample(sample) => Operand::Sample(self.sample(sample)?),
        })
    }

    fn sample(&mut self, raw: &RawFieldSample) -> anyhow::Result<u16> {
        let field = self
            .graphs
            .iter()
            .position(|graph| graph.name == raw.field)
            .ok_or_else(|| anyhow!("the rule names the unknown field {}", raw.field))?;
        let channel = self.graphs[field].channel(&raw.channel).ok_or_else(|| {
            anyhow!(
                "the rule names the unknown channel {} of the field {}",
                raw.channel,
                raw.field
            )
        })?;

        if raw.at.abs().max_element() > MAX_SAMPLE_OFFSET {
            return Err(anyhow!(
                "the rule reads a sample {} blocks away from its position, but at most {MAX_SAMPLE_OFFSET} are supported",
                raw.at.abs().max_element()
            ));
        }

        let sample = FieldSample {
            field: field as u8,
            channel,
            at: raw.at,
        };

        Ok(
            match self.samples.iter().position(|other| *other == sample) {
                Some(index) => index as u16,
                None => {
                    self.samples.push(sample);
                    (self.samples.len() - 1) as u16
                }
            },
        )
    }
}

/// A condition as it is written in a generator.
#[derive(Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum RawCondition {
    All {
        all: Vec<RawCondition>,
    },
    Any {
        any: Vec<RawCondition>,
    },
    Not {
        not: Box<RawCondition>,
    },
    Compare {
        lhs: RawValue,
        op: CompareOp,
        rhs: RawValue,
    },
}

/// The value a condition compares: a sample of a field, or a number.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum RawValue {
    Sample(RawFieldSample),
    Constant(f32),
}

/// A sample of a field as it is written in a generator.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawFieldSample {
    field: String,
    channel: String,
    /// The offset of the sample from the position the rule is evaluated at, in
    /// blocks.
    #[serde(default)]
    at: I8Vec3,
}

/// A rule as it is written in a generator.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum RawBlockRule {
    Block {
        block: String,
    },
    When {
        when: RawCondition,
        then: Box<RawBlockRule>,
        otherwise: Box<RawBlockRule>,
    },
}
