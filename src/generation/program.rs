//! The program of a generator: the fields it samples, the rule that turns their
//! samples into blocks, and the structures it places.

use super::field::FieldNode;
use super::rule::{BlockRule, FieldCondition, RawBlockRule, RawCondition};
use super::structure::{STRUCTURES, id_of};
use crate::game::{ASSET_OPTIONS, AssetId, ModRegistry};
use crate::util::Id;
use anyhow::{Context, anyhow};
use glam::U8Vec3;
use serde::Deserialize;

/// The number of sites a structure tries to place in a region.
const DEFAULT_ATTEMPTS: u16 = 64;

/// A compiled generator: the three phases a region runs through.
pub struct WorldProgram {
    /// The field of each grid, in the order the rules name them.
    pub graphs: Vec<super::field::FieldGraph>,
    pub terrain: BlockRule,
    pub placements: Vec<Placement>,
}

/// One structure the generator places, with what it takes to choose a site.
pub struct Placement {
    pub structure: Id,
    /// How often a region tries to place a site of this structure.
    pub attempts: u16,
    /// Whether a site may stand at a position.
    pub condition: FieldCondition,
    /// Whether a region is worth trying at all, read at the corners of the region
    /// expanded by the structure.
    pub gate: Option<FieldCondition>,
    /// The largest size of the variants of the structure, which the gate expands
    /// the region by.
    pub size: U8Vec3,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProgram {
    version: u32,
    grids: Vec<RawGrid>,
    terrain: RawTerrain,
    #[serde(default)]
    structures: Vec<RawPlacement>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGrid {
    name: String,
    interval: u8,
    channels: Vec<(String, u32)>,
    bias: Option<String>,
    nodes: Vec<FieldNode>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTerrain {
    rule: RawBlockRule,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlacement {
    structure: String,
    #[serde(default = "default_attempts")]
    attempts: u16,
    #[serde(default)]
    gate: Option<RawCondition>,
    condition: RawCondition,
}

const fn default_attempts() -> u16 {
    DEFAULT_ATTEMPTS
}

impl WorldProgram {
    /// Reads the program of the generator `asset`, resolving the fields, the
    /// channels, the blocks and the structures it names.
    pub fn load(asset: &AssetId, mods: &ModRegistry) -> anyhow::Result<Self> {
        let raw: RawProgram = ASSET_OPTIONS
            .from_reader(mods.read(asset)?.as_slice())
            .with_context(|| format!("failed to read the generator {}", asset.display()))?;

        if raw.version != 0 {
            return Err(anyhow!(
                "the generator {} has the version {}, but this build reads version 0",
                asset.display(),
                raw.version
            ));
        }

        let mut graphs = Vec::with_capacity(raw.grids.len());

        for grid in raw.grids {
            let RawGrid {
                name,
                interval,
                channels,
                bias,
                nodes,
            } = grid;

            if graphs
                .iter()
                .any(|other: &super::field::FieldGraph| other.name == name)
            {
                return Err(anyhow!(
                    "the generator {} names the field {name} twice",
                    asset.display()
                ));
            }

            graphs.push(
                super::field::FieldGraph::new(&name, interval, &channels, bias.as_deref(), nodes)
                    .with_context(|| format!("in the field {name}"))?,
            );
        }

        if graphs.is_empty() {
            return Err(anyhow!("the generator {} has no field", asset.display()));
        }

        let terrain = BlockRule::new(&raw.terrain.rule, &graphs)
            .context("in the terrain rule of the generator")?;
        let mut placements = Vec::with_capacity(raw.structures.len());

        for raw in raw.structures {
            let structure = id_of(&raw.structure)?;
            let name = STRUCTURES.name_of(structure);
            let asset = STRUCTURES.get(structure);
            let size = asset
                .variants()
                .iter()
                .fold(U8Vec3::ZERO, |size, variant| size.max(variant.size));

            let condition = FieldCondition::new(&raw.condition, &graphs)
                .with_context(|| format!("in the condition of the structure {name}"))?;
            let gate = raw
                .gate
                .as_ref()
                .map(|gate| FieldCondition::new(gate, &graphs))
                .transpose()
                .with_context(|| format!("in the gate of the structure {name}"))?;

            placements.push(Placement {
                structure,
                attempts: raw.attempts,
                condition,
                gate,
                size,
            });
        }

        Ok(Self {
            graphs,
            terrain,
            placements,
        })
    }
}
