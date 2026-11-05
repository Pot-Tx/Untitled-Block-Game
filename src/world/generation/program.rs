//! The world generation program: the sample schema and the ordered stages that
//! turn a region into blocks. The program is data; the stages are wasm modules.

use crate::game::{ASSET_OPTIONS, AssetId, ModRegistry};
use crate::util::files_in;
use crate::world::LOD_COUNT;
use crate::world::generation::sample::{MAX_SAMPLE_FLOATS, SampleLayout};
use crate::world::generation::structure::StructureAsset;
use anyhow::{Context, anyhow};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The bytes of one wasm module, together with the asset they came from.
pub struct ModuleSource {
    pub asset: AssetId,
    pub bytes: Vec<u8>,
}

/// One `sample` stage.
pub struct SampleStage {
    pub name: String,
    pub module: Arc<ModuleSource>,
    pub entry: String,
    pub params: Vec<u8>,
    pub reads: Vec<usize>,
    pub writes: Vec<usize>,
}

/// One `chunk` stage.
pub struct ChunkStage {
    pub name: String,
    pub module: Arc<ModuleSource>,
    pub entry: String,
    pub params: Vec<u8>,
    pub lods: LodMask,
}

/// One `place` stage.
pub struct PlaceStage {
    pub name: String,
    pub module: Arc<ModuleSource>,
    pub entry: String,
    pub params: Vec<u8>,
    pub structure: usize,
}

/// A task stage, in the order it runs for one region.
pub enum TaskStage {
    Chunk(ChunkStage),
    Stamp { structure: usize, lods: LodMask },
}

/// The levels of detail a stage runs at.
#[derive(Clone, Copy)]
pub struct LodMask(u8);

impl LodMask {
    pub const ALL: Self = Self((1 << LOD_COUNT) - 1);

    pub fn from_levels(levels: &[u8]) -> anyhow::Result<Self> {
        let mut mask = 0u8;

        for &level in levels {
            if level as usize >= LOD_COUNT {
                return Err(anyhow!("LOD {level} is outside the world"));
            }
            mask |= 1 << level;
        }

        Ok(Self(mask))
    }

    pub fn runs(self, lod: u8) -> bool {
        self.0 & (1 << lod) != 0
    }
}

/// A compiled world generation program.
pub struct WorldProgram {
    pub fingerprint: u64,
    pub layout: Arc<SampleLayout>,
    pub sampler: Vec<SampleStage>,
    pub task_stages: Vec<TaskStage>,
    /// One entry per structure, in the order of `structures`.
    pub placements: Vec<Option<PlaceStage>>,
    pub structures: Vec<StructureAsset>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProgram {
    version: u32,
    schema: RawSchema,
    structures: PathBuf,
    stages: Vec<RawStage>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSchema {
    interval: u8,
    channels: Vec<RawChannel>,
    halo: Option<RawHalo>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawChannel {
    name: String,
    components: u8,
    #[serde(default)]
    gradient_of: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHalo {
    density: String,
    erosion: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStage {
    name: String,
    kind: StageKind,
    #[serde(default)]
    module: Option<PathBuf>,
    #[serde(default)]
    entry: Option<String>,
    #[serde(default)]
    params: Option<ron::Value>,
    #[serde(default)]
    structure: Option<String>,
    #[serde(default)]
    lods: Option<Vec<u8>>,
    #[serde(default)]
    reads: Vec<String>,
    #[serde(default)]
    writes: Vec<String>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StageKind {
    Sample,
    Chunk,
    Place,
    Stamp,
}

impl WorldProgram {
    pub fn load(asset: &AssetId, registry: &ModRegistry) -> anyhow::Result<Self> {
        let path = registry.resolve(asset)?;
        let root = registry
            .root(&asset.mod_id)
            .ok_or_else(|| anyhow!("mod {} is not loaded", asset.mod_id))?;
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read the world program {}", path.display()))?;
        let raw: RawProgram = ASSET_OPTIONS
            .from_reader(BufReader::new(File::open(&path)?))
            .with_context(|| format!("failed to read the world program {}", path.display()))?;

        if raw.version != 1 {
            return Err(anyhow!(
                "the world program {} has version {}, but this build reads version 1",
                path.display(),
                raw.version
            ));
        }

        if raw.schema.interval == 0 || 32 % raw.schema.interval != 0 {
            return Err(anyhow!(
                "the sample interval {} does not divide a region",
                raw.schema.interval
            ));
        }

        if raw.schema.interval != 8 {
            return Err(anyhow!(
                "the sample interval {} is not supported yet; the grid code uses 8",
                raw.schema.interval
            ));
        }

        let mut channels = Vec::new();
        let mut gradients = Vec::new();

        for channel in raw.schema.channels {
            match channel.gradient_of {
                Some(source) => gradients.push((channel.name, source)),
                None => channels.push((channel.name, channel.components)),
            }
        }

        let halo = raw.schema.halo.map(|halo| (halo.density, halo.erosion));
        let layout = Arc::new(SampleLayout::new(channels, halo, gradients)?);

        if layout.stride() > MAX_SAMPLE_FLOATS {
            return Err(anyhow!(
                "the sample layout holds {} floats, but at most {MAX_SAMPLE_FLOATS} are supported",
                layout.stride()
            ));
        }

        let structures_dir = AssetId::new(
            asset.mod_id.clone(),
            relative_path(root, &root.join(&raw.structures))?,
        );
        let structures = load_structures(&structures_dir, registry)?;
        let mut structure_index = std::collections::HashMap::new();

        for (index, structure) in structures.iter().enumerate() {
            if structure_index
                .insert(structure.name.clone(), index)
                .is_some()
            {
                return Err(anyhow!("two structures are named {}", structure.name));
            }
        }

        let mut sampler = Vec::new();
        let mut task_stages = Vec::new();
        let mut placements = std::iter::repeat_with(|| None)
            .take(structures.len())
            .collect::<Vec<Option<PlaceStage>>>();
        let mut module_cache = std::collections::HashMap::<PathBuf, Arc<ModuleSource>>::new();
        let mut fingerprint = fnv1a(&bytes);

        for stage in raw.stages {
            let kind_name = stage.kind_name();
            let stage_name = stage.name.clone();
            let module_path = stage.module.clone();
            let entry_name = stage.entry.clone();
            let mut module = |module: Option<PathBuf>| -> anyhow::Result<Arc<ModuleSource>> {
                let module = module
                    .ok_or_else(|| anyhow!("the {kind_name} stage {stage_name} has no module"))?;
                if let Some(module) = module_cache.get(&module) {
                    return Ok(module.clone());
                }

                let asset = AssetId::new(
                    asset.mod_id.clone(),
                    relative_path(root, &root.join(&module))?,
                );
                let bytes = registry.read(&asset)?;
                let source = Arc::new(ModuleSource {
                    asset: asset.clone(),
                    bytes,
                });
                module_cache.insert(module, source.clone());
                Ok(source)
            };

            let entry = |entry: Option<String>| -> anyhow::Result<String> {
                entry.ok_or_else(|| anyhow!("the {kind_name} stage {stage_name} has no entry"))
            };

            let params = match &stage.params {
                Some(params) => ron::ser::to_string(params)
                    .map_err(|error| anyhow!("failed to encode the parameters: {error}"))?
                    .into_bytes(),
                None => Vec::new(),
            };

            match stage.kind {
                StageKind::Sample => {
                    let module = module(module_path)?;
                    let entry = entry(entry_name)?;
                    let mut writes = Vec::new();
                    let mut reads = Vec::new();

                    for name in stage.writes {
                        let channel = layout.channel(&name).ok_or_else(|| {
                            anyhow!(
                                "the sample stage {} writes the unknown channel {name}",
                                stage_name
                            )
                        })?;
                        writes.push(channel.offset);
                    }

                    for name in stage.reads {
                        let channel = layout.channel(&name).ok_or_else(|| {
                            anyhow!(
                                "the sample stage {} reads the unknown channel {name}",
                                stage_name
                            )
                        })?;
                        reads.push(channel.offset);
                    }

                    fingerprint ^= fnv1a(&module.bytes).rotate_left(1);
                    sampler.push(SampleStage {
                        name: stage_name,
                        module,
                        entry,
                        params,
                        reads,
                        writes,
                    });
                }

                StageKind::Chunk => {
                    let module = module(module_path)?;
                    let entry = entry(entry_name)?;
                    let lods = match &stage.lods {
                        Some(levels) => LodMask::from_levels(levels)?,
                        None => LodMask::ALL,
                    };

                    fingerprint ^= fnv1a(&module.bytes).rotate_left(2);
                    task_stages.push(TaskStage::Chunk(ChunkStage {
                        name: stage_name,
                        module,
                        entry,
                        params,
                        lods,
                    }));
                }

                StageKind::Place => {
                    let module = module(module_path)?;
                    let entry = entry(entry_name)?;
                    let name = stage.structure.ok_or_else(|| {
                        anyhow!("the place stage {stage_name} names no structure")
                    })?;
                    let structure = *structure_index.get(&name).ok_or_else(|| {
                        anyhow!("the place stage {stage_name} names the unknown structure {name}")
                    })?;

                    if placements[structure].is_some() {
                        return Err(anyhow!(
                            "the structure {name} has more than one place stage"
                        ));
                    }

                    fingerprint ^= fnv1a(&module.bytes).rotate_left(3);
                    placements[structure] = Some(PlaceStage {
                        name: stage_name,
                        module,
                        entry,
                        params,
                        structure,
                    });
                }

                StageKind::Stamp => {
                    let name = stage.structure.ok_or_else(|| {
                        anyhow!("the stamp stage {stage_name} names no structure")
                    })?;
                    let structure = *structure_index.get(&name).ok_or_else(|| {
                        anyhow!("the stamp stage {stage_name} names the unknown structure {name}")
                    })?;
                    let lods = match &stage.lods {
                        Some(levels) => LodMask::from_levels(levels)?,
                        None => LodMask::ALL,
                    };

                    task_stages.push(TaskStage::Stamp { structure, lods });
                }
            }
        }

        for (index, placement) in placements.iter().enumerate() {
            let stamped = task_stages.iter().any(
                |stage| matches!(stage, TaskStage::Stamp { structure, .. } if *structure == index),
            );

            if stamped && placement.is_none() {
                return Err(anyhow!(
                    "the structure {} is stamped but has no place stage",
                    structures[index].name
                ));
            }
        }

        if !task_stages
            .iter()
            .any(|stage| matches!(stage, TaskStage::Chunk(_)))
        {
            return Err(anyhow!("the world program has no chunk stage"));
        }

        fingerprint ^= fnv1a(&raw_path_bytes(&path)?).rotate_left(4);

        Ok(Self {
            fingerprint,
            layout,
            sampler,
            task_stages,
            placements,
            structures,
        })
    }
}

impl RawStage {
    fn kind_name(&self) -> &'static str {
        match self.kind {
            StageKind::Sample => "sample",
            StageKind::Chunk => "chunk",
            StageKind::Place => "place",
            StageKind::Stamp => "stamp",
        }
    }
}

fn load_structures(asset: &AssetId, registry: &ModRegistry) -> anyhow::Result<Vec<StructureAsset>> {
    let path = registry.resolve(asset)?;
    let mut structures = Vec::new();

    for path in files_in(&path)? {
        if path.extension().and_then(|extension| extension.to_str()) == Some("ron") {
            structures.push(StructureAsset::load(&path)?);
        }
    }

    Ok(structures)
}

fn relative_path(root: &Path, path: &Path) -> anyhow::Result<PathBuf> {
    path.strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| anyhow!("the path {} leaves the mod root", path.display()))
}

fn raw_path_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;

    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    hash
}
