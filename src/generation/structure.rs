use crate::actor::{ACTOR_TYPES, ActorMarker};
use crate::game::{ASSET_OPTIONS, AssetId, ModRegistry};
use crate::util::OnceInit;
use crate::util::bounding::AABB;
use crate::util::collection::{Registry, Volume};
use crate::util::decode_file;
use crate::world::LOD_COUNT;
use crate::world::block::{BlockChunk, BlockMeta, KEEP};
use anyhow::{Context, anyhow};
use glam::{U8Vec3, Vec3};
use log::error;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The directory of the structures inside a mod.
const STRUCTURE_DIR: &str = "assets/structures";

/// The structures the generators place, by name.
///
/// A structure is content of its own rather than part of one generator, so a mod
/// may add a structure the generators of another mod place, or override one.
pub static STRUCTURES: OnceInit<Registry<Arc<StructureAsset>>> = OnceInit::new();

/// Reads the structure files of the mods; their content is read by [`structure`],
/// because a generator only pays for the structures it names.
pub(crate) fn init_structures(mods: &ModRegistry) {
    match load_structures(mods) {
        Ok(structures) => STRUCTURES.init(structures),
        Err(e) => {
            error!("failed to load the structures: {e}");
            STRUCTURES.init(Registry::new());
        }
    }
}

fn load_structures(mods: &ModRegistry) -> anyhow::Result<Registry<Arc<StructureAsset>>> {
    let mut files = mods.files(Path::new(STRUCTURE_DIR))?;

    // The content of a structure is a file next to it, so only the descriptions
    // become entries; the ids stay the positions among them.
    files.retain(|asset| asset.path.extension().and_then(|e| e.to_str()) == Some("ron"));

    let mut structures = Registry::new();

    for (id, asset) in files.into_iter().enumerate() {
        structures.register(
            id as crate::util::Id,
            asset.name()?,
            Arc::new(StructureAsset::load(mods, &asset)?),
        );
    }

    Ok(structures)
}

/// The structure `name`, or an error that names the structures there are.
pub(super) fn id_of(name: &str) -> anyhow::Result<crate::util::Id> {
    STRUCTURES.names.get_by_right(name).copied().ok_or_else(|| {
        let loaded = (0..STRUCTURES.entries.len())
            .map(|id| STRUCTURES.name_of(id as crate::util::Id))
            .collect::<Vec<_>>()
            .join(", ");

        anyhow!("no loaded mod provides the structure {name}; the structures are: {loaded}")
    })
}

/// A block structure whose content is baked into a postcard `BlockChunk`.
///
/// A structure is stored as the transforms it may be placed with, so that
/// stamping a site never has to rotate anything: each variant carries its own
/// content, collision box, anchor and actor markers.
pub struct StructureAsset {
    variants: Vec<StructureVariant>,
    pub max_sites: u16,
}

/// One transform of the content of a structure.
pub struct StructureVariant {
    /// The size of the content after the transform, which swaps the two
    /// horizontal axes of a quarter turn.
    pub size: U8Vec3,
    /// The collision box, relative to the content origin of the site.
    pub bound: AABB<U8Vec3>,
    /// The offset from the content origin to the point the placement rules use as
    /// the structure's base.
    pub anchor: U8Vec3,
    /// The content of each level of detail. A level that is absent is not drawn at
    /// that distance.
    blocks: Vec<Volume<BlockMeta>>,
    pub actors: Vec<ActorMarker>,
}

impl StructureAsset {
    /// The variants a script may place, in the order the magic of a site names them.
    pub fn variants(&self) -> &[StructureVariant] {
        &self.variants
    }

    /// The variant `index`, or the first one when a script names one there is not.
    pub fn variant(&self, index: u8) -> &StructureVariant {
        self.variants
            .get(index as usize)
            .unwrap_or_else(|| &self.variants[0])
    }
}

impl StructureVariant {
    pub fn blocks(&self, lod: u8) -> Option<&Volume<BlockMeta>> {
        self.blocks.get(lod as usize)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStructure {
    content: PathBuf,
    anchor: U8Vec3,
    collision: AABB<U8Vec3>,
    #[serde(default)]
    actors: Vec<RawActor>,
    #[serde(default)]
    lod: LodPolicy,
    #[serde(default = "default_max_sites")]
    max_sites: u16,
    /// The quarter turns the structure may be placed with. A missing list keeps
    /// it unturned.
    #[serde(default)]
    rotations: Vec<u8>,
    /// Whether the structure may also be placed mirrored along X, before a turn.
    #[serde(default)]
    mirror: bool,
}

#[derive(Deserialize)]
struct RawActor {
    pos: Vec3,
    actor: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum LodPolicy {
    #[serde(rename = "derive")]
    Derive {
        #[serde(default = "default_lod_max")]
        max: u8,
    },
    #[serde(rename = "explicit")]
    Explicit { levels: Vec<PathBuf> },
}

impl Default for LodPolicy {
    fn default() -> Self {
        Self::Derive {
            max: default_lod_max(),
        }
    }
}

const fn default_max_sites() -> u16 {
    64
}

const fn default_lod_max() -> u8 {
    3
}

impl StructureAsset {
    /// Reads the structure `asset` of the mods, with the content it names, and
    /// bakes every transform the structure allows.
    pub fn load(mods: &ModRegistry, asset: &AssetId) -> anyhow::Result<Self> {
        let raw: RawStructure = ASSET_OPTIONS
            .from_reader(mods.read(asset)?.as_slice())
            .with_context(|| format!("failed to read the structure {}", asset.display()))?;

        let base = asset.path.parent().unwrap_or(Path::new("."));
        let first = read_chunk(mods, &base.join(&raw.content))?;
        let size = first.size;
        let mut levels = vec![first.into_volume()?];

        match &raw.lod {
            LodPolicy::Derive { max } => {
                if *max as usize >= LOD_COUNT {
                    return Err(anyhow!(
                        "the structure {} asks for LOD {max}, but the world has only {LOD_COUNT}",
                        asset.display()
                    ));
                }

                for _ in 1..=*max {
                    levels.push(reduce(levels.last().expect("the structure has LOD0")));
                }
            }

            LodPolicy::Explicit { levels: files } => {
                for level in files {
                    let chunk = read_chunk(mods, &base.join(level))?;
                    levels.push(chunk.into_volume()?);
                }

                if levels.len() > LOD_COUNT {
                    return Err(anyhow!(
                        "the structure {} has {} levels, but the world has only {LOD_COUNT}",
                        asset.display(),
                        levels.len()
                    ));
                }
            }
        }

        if size.min_element() == 0 || size.max_element() > crate::world::region::REGION_SIZE {
            return Err(anyhow!(
                "the structure {} has the unsupported size {size}",
                asset.display()
            ));
        }

        if raw.anchor.max_element() >= size.max_element() {
            return Err(anyhow!(
                "the anchor {} of {} is outside its {size} content",
                raw.anchor,
                asset.display()
            ));
        }

        let bound = AABB {
            min: raw.anchor + raw.collision.min,
            max: raw.anchor + raw.collision.max,
        };

        if bound.max.max_element() > size.max_element() {
            return Err(anyhow!(
                "the collision box {bound:?} of {} leaves its {size} content",
                asset.display()
            ));
        }

        let actors = raw
            .actors
            .iter()
            .map(|raw| ActorMarker {
                pos: raw.pos,
                actor: ACTOR_TYPES.id_of(&raw.actor),
            })
            .collect::<Vec<_>>();

        let mut variants = Vec::new();

        for transform in transforms(&raw, asset)? {
            let variant = transform.bake(size, &levels, raw.anchor, bound, &actors);

            if variant.anchor.max_element() >= variant.size.max_element()
                || variant.bound.max.max_element() > variant.size.max_element()
            {
                return Err(anyhow!(
                    "the transform of {} puts its anchor or collision box outside its \
                     {} content",
                    asset.display(),
                    variant.size
                ));
            }

            if variants.iter().any(|other| same(other, &variant)) {
                continue;
            }

            variants.push(variant);
        }

        Ok(Self {
            variants,
            max_sites: raw.max_sites,
        })
    }
}

/// The transforms the structure allows, the untransformed one first.
fn transforms(raw: &RawStructure, asset: &AssetId) -> anyhow::Result<Vec<Transform>> {
    let rotations = if raw.rotations.is_empty() {
        &[0][..]
    } else {
        &raw.rotations[..]
    };

    let mut transforms = Vec::new();

    for &quarters in rotations {
        if quarters > 3 {
            return Err(anyhow!(
                "the structure {} asks for a {quarters} quarter turn, but a turn is 0 to 3",
                asset.display()
            ));
        }

        transforms.push(Transform {
            quarters,
            mirror: false,
        });
    }

    if raw.mirror {
        for &quarters in rotations {
            transforms.push(Transform {
                quarters,
                mirror: true,
            });
        }
    }

    transforms.sort_by_key(|transform| (transform.mirror, transform.quarters));
    transforms.dedup_by_key(|transform| (transform.mirror, transform.quarters));

    Ok(transforms)
}

/// Whether two variants hold the same content, which is what collapses the
/// transforms a symmetric structure does not tell apart.
fn same(a: &StructureVariant, b: &StructureVariant) -> bool {
    a.size == b.size
        && a.anchor == b.anchor
        && a.bound.min == b.bound.min
        && a.bound.max == b.bound.max
        && a.blocks.len() == b.blocks.len()
        && a.blocks
            .iter()
            .zip(b.blocks.iter())
            .all(|(a, b)| a.size == b.size && a.vec == b.vec)
        && a.actors.len() == b.actors.len()
        && a.actors
            .iter()
            .zip(b.actors.iter())
            .all(|(a, b)| a.pos == b.pos && a.actor == b.actor)
}

/// A quarter turn about the vertical axis through the content origin, optionally
/// after mirroring the content along X.
#[derive(Clone, Copy)]
struct Transform {
    quarters: u8,
    mirror: bool,
}

impl Transform {
    /// The size of a volume after the transform.
    fn size(self, size: U8Vec3) -> U8Vec3 {
        if self.quarters % 2 == 1 {
            U8Vec3::new(size.z, size.y, size.x)
        } else {
            size
        }
    }

    /// Where the cell `pos` of a volume of `size` goes.
    fn cell(self, size: U8Vec3, pos: U8Vec3) -> U8Vec3 {
        let mut x = pos.x;
        let z = pos.z;

        if self.mirror {
            x = size.x - 1 - x;
        }

        match self.quarters {
            1 => U8Vec3::new(size.z - 1 - z, pos.y, x),
            2 => U8Vec3::new(size.x - 1 - x, pos.y, size.z - 1 - z),
            3 => U8Vec3::new(z, pos.y, size.x - 1 - x),
            _ => U8Vec3::new(x, pos.y, z),
        }
    }

    /// Where the point `point` of the content goes, for the positions of actors
    /// and other continuous positions.
    fn point(self, size: U8Vec3, point: Vec3) -> Vec3 {
        let mut x = point.x;
        let z = point.z;
        let (sx, sz) = (size.x as f32, size.z as f32);

        if self.mirror {
            x = sx - x;
        }

        match self.quarters {
            1 => Vec3::new(sz - z, point.y, x),
            2 => Vec3::new(sx - x, point.y, sz - z),
            3 => Vec3::new(z, point.y, sx - x),
            _ => Vec3::new(x, point.y, z),
        }
    }

    fn volume(self, volume: &Volume<BlockMeta>) -> Volume<BlockMeta> {
        let mut transformed = Volume::splat(self.size(volume.size), KEEP);

        for x in 0..volume.size.x {
            for y in 0..volume.size.y {
                for z in 0..volume.size.z {
                    let pos = U8Vec3::new(x, y, z);
                    let to = self.cell(volume.size, pos);

                    transformed.set(to, *volume.get(pos));
                }
            }
        }

        transformed
    }

    fn bound(self, size: U8Vec3, bound: AABB<U8Vec3>) -> AABB<U8Vec3> {
        let min = self.cell(size, bound.min);
        let max = self.cell(size, bound.max - U8Vec3::ONE);

        AABB {
            min: min.min(max),
            max: min.max(max) + U8Vec3::ONE,
        }
    }

    fn bake(
        self,
        size: U8Vec3,
        levels: &[Volume<BlockMeta>],
        anchor: U8Vec3,
        bound: AABB<U8Vec3>,
        actors: &[ActorMarker],
    ) -> StructureVariant {
        StructureVariant {
            size: self.size(size),
            bound: self.bound(size, bound),
            anchor: self.cell(size, anchor),
            blocks: levels.iter().map(|level| self.volume(level)).collect(),
            actors: actors
                .iter()
                .map(|marker| ActorMarker {
                    pos: self.point(size, marker.pos),
                    actor: marker.actor,
                })
                .collect(),
        }
    }
}

fn read_chunk(mods: &ModRegistry, path: &Path) -> anyhow::Result<BlockChunk> {
    let asset = mods.asset(path)?;

    decode_file(&mods.read(&asset)?, b"STRU", 0, &asset.path)
}

fn reduce(volume: &Volume<BlockMeta>) -> Volume<BlockMeta> {
    let size = U8Vec3::new(
        volume.size.x.div_ceil(2),
        volume.size.y.div_ceil(2),
        volume.size.z.div_ceil(2),
    );
    let mut reduced = Volume::new(size);

    for x in 0..size.x {
        for y in 0..size.y {
            for z in 0..size.z {
                let mut counts = HashMap::<BlockMeta, usize>::new();

                for dx in 0..2 {
                    for dy in 0..2 {
                        for dz in 0..2 {
                            let pos = U8Vec3::new(x * 2 + dx, y * 2 + dy, z * 2 + dz);

                            if pos.x < volume.size.x
                                && pos.y < volume.size.y
                                && pos.z < volume.size.z
                            {
                                let block = *volume.get(pos);

                                if block != KEEP {
                                    *counts.entry(block).or_default() += 1;
                                }
                            }
                        }
                    }
                }

                let block = counts
                    .into_iter()
                    .max_by_key(|&(block, count)| (count, std::cmp::Reverse(block)))
                    .map(|(block, _)| block)
                    .unwrap_or(KEEP);

                reduced.set(U8Vec3::new(x, y, z), block);
            }
        }
    }

    reduced
}

impl BlockChunk {
    fn into_volume(self) -> anyhow::Result<Volume<BlockMeta>> {
        Ok(self.0)
    }
}
