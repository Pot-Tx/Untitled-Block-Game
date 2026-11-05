use crate::actor::{ACTOR_TYPES, ActorMarker};
use crate::game::ASSET_OPTIONS;
use crate::util::bounding::AABB;
use crate::util::collection::Volume;
use crate::world::LOD_COUNT;
use crate::world::block::{BlockChunk, BlockMeta, KEEP};
use anyhow::{Context, anyhow};
use glam::{U8Vec3, Vec3};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// A block structure whose content is baked into a postcard `BlockChunk`.
pub struct StructureAsset {
    pub name: String,
    pub name_hash: u64,
    /// The content of each level of detail. A level that is absent is not drawn
    /// at that distance.
    pub blocks: Vec<Volume<BlockMeta>>,
    /// The collision box, relative to the structure's site.
    pub bound: AABB<U8Vec3>,
    /// The offset from the content origin to the point the placement rules use
    /// as the structure's base.
    pub anchor: U8Vec3,
    pub max_sites: u16,
    pub actors: Vec<ActorMarker>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStructure {
    content: PathBuf,
    anchor: U8Vec3,
    collision: RawCollision,
    #[serde(default)]
    actors: Vec<RawActor>,
    #[serde(default)]
    lod: LodPolicy,
    #[serde(default = "default_max_sites")]
    max_sites: u16,
}

#[derive(Deserialize)]
struct RawCollision {
    min: U8Vec3,
    max: U8Vec3,
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
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open the structure {}", path.display()))?;
        let raw: RawStructure = ASSET_OPTIONS
            .from_reader(BufReader::new(file))
            .with_context(|| format!("failed to read the structure {}", path.display()))?;

        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let first = read_chunk(&base.join(&raw.content))?;
        let size = first.size;
        let mut blocks = vec![first.into_volume()?];

        match raw.lod {
            LodPolicy::Derive { max } => {
                if max as usize >= LOD_COUNT {
                    return Err(anyhow!(
                        "the structure {} asks for LOD {max}, but the world has only {LOD_COUNT}",
                        path.display()
                    ));
                }

                for _ in 1..=max {
                    blocks.push(reduce(blocks.last().expect("the structure has LOD0")));
                }
            }

            LodPolicy::Explicit { levels } => {
                for level in levels {
                    let chunk = read_chunk(&base.join(level))?;
                    blocks.push(chunk.into_volume()?);
                }

                if blocks.len() > LOD_COUNT {
                    return Err(anyhow!(
                        "the structure {} has {} levels, but the world has only {LOD_COUNT}",
                        path.display(),
                        blocks.len()
                    ));
                }
            }
        }

        if size.min_element() == 0 || size.max_element() > crate::world::region::REGION_SIZE {
            return Err(anyhow!(
                "the structure {} has the unsupported size {size}",
                path.display()
            ));
        }

        let anchor = raw.anchor;
        if anchor.max_element() >= size.max_element() {
            return Err(anyhow!(
                "the anchor {anchor} of {} is outside its {size} content",
                path.display()
            ));
        }

        let bound = AABB {
            min: anchor + raw.collision.min,
            max: anchor + raw.collision.max,
        };

        if bound.max.max_element() > size.max_element() {
            return Err(anyhow!(
                "the collision box {bound:?} of {} leaves its {size} content",
                path.display()
            ));
        }

        let name = path
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or_else(|| anyhow!("the structure {} has no name", path.display()))?;

        let actors = raw
            .actors
            .into_iter()
            .map(|raw| ActorMarker {
                pos: raw.pos,
                actor: ACTOR_TYPES.id_of(&raw.actor),
            })
            .collect();

        Ok(Self {
            name_hash: fnv1a(name.as_bytes()),
            name,
            blocks,
            bound,
            anchor,
            max_sites: raw.max_sites,
            actors,
        })
    }

    pub fn blocks(&self, lod: u8) -> Option<&Volume<BlockMeta>> {
        self.blocks.get(lod as usize)
    }
}

fn read_chunk(path: &Path) -> anyhow::Result<BlockChunk> {
    let chunk: BlockChunk = crate::util::read_file(path, b"STRU", 0)?
        .ok_or_else(|| anyhow!("the structure content {} is missing", path.display()))?;
    Ok(chunk)
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

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;

    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    hash
}
