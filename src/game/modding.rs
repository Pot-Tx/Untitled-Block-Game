//! The mod packages the game is loaded from. A mod is a directory with a
//! `mod.ron` at its root and assets below it. The registry scans every mod,
//! resolves dependencies in a stable order, and hands out paths without ever
//! leaving a mod's root.

use crate::util::files_in;
use anyhow::{Context, anyhow};
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Component, Path, PathBuf};

/// The identifier of a mod, which is the `id` of its manifest.
pub type ModId = String;

/// A path inside one mod's package.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetId {
    pub mod_id: ModId,
    pub path: PathBuf,
}

impl AssetId {
    pub fn new(mod_id: impl Into<ModId>, path: impl Into<PathBuf>) -> Self {
        Self {
            mod_id: mod_id.into(),
            path: path.into(),
        }
    }

    pub fn display(&self) -> String {
        format!("{}:{}", self.mod_id, self.path.display())
    }
}

/// One loaded mod.
pub struct Mod {
    pub manifest: ModManifest,
    pub root: PathBuf,
}

/// The contents of a `mod.ron`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModManifest {
    pub id: ModId,
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub dependencies: Vec<ModId>,
    /// The world program this mod contributes, relative to the mod root.
    #[serde(default)]
    pub world: Option<PathBuf>,
}

impl ModManifest {
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open the mod manifest {}", path.display()))?;
        let manifest: Self = crate::game::ASSET_OPTIONS
            .from_reader(BufReader::new(file))
            .with_context(|| format!("failed to read the mod manifest {}", path.display()))?;

        if manifest.id.is_empty() {
            return Err(anyhow!("a mod manifest has an empty id"));
        }

        if manifest.id.contains(['/', '\\', ':']) {
            return Err(anyhow!(
                "the mod id {} contains a path separator",
                manifest.id
            ));
        }

        Ok(manifest)
    }
}

/// The id of the mod that carries the content the game ships with.
///
/// It loads before every other mod, so that a mod can override any of its files
/// and no mod can change what the game itself consists of.
const CORE_MOD: &str = "core";

/// Every mod the game found, with the paths of their assets.
pub struct ModRegistry {
    mods: BTreeMap<ModId, Mod>,
    /// The order dependencies put the mods in. Assets are loaded in this order.
    order: Vec<ModId>,
}

impl ModRegistry {
    /// Scans the immediate child directories of `root` for a `mod.ron`.
    pub fn load(root: &Path) -> anyhow::Result<Self> {
        let mut manifests = Vec::new();

        for entry in fs::read_dir(root)
            .with_context(|| format!("failed to read the mod directory {}", root.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let manifest_path = path.join("mod.ron");

            if path.is_dir() && manifest_path.is_file() {
                let manifest = ModManifest::read(&manifest_path)?;
                manifests.push((manifest, path));
            }
        }

        manifests.sort_by(|(a, _), (b, _)| a.id.cmp(&b.id));

        let mut by_id = BTreeMap::new();
        for (manifest, root) in manifests {
            if by_id
                .insert(manifest.id.clone(), Mod { manifest, root })
                .is_some()
            {
                return Err(anyhow!("two mods use the same id"));
            }
        }

        let order = resolve_order(&by_id)?;

        Ok(Self { mods: by_id, order })
    }

    pub fn mods(&self) -> impl Iterator<Item = &Mod> {
        self.order.iter().filter_map(|id| self.mods.get(id))
    }

    pub fn root(&self, id: &ModId) -> Option<&Path> {
        self.mods.get(id).map(|package| package.root.as_path())
    }

    /// The path of an asset, checked so that it cannot leave its mod's root.
    pub fn resolve(&self, asset: &AssetId) -> anyhow::Result<PathBuf> {
        let package = self
            .mods
            .get(&asset.mod_id)
            .ok_or_else(|| anyhow!("mod {} is not loaded", asset.mod_id))?;

        if asset.path.is_absolute() {
            return Err(anyhow!("asset {} is an absolute path", asset.display()));
        }

        if !confined(&asset.path) {
            return Err(anyhow!("asset {} leaves the mod root", asset.display()));
        }

        Ok(package.root.join(&asset.path))
    }

    pub fn read(&self, asset: &AssetId) -> anyhow::Result<Vec<u8>> {
        let path = self.resolve(asset)?;
        fs::read(&path).with_context(|| format!("failed to read the asset {}", path.display()))
    }

    /// The whole text of an asset, which the shaders and the mod manifests are
    /// read from.
    pub fn text(&self, asset: &AssetId) -> anyhow::Result<String> {
        let path = self.resolve(asset)?;
        fs::read_to_string(&path)
            .with_context(|| format!("failed to read the asset {}", path.display()))
    }

    /// The file of the merged namespace at `path`.
    ///
    /// The namespace is what the assets of the mods would be if they were one
    /// package: a mod that loads later replaces the file of the mods before it,
    /// so a mod can override the files it depends on.
    pub fn asset(&self, path: &Path) -> anyhow::Result<AssetId> {
        if !confined(path) {
            return Err(anyhow!("the asset path {} leaves the mod root", path.display()));
        }

        let mut found: Option<AssetId> = None;

        for package in self.mods() {
            if package.root.join(path).is_file() {
                let asset = AssetId::new(package.manifest.id.clone(), path);

                if let Some(previous) = found.replace(asset.clone()) {
                    info!("{} is overridden by {}", previous.display(), asset.display());
                }
            }
        }

        found.ok_or_else(|| anyhow!("no loaded mod provides the asset {}", path.display()))
    }

    /// Every file of the merged directory `dir`, in name order.
    ///
    /// The order is the names of the files rather than the order of the mods, so
    /// that the position of a file, which a [`Registry`](crate::util::collection::Registry)
    /// uses as the id of its entry, does not depend on the file system or on
    /// which mod a file came from.
    pub fn files(&self, dir: &Path) -> anyhow::Result<Vec<AssetId>> {
        if !confined(dir) {
            return Err(anyhow!(
                "the asset directory {} leaves the mod root",
                dir.display()
            ));
        }

        let mut merged: BTreeMap<PathBuf, AssetId> = BTreeMap::new();

        for package in self.mods() {
            let path = package.root.join(dir);
            if !path.is_dir() {
                continue;
            }

            for file in files_in(&path)? {
                let name = file
                    .file_name()
                    .ok_or_else(|| anyhow!("failed to get the name of {}", file.display()))?;
                let asset = AssetId::new(package.manifest.id.clone(), dir.join(name));

                if let Some(previous) = merged.insert(asset.path.clone(), asset.clone()) {
                    info!("{} is overridden by {}", previous.display(), asset.display());
                }
            }
        }

        if merged.is_empty() {
            return Err(anyhow!(
                "no loaded mod provides the directory {}",
                dir.display()
            ));
        }

        Ok(merged.into_values().collect())
    }

    /// The one world program the loaded mods contribute.
    pub fn world_program(&self) -> anyhow::Result<AssetId> {
        let mut programs = Vec::new();

        for package in self.mods() {
            if let Some(world) = &package.manifest.world {
                programs.push(AssetId::new(package.manifest.id.clone(), world.clone()));
            }
        }

        match programs.len() {
            1 => Ok(programs.remove(0)),
            0 => Err(anyhow!("no loaded mod provides a world program")),
            _ => Err(anyhow!(
                "more than one mod provides a world program: {}",
                programs
                    .iter()
                    .map(AssetId::display)
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

/// Whether `path` stays inside the mod it is joined to.
fn confined(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn resolve_order(mods: &BTreeMap<ModId, Mod>) -> anyhow::Result<Vec<ModId>> {
    fn visit(
        id: &str,
        mods: &BTreeMap<ModId, Mod>,
        done: &mut HashSet<ModId>,
        visiting: &mut Vec<ModId>,
        order: &mut Vec<ModId>,
    ) -> anyhow::Result<()> {
        if done.contains(id) {
            return Ok(());
        }

        if visiting.iter().any(|entry| entry == id) {
            visiting.push(id.to_owned());
            return Err(anyhow!(
                "mods depend on each other in a cycle: {}",
                visiting.join(" -> ")
            ));
        }

        let package = mods.get(id).ok_or_else(|| anyhow!("mod {id} is missing"))?;
        visiting.push(id.to_owned());

        for dependency in &package.manifest.dependencies {
            if !mods.contains_key(dependency) {
                return Err(anyhow!("mod {id} depends on the missing mod {dependency}"));
            }

            visit(dependency, mods, done, visiting, order)?;
        }

        visiting.pop();
        done.insert(id.to_owned());
        order.push(id.to_owned());
        Ok(())
    }

    let mut done = HashSet::new();
    let mut visiting = Vec::new();
    let mut order = Vec::new();

    // The core mod is the floor every other mod overrides, so it comes first no
    // matter what its id orders it next to.
    if mods.contains_key(CORE_MOD) {
        visit(CORE_MOD, mods, &mut done, &mut visiting, &mut order)?;
    }

    for id in mods.keys() {
        visit(id, mods, &mut done, &mut visiting, &mut order)?;
    }

    Ok(order)
}
