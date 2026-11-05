//! The persistent identity of the entities of the simulation world.
//!
//! The world streams its regions in and out of memory, while the entities live
//! scattered over the flat [`EntityManager`]. The store is the bridge between
//! the two: it keeps the identity of every entity that is in memory, and it is
//! the only place that knows which region an entity belongs to.

use super::{Persistent, Position, SessionEntity};
use crate::ecs::*;
use crate::game::SAVE_DIR;
use crate::util;
use crate::util::Id;
use crate::world::{RegionPos, SAVE_DURATION, World, region_path};
use bimap::BiMap;
use crossbeam_channel::{Receiver, Sender, unbounded};
use log::{error, warn};
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

/// The identity of an entity in a save.
///
/// [`Id`] is a runtime index that the manager hands out again once the entity
/// dies, so it cannot name an entity in a file. A handle is never reused, which
/// is what lets a save and a reference point at an entity across sessions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Handle(u64);

impl Handle {
    /// The handle the first entity of a world is given.
    pub const ZERO: Self = Self(0);

    /// The handle after this one, which is never handed out again.
    #[inline]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for Handle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The magic of the file the entities of a region are saved in.
const ENTITY_MAGIC: &[u8; 4] = b"ENTS";
/// The version of the entity file format.
const ENTITY_VERSION: u32 = 0;
/// The magic of the session file.
const SESSION_MAGIC: &[u8; 4] = b"BSES";
/// The version of the session file format.
const SESSION_VERSION: u32 = 0;

/// The directory the entities of the regions are saved in.
static ENTITY_DIR: LazyLock<PathBuf> = LazyLock::new(|| SAVE_DIR.join("entities"));
/// The file the entities that are not part of a region and the state of the
/// world live in.
static SESSION_FILE: LazyLock<PathBuf> = LazyLock::new(|| SAVE_DIR.join("session.sav"));

/// Which entities a bucket holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every live entity with a position; a snapshot of the scope writes these.
    All,
    /// Only the entities marked [`Persistent`]; these are the ones a region
    /// keeps when it leaves the scope.
    Persistent,
}

/// The entities of a region that a read returned, or the error it failed with.
type LoadResult = (RegionPos, anyhow::Result<Option<EntitySet>>);
/// The outcome of writing the entities of a region.
type WriteReport = (RegionPos, anyhow::Result<()>);

/// The persistent identity of the entities that are in memory.
pub struct EntityStore {
    /// The identity to hand out next.
    next_handle: Handle,
    /// The identity of every entity that is in memory, in both directions: a
    /// save and a reference name an entity by its handle, the ECS by its id.
    identities: BiMap<Handle, Id>,
    /// The regions whose entities still have to be written.
    pending: HashMap<RegionPos, PendingWrite>,
    /// The regions of the scope that already have a file.
    saved: HashSet<RegionPos>,
    /// The regions of the scope, so that a read that comes back after the
    /// player walked past the region can be dropped.
    resident: HashSet<RegionPos>,
    /// The regions whose entities are being read.
    loading: HashSet<RegionPos>,
    /// The regions whose entities are being written.
    writing: HashSet<RegionPos>,
    /// When the scope was last written back to disk.
    save_timer: Instant,

    /// The results of the reads and writes that run on the runtime.
    load_tx: Sender<LoadResult>,
    load_rx: Receiver<LoadResult>,
    write_tx: Sender<WriteReport>,
    write_rx: Receiver<WriteReport>,
}

/// The entities of a region that wait for their file to be written.
struct PendingWrite {
    set: EntitySet,
    /// How often writing this file has failed.
    attempts: u8,
}

impl EntityStore {
    /// How often writing a region file is retried before it is given up on.
    const MAX_WRITE_ATTEMPTS: u8 = 3;
    /// How many write rounds the exit flush runs, which is enough for the
    /// writes that were already running and the ones that failed.
    const MAX_FLUSH_ROUNDS: usize = 2;

    pub fn new() -> Self {
        let (load_tx, load_rx) = unbounded();
        let (write_tx, write_rx) = unbounded();

        Self {
            next_handle: Handle::ZERO,
            identities: BiMap::new(),
            pending: HashMap::new(),
            saved: HashSet::new(),
            resident: HashSet::new(),
            loading: HashSet::new(),
            writing: HashSet::new(),
            save_timer: Instant::now(),

            load_tx,
            load_rx,
            write_tx,
            write_rx,
        }
    }

    /// Drops the identity of the entities that died since the last flush,
    /// retries the writes that failed and writes the whole scope back to disk
    /// every [`SAVE_DURATION`].
    ///
    /// The identities have to be dropped before the id of a dead entity is
    /// handed out again: the manager recycles an id as soon as the entity dies,
    /// while its components stay in the storage until the next flush, so the
    /// entity that gets the id next would otherwise inherit the identity of the
    /// one before it.
    pub fn sync(&mut self, entities: &EntityManager) {
        {
            let commander = entities.commander();
            let guard = commander.lock();

            for &entity in guard.removed() {
                let _ = self.identities.remove_by_right(&entity);
            }
        }

        self.poll_loads(entities);
        self.poll_writes();
        self.begin_writing();

        if self.save_timer.elapsed() >= *SAVE_DURATION {
            self.save_timer = Instant::now();

            self.snapshot(entities);
            self.begin_writing();
        }
    }

    /// The handle of `entity`, giving it one when it has none yet.
    pub fn identity(&mut self, entity: Id) -> Handle {
        if let Some(&handle) = self.identities.get_by_right(&entity) {
            return handle;
        }

        let handle = self.next_handle;
        self.next_handle = self.next_handle.next();
        self.bind(handle, entity);

        handle
    }

    /// The entity a handle names, as long as it is in memory.
    #[inline]
    pub fn entity(&self, handle: Handle) -> Option<Id> {
        self.identities.get_by_left(&handle).copied()
    }

    /// The live entities of `scope`, bucketed by the region their position
    /// falls in.
    ///
    /// The components of a despawned entity outlive it until the next flush, so
    /// every entity is checked against the manager before it is bucketed.
    pub fn bucket(&self, entities: &EntityManager, scope: Scope) -> HashMap<RegionPos, Vec<Id>> {
        let mut buckets = HashMap::new();

        let Ok(position) = entities.components.get::<Position>() else {
            return buckets;
        };

        let marker = match scope {
            Scope::All => None,
            Scope::Persistent => match entities.components.get::<Persistent>() {
                Ok(marker) => Some(marker),
                Err(_) => return buckets,
            },
        };

        let session = entities.components.get::<SessionEntity>().ok();
        let commander = entities.commander();
        let guard = commander.lock();

        for (entity, pos) in position.iter::<Position>() {
            if !guard.contains(entity) {
                continue;
            }

            if session.is_some_and(|marker| marker.contains(entity)) {
                continue;
            }

            if let Some(marker) = marker
                && !marker.contains(entity)
            {
                continue;
            }

            let region = World::cast_pos(pos.cur.floor().as_ivec3()).0;
            buckets.entry(region).or_default().push(entity);
        }

        buckets
    }

    /// Reads the session, if there is one, and continues the identities where
    /// it left off.
    pub fn read_session(&mut self) -> Option<Session> {
        match Session::read_from() {
            Ok(Some(session)) => {
                self.next_handle = session.next_handle;
                Some(session)
            }
            Ok(None) => None,
            Err(e) => {
                error!("failed to read the session: {e}");
                None
            }
        }
    }

    /// Spawns the entities of the session, which are the ones that are not part
    /// of a region.
    pub fn load_session(&mut self, entities: &EntityManager, session: &Session) {
        let descriptors = session.entities.descriptors(&entities.components);

        if descriptors.is_empty() {
            return;
        }

        let mut guard = entities.commander().lock();

        for (handle, descriptor) in descriptors {
            let entity = guard.spawn(descriptor.with(SessionEntity));
            self.bind(handle, entity);
        }
    }

    /// Writes the session: the entities that are not part of a region and the
    /// state the world needs before any region is loaded.
    pub fn write_session(
        &mut self,
        resources: &ResourceManager,
        entities: &EntityManager,
        seed: u64,
    ) {
        let entries = self.session_entries(entities);

        let set = match EntitySet::build(&entities.components, &entries) {
            Ok(set) => set,
            Err(e) => {
                error!("failed to encode the entities of the session: {e}");
                return;
            }
        };

        let resources = match ResourceSet::build(resources) {
            Ok(resources) => resources,
            Err(e) => {
                error!("failed to encode the resource state: {e}");
                return;
            }
        };

        let session = Session {
            seed,
            next_handle: self.next_handle,
            entities: set,
            resources,
        };

        if let Err(e) = session.write_to() {
            error!("failed to write the session: {e}");
        }
    }

    /// Starts reading the entities of the regions that entered the scope; the
    /// ones that come back are spawned by [`Self::sync`].
    pub fn load(&mut self, entered: &[RegionPos]) {
        for &pos in entered {
            self.resident.insert(pos);

            if !self.loading.insert(pos) {
                continue;
            }

            let tx = self.load_tx.clone();

            crate::RUNTIME.spawn_blocking(move || {
                let result = EntitySet::read_from(pos);

                if let Err(e) = tx.try_send((pos, result)) {
                    error!("failed to report the entities of the region {pos}: {e}");
                }
            });
        }
    }

    /// Writes the entities of the regions that left the scope and drops them.
    ///
    /// Only the entities marked [`Persistent`] are written; the others end with
    /// the region. A region that already had a file is rewritten even when
    /// nothing is left, because an empty file is what tells a later load that
    /// the region was populated already and must not be filled again.
    pub fn evict(&mut self, entities: &EntityManager, left: &[RegionPos]) {
        if left.is_empty() {
            return;
        }

        let all = self.bucket(entities, Scope::All);
        let persistent = self.bucket(entities, Scope::Persistent);

        let mut guard = entities.commander().lock();

        for &pos in left {
            self.resident.remove(&pos);

            let keep = persistent.get(&pos).map(Vec::as_slice).unwrap_or_default();
            let entries = self.entries(keep);

            // A region that never held an entity has nothing to record, so
            // walking through empty land does not litter it with empty files.
            if self.saved.contains(&pos) || !entries.is_empty() {
                match EntitySet::build(&entities.components, &entries) {
                    Ok(set) => {
                        self.pending.insert(pos, PendingWrite { set, attempts: 0 });
                    }
                    Err(e) => error!("failed to encode the entities of the region {pos}: {e}"),
                }
            }

            self.saved.remove(&pos);

            if let Some(leaving) = all.get(&pos) {
                for &entity in leaving {
                    guard.despawn(entity);
                }
            }
        }
    }

    /// Queues the entities of every region of the scope, so that a save is not
    /// lost when the game ends without the regions leaving it.
    pub fn snapshot(&mut self, entities: &EntityManager) {
        let live = self.bucket(entities, Scope::All);

        for (pos, ids) in &live {
            let entries = self.entries(ids);
            self.queue(*pos, entities, entries);
        }

        // A region whose entities all died still has to be rewritten, or the
        // next load would bring them back.
        let stale = self.saved.iter().copied().collect::<Vec<_>>();

        for pos in stale {
            if !live.contains_key(&pos) {
                self.queue(pos, entities, Vec::new());
            }
        }
    }

    /// Writes the scope and waits for the queued regions before the window
    /// closes.
    pub fn flush(&mut self, entities: &EntityManager) {
        self.snapshot(entities);

        for _ in 0..Self::MAX_FLUSH_ROUNDS {
            self.begin_writing();

            while !self.writing.is_empty() {
                self.poll_writes();

                if self.writing.is_empty() {
                    break;
                }

                std::thread::sleep(Duration::from_millis(1));
            }
        }

        if !self.pending.is_empty() {
            warn!(
                "gave up writing the entities of {} region(s)",
                self.pending.len()
            );

            self.pending.clear();
        }
    }

    /// The handles and ids of the entities of `ids`, giving a handle to the
    /// entities that do not have one yet.
    fn entries(&mut self, ids: &[Id]) -> Vec<(Handle, Id)> {
        ids.iter()
            .map(|&entity| (self.identity(entity), entity))
            .collect()
    }

    /// Queues the entities of a region to be written.
    fn queue(&mut self, pos: RegionPos, entities: &EntityManager, entries: Vec<(Handle, Id)>) {
        match EntitySet::build(&entities.components, &entries) {
            Ok(set) => {
                self.pending.insert(pos, PendingWrite { set, attempts: 0 });
                self.saved.insert(pos);
            }
            Err(e) => error!("failed to encode the entities of the region {pos}: {e}"),
        }
    }

    /// Spawns the entities of the regions whose save has been read.
    fn poll_loads(&mut self, entities: &EntityManager) {
        for (pos, result) in self.load_rx.try_iter().collect::<Vec<_>>() {
            self.loading.remove(&pos);

            // The player may have walked past the region while it was read.
            if !self.resident.contains(&pos) {
                continue;
            }

            let set = match result {
                Ok(Some(set)) => set,
                Ok(None) => continue,
                Err(e) => {
                    error!("failed to read the entities of region {pos}: {e}");
                    continue;
                }
            };

            self.saved.insert(pos);

            let mut guard = entities.commander().lock();

            for (handle, descriptor) in set.descriptors(&entities.components) {
                let entity = guard.spawn(descriptor);
                self.bind(handle, entity);
            }
        }
    }

    /// Starts writing the regions whose entities are queued.
    fn begin_writing(&mut self) {
        for (&pos, write) in self.pending.iter() {
            if !self.writing.insert(pos) {
                continue;
            }

            let set = write.set.clone();
            let tx = self.write_tx.clone();

            crate::RUNTIME.spawn_blocking(move || {
                let result = set.write_to(pos);

                if let Err(e) = tx.try_send((pos, result)) {
                    error!("failed to report the entities of the region {pos}: {e}");
                }
            });
        }
    }

    /// Applies the writes that finished, and keeps the ones that failed.
    fn poll_writes(&mut self) {
        for (pos, result) in self.write_rx.try_iter().collect::<Vec<_>>() {
            self.writing.remove(&pos);

            match result {
                Ok(()) => {
                    self.pending.remove(&pos);
                }

                Err(e) => {
                    let attempts = match self.pending.get_mut(&pos) {
                        Some(write) => {
                            write.attempts += 1;
                            write.attempts
                        }
                        None => continue,
                    };

                    if attempts < Self::MAX_WRITE_ATTEMPTS {
                        warn!("failed to write the entities of the region {pos}: {e}");
                    } else {
                        error!("gave up writing the entities of the region {pos}: {e}");
                        self.pending.remove(&pos);
                    }
                }
            }
        }
    }

    /// The handles and ids of the session entities that are in memory.
    fn session_entries(&mut self, entities: &EntityManager) -> Vec<(Handle, Id)> {
        let Ok(marker) = entities.components.get::<SessionEntity>() else {
            return Vec::new();
        };

        let ids = {
            let commander = entities.commander();
            let guard = commander.lock();

            marker
                .iter::<SessionEntity>()
                .filter(|(entity, _)| guard.contains(*entity))
                .map(|(entity, _)| entity)
                .collect::<Vec<_>>()
        };

        self.entries(&ids)
    }

    /// Binds `entity` to `handle`, which both have to be free.
    fn bind(&mut self, handle: Handle, entity: Id) {
        if let Err((handle, entity)) = self.identities.insert_no_overwrite(handle, entity) {
            error!("failed to bind entity {entity} to the identity {handle}: it is taken");
        }
    }
}

/// The entities of one file: the components they use, named once, and the
/// payload of each of them.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct EntitySet {
    /// The components used in this file together with the version of the shape
    /// they were written with.
    pub names: Vec<(String, u8)>,
    /// One entry per entity, in the order the entities were collected.
    pub entities: Vec<SavedEntity>,
}

/// One entity: its identity and the payload of each of its components.
#[derive(Clone, Serialize, Deserialize)]
pub struct SavedEntity {
    /// The identity the entity keeps across saves; see [`Handle`].
    pub handle: Handle,
    /// The payload of each component, as an index into the name table.
    pub components: Vec<(u16, Vec<u8>)>,
}

/// The state of the world that is not part of a region.
#[derive(Serialize, Deserialize)]
pub struct Session {
    /// The seed the world is generated with.
    pub seed: u64,
    /// The identity the store hands out next.
    pub next_handle: Handle,
    /// The entities that do not belong to a region, which is the player.
    pub entities: EntitySet,
    /// The global state of the world that the ECS stores as resources.
    pub resources: ResourceSet,
}

impl EntitySet {
    /// Collects the components of the entities, which the store passes as pairs
    /// of the identity of an entity and the entity that identity names.
    pub fn build(components: &ComponentManager, entities: &[(Handle, Id)]) -> anyhow::Result<Self> {
        let registered = components
            .registered()
            .filter(|(_, component)| component.settings.save.is_some())
            .collect::<Vec<_>>();

        // The components the entities use at all, in the order of the names.
        let used = registered
            .iter()
            .map(|(_, component)| {
                entities
                    .iter()
                    .any(|(_, entity)| component.contains(*entity))
            })
            .collect::<Vec<_>>();

        // The name table is written once per file and every component indexes
        // into it, which keeps a save from repeating the component names.
        let mut names = Vec::new();
        let mut index = vec![0u16; registered.len()];

        for (i, (name, component)) in registered.iter().enumerate() {
            if used[i] {
                index[i] = names.len() as u16;
                names.push((String::from(*name), component.save_version()));
            }
        }

        let mut saved = Vec::with_capacity(entities.len());

        for (handle, entity) in entities {
            let mut payloads = Vec::new();

            for (i, (_, component)) in registered.iter().enumerate() {
                if used[i]
                    && let Some(value) = component.erased_ref(*entity)
                {
                    payloads.push((index[i], component.write_save(value)?));
                }
            }

            saved.push(SavedEntity {
                handle: *handle,
                components: payloads,
            });
        }

        Ok(Self {
            names,
            entities: saved,
        })
    }

    /// The entities of the set, in the order they were saved.
    ///
    /// A component this build does not know, or one whose shape changed, is
    /// left out with a warning, so an entity keeps the components that are
    /// still readable.
    pub fn descriptors(&self, components: &ComponentManager) -> Vec<(Handle, EntityDescriptor)> {
        let resolved = self
            .names
            .iter()
            .map(|(name, version)| Self::resolve(components, name, *version))
            .collect::<Vec<_>>();

        self.entities
            .iter()
            .map(|saved| {
                let mut descriptor = EntityDescriptor::new();

                for (index, payload) in saved.components.iter() {
                    let Some(component) = resolved.get(*index as usize).and_then(|c| *c) else {
                        continue;
                    };

                    match component.read_save(payload) {
                        Ok(value) => {
                            descriptor.values.insert(component.id, value);
                        }
                        Err(e) => error!("failed to read component \"{}\": {}", component.name, e),
                    }
                }

                (saved.handle, descriptor)
            })
            .collect()
    }

    /// The component a name and a shape version name, when this build has it.
    fn resolve<'a>(
        components: &'a ComponentManager,
        name: &str,
        version: u8,
    ) -> Option<&'a ErasedComponent> {
        let component = match components.by_name(name) {
            Ok(component) => component,
            Err(_) => {
                warn!("a save names the component {name}, which this build does not have");
                return None;
            }
        };

        if component.settings.save.is_none() {
            warn!("the component {name} cannot be read from a save");
            return None;
        }

        if component.save_version() != version {
            warn!(
                "a save holds the component {name} of shape {version}, this build reads {}",
                component.save_version()
            );
            return None;
        }

        Some(component)
    }

    /// Reads the entities of `pos`, or `None` when the region has no save.
    pub fn read_from(pos: RegionPos) -> anyhow::Result<Option<Self>> {
        util::read_file(&region_path(&ENTITY_DIR, pos), ENTITY_MAGIC, ENTITY_VERSION)
    }

    /// Writes the entities of `pos`.
    pub fn write_to(&self, pos: RegionPos) -> anyhow::Result<()> {
        util::write_file(
            &region_path(&ENTITY_DIR, pos),
            ENTITY_MAGIC,
            ENTITY_VERSION,
            self,
        )
    }
}

impl Session {
    /// The region the player stands in, which is where the world is centered
    /// before any region has been loaded.
    pub fn player_region(&self, components: &ComponentManager) -> Option<RegionPos> {
        let (_, descriptor) = self.entities.descriptors(components).into_iter().next()?;
        let position = descriptor.values.get(&TypeId::of::<Position>())?;

        Some(World::cast_pos(position.cast::<Position>().cur.floor().as_ivec3()).0)
    }

    /// Reads the session, or `None` when the world was never saved.
    pub fn read_from() -> anyhow::Result<Option<Self>> {
        let bytes = match fs::read(SESSION_FILE.as_path()) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        decode_session(&bytes).map(Some)
    }

    /// Writes the session.
    pub fn write_to(&self) -> anyhow::Result<()> {
        util::write_file(&SESSION_FILE, SESSION_MAGIC, SESSION_VERSION, self)
    }
}

/// Decodes a session, upgrading the format used before resources were saved.
fn decode_session(bytes: &[u8]) -> anyhow::Result<Session> {
    if bytes.len() < 8 || &bytes[0..4] != SESSION_MAGIC {
        return Err(anyhow::anyhow!(
            "{} is not a save file",
            SESSION_FILE.display()
        ));
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let payload = &bytes[8..];

    match version {
        SESSION_VERSION => postcard::from_bytes(payload)
            .map_err(|e| anyhow::anyhow!("failed to decode {}: {e}", SESSION_FILE.display())),

        found => Err(anyhow::anyhow!(
            "{} has the version {found}, which this build cannot read",
            SESSION_FILE.display()
        )),
    }
}
