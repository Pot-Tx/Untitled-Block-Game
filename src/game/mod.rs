//! The game loop -- the event loop handler, the timers and the input map --
//! the mod packages the game is loaded from, and the world types a world starts
//! with.

mod client;
mod input;
mod modding;

use crate::ecs::*;
use crate::game::client::GameClient;
use crate::resources;
use anyhow::{Context, anyhow};
use log::error;
use rand::Rng;
use rand::rngs::ThreadRng;
use ron::Options;
use ron::extensions::Extensions;
use serde::{Deserialize, Serialize};
use std::backtrace::Backtrace;
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

pub use input::*;
pub use modding::*;

/// Time between two simulation ticks.
pub(crate) static TICK_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_millis(50));
/// Shortest time between two frames.
pub(crate) static FRAME_DURATION: LazyLock<Duration> = LazyLock::new(|| Duration::from_millis(5));

pub static ASSET_OPTIONS: LazyLock<Options> = LazyLock::new(|| {
    Options::default()
        .with_default_extension(Extensions::UNWRAP_NEWTYPES)
        .with_default_extension(Extensions::IMPLICIT_SOME)
});

/// The directory the saves of the world live in.
///
/// It is named by `--save <dir>`, or by `--save=<dir>`, on the command line;
/// `save` is used when the game was started without either of them.
pub(crate) static SAVE_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    arg_value("save", std::env::args().skip(1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("save"))
});

/// The directory the mods are loaded from.
///
/// It is named by `--mods <dir>`, or by `--mods=<dir>`, on the command line;
/// `mods` is used when the game was started without either of them.
pub(crate) static MODS_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    arg_value("mods", std::env::args().skip(1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("mods"))
});

/// The value `--<name>` names, or `None` when the arguments do not name one.
fn arg_value(name: &str, mut args: impl Iterator<Item = String>) -> Option<String> {
    let joined = format!("--{name}=");
    let separate = format!("--{name}");

    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix(joined.as_str()) {
            return Some(value.to_owned());
        }

        if arg == separate {
            return args.next();
        }
    }

    None
}

resources! {
    /// Whether an open screen pauses the game.
    pub struct Paused(bool);

    /// How many ticks the world has run, counted from the first one.
    ///
    /// Everything that follows the day is derived from this instead of a clock:
    /// a tick is a fixed step, so the same number of ticks always puts the world
    /// in the same state, whatever the machine it runs on and however long it
    /// took to get there.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct TotalTicks(u64): save;
}

/// Counts the ticks of the world; it runs before every system that reads them.
struct Ticker;

impl System<1> for Ticker {
    type CompQueries = ((),);
    type ResQuery = (ResWrite<TotalTicks>,);

    fn update(
        &mut self,
        _: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        res.0.0 += 1;
    }
}

/// Registers the resources and the systems of the game loop.
pub(crate) fn register(frame: &mut SystemManager<2>, resources: &mut ResourceManager) {
    if let Err(e) = init_input_map() {
        error!("failed to load controls: {e}");
    }
    resources.register("input_state", InputState::new());
    resources.register("paused", Paused(false));
    resources.register("total_ticks", TotalTicks(0));

    frame.register(0, "cursor_applier", CursorApplier::new());
}

/// The directory of the world types inside a mod.
const WORLD_DIR: &str = "assets/worlds";

/// One kind of world, as it is written in `assets/worlds/<name>.ron`.
///
/// A world type is the resources a world of that kind starts with: the generator
/// that shapes it and the values that belong to its content. The values are the
/// ones the resources declare they can be read from data with, so a world type
/// names them instead of describing them a second time.
pub struct WorldType {
    /// The name of the world type, which is the file it is read from.
    pub name: String,
    /// The resources a world of this type starts with.
    pub resources: RawResources,
}

impl WorldType {
    /// Reads the world type `name`.
    pub fn load(mods: &ModRegistry, name: &str) -> anyhow::Result<Self> {
        let asset = mods.asset(&Path::new(WORLD_DIR).join(format!("{name}.ron")))?;
        let resources: RawResources = ASSET_OPTIONS
            .from_reader(mods.read(&asset)?.as_slice())
            .with_context(|| format!("failed to read the world type {}", asset.display()))?;

        Ok(Self {
            name: name.to_owned(),
            resources,
        })
    }

    /// The world type the game starts with: the one named by `world`, or the
    /// only one the mods provide.
    pub fn select(mods: &ModRegistry, world: Option<&str>) -> anyhow::Result<Self> {
        if let Some(name) = world {
            return Self::load(mods, name);
        }

        let names = mods
            .files(Path::new(WORLD_DIR))?
            .into_iter()
            .filter_map(|asset| asset.name().ok())
            .collect::<Vec<_>>();

        match names.len() {
            1 => Self::load(mods, &names[0]),
            0 => Err(anyhow!("no loaded mod provides a world type")),
            _ => Err(anyhow!(
                "more than one mod provides a world type; name one with --world: {}",
                names.join(", ")
            )),
        }
    }

    /// Sets the resources of a new world to the values of this world type.
    pub fn apply(&self, resources: &mut ResourceManager) -> anyhow::Result<()> {
        resources
            .read_asset(&self.resources)
            .with_context(|| format!("in the world type {}", self.name))
    }
}

/// Everything the game was started with.
///
/// It is used once, when the window opens: the mods are read from, the world type
/// sets the content of a new world and the seed starts its generator. A world
/// that was saved replaces the values of the world type and of the seed, so the
/// running game keeps nothing of them.
pub(crate) struct Startup {
    pub mods: ModRegistry,
    pub world: WorldType,
    pub seed: u64,
}

/// The game, which owns the client that drives the event loop.
pub struct Game {
    client: GameClient,
}

impl Game {
    /// Creates the game, which loads the mods and reads the world it starts with
    /// from the command line.
    ///
    /// `--world <name>` and `--seed <number>` only apply to a world that has no
    /// session yet: a world that was saved keeps the content it started with.
    pub fn new() -> anyhow::Result<Self> {
        let mods = ModRegistry::load(&MODS_DIR)?;
        // Each argument is looked for on its own pass over the command line, so
        // that the order of the flags does not matter.
        let world = arg_value("world", std::env::args().skip(1));
        let seed = match arg_value("seed", std::env::args().skip(1)) {
            Some(value) => value
                .parse::<u64>()
                .with_context(|| format!("--seed has to be a whole number, not {value}"))?,
            None => ThreadRng::default().next_u64(),
        };
        let world = WorldType::select(&mods, world.as_deref())?;

        Ok(Self {
            client: GameClient::new(Startup { mods, world, seed }),
        })
    }

    /// Registers the components, the systems, the resources and the data of
    /// the game, which the window is not needed for.
    pub fn init(&mut self) {
        self.client.setup();
    }

    /// Runs the event loop of the window until it is closed.
    pub fn run(&mut self) -> Result<(), EventLoopError> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        event_loop.run_app(&mut self.client)
    }

    /// Logs a panic and the backtrace it was raised with.
    ///
    /// Installed as the panic hook of the process, so that a crash ends up in
    /// the log instead of on a console that a release build does not have.
    pub fn crash(info: &panic::PanicHookInfo) {
        error!("game crashed: {}", info);
        error!("{}", Backtrace::capture());
    }
}
