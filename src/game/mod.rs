//! The game loop -- the event loop handler, the timers and the input map --
//! and the mod packages the game is loaded from.

mod client;
mod input;
mod modding;

use crate::ecs::*;
use crate::game::client::GameClient;
use crate::resources;
use log::error;
use ron::Options;
use ron::extensions::Extensions;
use serde::{Deserialize, Serialize};
use std::backtrace::Backtrace;
use std::fs::File;
use std::io::BufReader;
use std::panic;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

pub use input::*;
pub use modding::{AssetId, Mod, ModId, ModManifest, ModRegistry};

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
    arg_dir_of("save", std::env::args().skip(1)).unwrap_or_else(|| PathBuf::from("save"))
});

/// The directory the mods are loaded from.
///
/// It is named by `--mods <dir>`, or by `--mods=<dir>`, on the command line;
/// `mods` is used when the game was started without either of them.
pub(crate) static MODS_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    arg_dir_of("mods", std::env::args().skip(1)).unwrap_or_else(|| PathBuf::from("mods"))
});

/// The directory `--<name>` names, or `None` when the arguments do not name one.
fn arg_dir_of(name: &str, mut args: impl Iterator<Item = String>) -> Option<PathBuf> {
    let joined = format!("--{name}=");
    let separate = format!("--{name}");

    while let Some(arg) = args.next() {
        if let Some(dir) = arg.strip_prefix(joined.as_str()) {
            return Some(PathBuf::from(dir));
        }

        if arg == separate {
            return args.next().map(PathBuf::from);
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

/// Loads the resource settings that are part of the game instead of a save.
pub(crate) fn init_resource_assets(resources: &mut ResourceManager) -> anyhow::Result<()> {
    let file = File::open("assets/settings/resources.ron")?;
    let raw: RawResources = ASSET_OPTIONS.from_reader(BufReader::new(file))?;
    resources.read_asset(&raw)
}

/// The game, which owns the client that drives the event loop.
#[derive(Default)]
pub struct Game {
    client: GameClient,
}

impl Game {
    /// Creates the game; everything it needs is registered by [`Self::init`].
    pub fn new() -> Self {
        Self::default()
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
