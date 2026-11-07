//! The core of the block game: a small entity-component-system, a `wgpu`
//! renderer, the voxel world and the user interface.
//!
//! The binary of the project only installs the panic hook and runs the event
//! loop of [`game::Game`]; everything else lives here so that the modules can be
//! built on their own.

use std::sync::LazyLock;
use tokio::runtime::{Builder, Runtime};

/// The runtime every background task of the game is spawned on.
pub(crate) static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("io")
        .build()
        .expect("failed to build io runtime")
});

pub mod actor;
pub mod ecs;
pub mod game;
pub mod generation;
pub mod render;
pub mod ui;
pub mod util;
pub mod world;
