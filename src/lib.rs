//! The core of the block game: a small entity-component-system, a `wgpu`
//! renderer, the voxel world and the user interface.
//!
//! The binary of the project only installs the panic hook and runs the event
//! loop of [`game::Game`]; everything else lives here so that the modules can be
//! built and tested on their own.

extern crate core;

pub mod actor;
pub mod ecs;
pub mod game;
pub mod render;
pub mod ui;
pub mod util;
pub mod world;
