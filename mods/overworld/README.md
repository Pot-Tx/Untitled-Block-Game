# Overworld mod

The mod is a directory with a `mod.ron`, assets below `assets/`, and the wasm
modules below `scripts/`.

## Build the script

Install the wasm target once, then build the crate:

```text
rustup target add wasm32-unknown-unknown
cargo build --manifest-path mods/overworld/Cargo.toml --target wasm32-unknown-unknown --release
copy mods\overworld\target\wasm32-unknown-unknown\release\overworld.wasm mods\overworld\scripts\terrain.wasm
```

The game reads `assets/world/main.ron` and the module it names. The module must
export `abi_version`, `alloc`, `dealloc`, and the stage entries the manifest
names.

## Structure chunks

`tree.bin` and `camp.bin` are postcard `BlockChunk` documents with a `STRU`
header. Regenerate them with:

```text
python mods/overworld/tools/make_structures.py
```

The Python tool mirrors the run length encoding the game's `BlockChunk` serde
uses; changing that encoding means changing the tool.
