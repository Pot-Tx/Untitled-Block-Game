# Core mod

The content the game ships with: a directory with a `mod.ron` at its root, the
files the game reads below `assets/`, and the tools that bake their content.

| Directory | Contents |
|---|---|
| `assets/actors/` | the actor types, one file per type |
| `assets/blocks/` | the block types |
| `assets/generators/` | the world generation programs, one file per generator |
| `assets/models/` | the block models and the templates they are built from |
| `assets/screens/` | the screens of the interface and their elements |
| `assets/shaders/` | the wgsl sources of every pipeline |
| `assets/sprites/` | the sprite sets and the animations that drive them |
| `assets/structures/` | the structures a generator places, with their content |
| `assets/textures/` | the textures of the blocks, actors and interface |
| `assets/fonts/` | the font of the interface |
| `assets/worlds/` | the world types: the resources a world starts with |
| `tools/` | the tools that bake the structure content |

A file is named after the entry it describes, and a mod that loads after this
one replaces a file of the same path. The `core` mod is always the first mod
loaded, so every other mod may override any of these files.

## The generator

A generator is data: `assets/generators/overworld.ron` names the fields a region
samples, the rule that turns their samples into blocks, and the structures it
places. Nothing of it is code, so a mod may replace it with a file of its own.

## Structure chunks

`assets/structures/tree.ron` and `camp.ron` name a postcard `BlockChunk`
document with a `STRU` header, which sits next to them. A cell of such a
document is one of three things:

| Cell | Meaning |
|---|---|
| `0xFFFF` | the cell is not part of the structure; the world keeps the block it has |
| `0` | air, which clears the block the world has |
| other | the meta of a block, which replaces the block the world has |

A structure volume therefore starts out as `0xFFFF`, and only the cells the
structure fills in become blocks; a volume that starts out as `0` carves its
whole box out of the terrain. Regenerate the documents with:

```text
python mods/core/tools/make_structures.py
```

The Python tool mirrors the run length encoding the game's `BlockChunk` serde
uses; changing that encoding means changing the tool.

### Placement variants

A structure may be placed turned and mirrored:

```ron
rotations: [0, 1, 2, 3],   // quarter turns, default [0]
mirror: true,              // also mirrored along X, default false
```

Each allowed transform is baked when the structure is loaded: it gets its own
content, collision box, anchor and actor positions, so stamping never rotates
anything. Transforms that end up with the same content collapse into one, which
is why `tree.ron` has four variants and not eight. A placement stage picks the
variant of every site it returns (the low three bits of the candidate flags); a
structure without rotations has exactly one variant.
