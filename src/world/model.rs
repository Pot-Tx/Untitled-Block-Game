//! Block models and the mesher that turns chunks into geometry.
//!
//! A block model is a list of parts, each of which is a mesh template with a
//! texture. Parts of the same model that share a texture are merged into
//! rectangles before they are uploaded, which is what keeps the vertex count of
//! a chunk low.

use crate::game::ASSET_OPTIONS;
use crate::render::*;
use crate::util::Id;
use crate::util::OnceInit;
use crate::util::collection::Registry;
use crate::util::coord::{Axis, Coord3, Direction, ICoord3};
use crate::util::files_in;
use crate::world::*;
use anyhow::{Context, anyhow};
use bytemuck::{Pod, Zeroable};
use crossbeam_channel::{Receiver, Sender};
use glam::{U8Vec3, Vec2, Vec3};
use log::error;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::array;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::mem::offset_of;
use std::path::Path;
use wgpu::*;

/// What the cpu reads of the templates the block models are built from, loaded from
/// `assets/models/block/templates`.
pub(super) static BLOCK_MODEL_TEMPLATES: OnceInit<Registry<TemplateProperties>> = OnceInit::new();

/// Those templates as the vertex shaders of the passes that draw them read them: one
/// entry per template, in the order of the template ids.
pub(super) static BLOCK_TEMPLATES: OnceInit<BindSet<BlockTemplates>> = OnceInit::new();

/// The textures of the blocks, loaded from `assets/textures/block`.
pub(crate) static BLOCK_TEXTURES: OnceInit<BindSet<TextureArraySampler>> = OnceInit::new();

/// The color and material arrays used by the world block pass.
pub(crate) static BLOCK_SURFACES: OnceInit<BindSet<BlockTextureSet>> = OnceInit::new();
pub(crate) static BLOCK_TEXTURE_HANDLES: OnceInit<Registry<bool>> = OnceInit::new();
pub(crate) static BLOCK_MATERIAL_HANDLES: OnceInit<Registry<bool>> = OnceInit::new();

/// The patterns of the ambient occlusion of the blocks, loaded from
/// `assets/textures/occlusion`.
pub(crate) static OCCLUSION_TEXTURES: OnceInit<BindSet<TextureArraySampler>> = OnceInit::new();
pub(crate) static OCCLUSION_TEXTURE_HANDLES: OnceInit<Registry<()>> = OnceInit::new();

/// Builds the block model templates the block models are built from.
pub(super) fn init_block_model_templates(canvas: &Canvas) -> anyhow::Result<()> {
    let (reg, buffer) = load_template_array(canvas, "assets/models/block/templates")?;

    let binding = BindSet::new(
        canvas,
        &BindSetConfig {
            name: "block_template",
            content: &buffer,
        },
    );

    BLOCK_MODEL_TEMPLATES.init(reg);
    BLOCK_TEMPLATES.init(binding);
    Ok(())
}

/// Reads the templates of `path` and uploads them as the table the passes that draw
/// blocks read.
///
/// A template is loaded in two halves, like a texture: the face the vertex shaders read
/// goes into the table, which is dropped once it is uploaded, and what the cpu reads of
/// it goes into the registry, which stays. A file that holds a face the shaders could not
/// read the way the mesher draws it fails the load, with the name of the file it came
/// from.
pub(super) fn load_template_array<P: AsRef<Path>>(
    canvas: &Canvas,
    path: P,
) -> anyhow::Result<(Registry<TemplateProperties>, Buffer)> {
    let mut registry = Registry::new();
    let mut faces = Vec::new();

    for (id, path) in files_in(path.as_ref())?.into_iter().enumerate() {
        let name = path
            .file_stem()
            .ok_or(anyhow!("failed to get file name"))?
            .to_string_lossy()
            .into_owned();
        let file = File::open(&path)?;
        let raw: RawTemplate = ASSET_OPTIONS
            .from_reader(BufReader::new(file))
            .with_context(|| format!("failed to read {}", path.display()))?;
        let (face, properties) = raw
            .halves()
            .with_context(|| format!("in the template {name}"))?;

        registry.register(id as Id, name, properties);
        faces.push(face);
    }

    let buffer = Buffer::new(
        canvas,
        &BufferConfig {
            name: "block_template",
            init: BufferInit::Content(&faces),
            usage: BufferUsages::STORAGE,
        },
    );

    Ok((registry, buffer))
}

/// Builds the block textures.
pub(super) fn init_block_textures(canvas: &Canvas) -> anyhow::Result<()> {
    let (color_handles, _, color_view) = load_texture_array(
        canvas,
        "assets/textures/block",
        "block",
        TextureFormat::Rgba8UnormSrgb,
        true,
        |tex| {
            tex.data
                .chunks_exact(4)
                .any(|pixel| pixel[3] > 0 && pixel[3] < 255)
        },
    )?;

    let (material_handles, _, material_view) = load_texture_array(
        canvas,
        "assets/textures/block/material",
        "block_material",
        TextureFormat::Rgba8Unorm,
        false,
        |tex| tex.data.chunks_exact(4).any(|pixel| pixel[3] > 0),
    )?;

    let sampler = Sampler::new(
        canvas,
        &SamplerConfig {
            name: "block",
            filter: FilterMode::Nearest,
            address_mode: AddressMode::Repeat,
            mipmap_filter: MipmapFilterMode::Linear,
            mip_level_count: 1,
            compare: None,
        },
    );

    let textures = BindSet::new(
        canvas,
        &BindSetConfig {
            name: "block",
            content: (&color_view, &sampler),
        },
    );

    let surfaces = BindSet::new(
        canvas,
        &BindSetConfig {
            name: "block",
            content: (&color_view, &material_view, &sampler),
        },
    );

    BLOCK_TEXTURE_HANDLES.init(color_handles);
    BLOCK_MATERIAL_HANDLES.init(material_handles);
    BLOCK_TEXTURES.init(textures);
    BLOCK_SURFACES.init(surfaces);
    Ok(())
}

/// The color texture array, material texture array and their shared sampler.
pub struct BlockTextureSet;

impl BindSignature for BlockTextureSet {
    const NAME: &'static str = "block_surface";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
    ];
    type Content<'a> = (&'a TextureView, &'a TextureView, &'a Sampler);
}

/// Builds the occlusion textures.
///
/// A pattern holds the occlusion of the four corners of a face and is named
/// after its code, which counts the corners up in base three in the order they
/// appear in the vertices of a template; a corner that nothing shades is the
/// first of the three values of the alpha of the texture. The codes are zero
/// padded, so that the name of a pattern orders it where its code does and its
/// layer is the code itself.
///
/// The patterns are mipmapped, because a face repeats its pattern once per
/// block: at a distance one pixel covers many of them, and the levels that
/// average them away are what keeps the pattern from turning into a moire over
/// the ground.
pub(super) fn init_occlusion_textures(canvas: &Canvas) -> anyhow::Result<()> {
    let (reg, _, view) = load_texture_array(
        canvas,
        "assets/textures/occlusion",
        "occlusion",
        TextureFormat::Rgba8UnormSrgb,
        true,
        |_| (),
    )?;

    let sampler = Sampler::new(
        canvas,
        &SamplerConfig {
            name: "occlusion",
            filter: FilterMode::Nearest,
            address_mode: AddressMode::Repeat,
            mipmap_filter: MipmapFilterMode::Linear,
            mip_level_count: 1,
            compare: None,
        },
    );

    let binding = BindSet::new(
        canvas,
        &BindSetConfig {
            name: "occlusion",
            content: (&view, &sampler),
        },
    );

    OCCLUSION_TEXTURES.init(binding);
    OCCLUSION_TEXTURE_HANDLES.init(reg);
    Ok(())
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(into = "Vec<BlockModelPart>")]
#[serde(from = "Vec<BlockModelPart>")]
pub struct BlockModel {
    parts: Vec<BlockModelPart>,
    /// Whether the face in the direction at the index is culled by an opaque
    /// neighbor; indexed by [`Direction::idx`].
    cull: [Option<bool>; 6],
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
#[serde(into = "RawBlockModelPart")]
#[serde(from = "RawBlockModelPart")]
pub struct BlockModelPart {
    pub template: Id,
    pub texture: Id,
    pub material: Id,
    pub no_cull: bool,
}

impl Hash for BlockModelPart {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.template.hash(state);
        self.texture.hash(state);
        self.material.hash(state);
    }
}

/// A block model part as it is written to a `.ron` file, with names instead of
/// ids so that the registries can be reordered without breaking the files.
#[derive(Serialize, Deserialize)]
struct RawBlockModelPart<'a> {
    template: &'a str,
    texture: &'a str,
    material: &'a str,
    #[serde(default)]
    no_cull: bool,
}

/// One corner of a block model template, as the vertex shaders of the passes that
/// draw blocks read it.
///
/// The padding fields make the layout match the one WGSL gives this data, where a
/// `vec3` member is aligned to 16 bytes: the `uv` that follows a position starts at
/// 16 there, and the struct is rounded up to a multiple of 16, which is 32 bytes. A
/// `Pod` type may not hold padding that is never written out, so the padding WGSL
/// leaves implicit is spelled here instead.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TemplateVert {
    /// Position in block coordinates.
    pub pos: Vec3,
    _pad0: f32,
    /// Texture coordinates.
    pub uv: Vec2,
    _pad1: Vec2,
}

/// A face of a block model template, as the vertex shaders of the passes that draw
/// blocks read it.
///
/// A face is a quad: the four corners of `verts`, in the order of the indices of
/// [`QUAD_INDICES`], which share the normal `norm` and lie in its plane.
///
/// This is the half of a template that is uploaded: like the pixels of a texture, it is
/// built to be read by a shader and dropped once it is on the gpu, while what the cpu
/// needs of the same face stays in a [`TemplateProperties`]. A value that holds more than
/// one number is packed into a word rather than written as the enumerations it stands
/// for, because WGSL declares every scalar of a buffer as a word: the layout of this
/// struct is the layout of the table.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TemplateModel {
    /// The normal of the face, which is a vector rather than one of the six directions,
    /// because a face of a block model can stand at an angle.
    pub norm: Vec3,
    /// The merges of the face, four bits each: the low four bits are the first merge and
    /// the next four the second, each the axis a merge grows along (or 3 while the slot
    /// holds no merge) followed by the direction the uv of the corners it moves advances
    /// in, counting `+u`, `-u`, `+v` and `-v` up from zero.
    ///
    /// The corners a merge moves are the ones on the far side of its axis, which is where
    /// the shader reads them off.
    pub spans: u32,
    /// Tangent of the texture U axis, orthogonal to [`Self::norm`].
    pub tangent: Vec3,
    _pad_tangent: f32,
    /// Tangent of the texture V axis, orthogonal to the normal and tangent.
    pub bitangent: Vec3,
    _pad_bitangent: f32,
    /// The corners of the face, in the order of its indices.
    pub verts: [TemplateVert; 4],
}

/// What the cpu reads of a block model template.
///
/// The mesher merges and culls the faces of a block by `spans` and `cull`, leaves the
/// occlusion of a `translucent` face open, and reads the side of each axis a corner lies
/// on off `verts`; the interface draws a block from the same corners and normal. The
/// table the passes read holds the same face, written for a shader.
#[derive(Debug)]
pub struct TemplateProperties {
    /// The axis each merge grows along and the direction the uv of the corners it moves
    /// advances in, one slot per merge of the table, `None` where a slot holds none.
    pub spans: [Option<Axis>; 2],
    /// The direction a neighbor culls the face from, or `None` while no neighbor does.
    ///
    /// A translucent block, such as one of leaves, asks for `None`, so that the faces of
    /// the surrounding blocks are kept too.
    pub cull: Option<Direction>,
    /// The normal of the face.
    pub norm: Vec3,
    /// The corners of the face, in the order of its indices.
    pub verts: [TemplateVert; 4],
}

/// The value of a slot of a face that holds no merge: the axis 3 is the one no merge
/// grows along.
const NO_MERGE: u32 = 3;

/// How far the uv of the corners a merge moves advances per block, indexed by the two
/// bits of a merge, in the order they count up from zero.
const UV_STEP: [Vec2; 4] = [
    Vec2::new(1.0, 0.0),
    Vec2::new(-1.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, -1.0),
];

/// A face of a block model template as it is written to a `.ron` file.
///
/// The file spells a face the way a person reads one: the normal as a vector, the
/// merges as the axes they grow along, and the corners without the padding the table
/// needs.
/// What a file leaves out — which corners a merge moves and how far their uv advances
/// — is read off the corners while the file is loaded.
#[derive(Serialize, Deserialize)]
struct RawTemplate {
    norm: Vec3,
    spans: [Option<Axis>; 2],
    cull: Option<Direction>,
    verts: Vec<RawVert>,
}

/// One corner of a face as it is written to a `.ron` file.
#[derive(Serialize, Deserialize)]
struct RawVert {
    pos: Vec3,
    uv: Vec2,
}

/// The bind group of the block model template table.
pub struct BlockTemplates;

impl BindSignature for BlockTemplates {
    const NAME: &'static str = "template";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    type Content<'a> = &'a Buffer;
}

impl TemplateProperties {
    /// The face as the mesh of one face of a block, which is what the mesher bends the
    /// rectangles of a chunk into and what a block of the interface is drawn with.
    pub(crate) fn mesh(&self) -> Mesh<NormUvVertex> {
        let vertices = self
            .verts
            .iter()
            .map(|vert| NormUvVertex {
                pos: vert.pos,
                uv: vert.uv,
                norm: self.norm,
            })
            .collect();

        Mesh {
            vertices,
            indices: Vec::from(QUAD_INDICES),
        }
    }
}

impl RawTemplate {
    /// The two halves of a template: the face the passes read, and what the cpu reads.
    ///
    /// A file holds one face, so both halves are read at once.
    ///
    /// A merge grows along an axis of the block, so it moves the corners on the far side
    /// of it, and their uv advances by as much as the uv of the face changes over one
    /// block along that axis. Both are read off the corners here, once, because a vertex
    /// shader has neither the room nor the reason to work them out per corner.
    ///
    /// Fails on a face the shaders could not read the way the mesher draws it: the corners
    /// have to be a quad in the plane of the normal and wound as its front face, a merge
    /// has to grow along an axis the face covers whole and advance the uv of the corners
    /// it moves by a whole block at a time, and a neighbor can only cull a face that lies
    /// on the border of the block.
    fn halves(self) -> anyhow::Result<(TemplateModel, TemplateProperties)> {
        if self.verts.len() != 4 {
            anyhow::bail!(
                "a face has to be drawn from four corners, but this one has {}",
                self.verts.len()
            );
        }

        let corners = array::from_fn(|i| TemplateVert {
            pos: self.verts[i].pos,
            _pad0: 0.0,
            uv: self.verts[i].uv,
            _pad1: Vec2::ZERO,
        });

        let norm = self.norm;
        // A normal of zero length has no plane, and a normal that is not of unit length
        // would slant the light that the shading of a face weighs it by.
        if (norm.length() - 1.0).abs() > 1e-3 {
            anyhow::bail!("the normal {} of a face has to be a unit vector", norm);
        }

        if let Some(vert) = corners
            .iter()
            .find(|vert| norm.dot(vert.pos - corners[0].pos).abs() > 1e-4)
        {
            anyhow::bail!(
                "the corners of a face have to lie in the plane of its normal, but {} does not",
                vert.pos
            );
        }

        // The winding decides which side of the quad is its front face, and the block
        // passes cull the other one, so a face wound the other way would not be drawn
        // at all.
        let (a, b) = (
            corners[1].pos - corners[0].pos,
            corners[3].pos - corners[0].pos,
        );
        if a.cross(b).dot(norm) <= 0.0 {
            anyhow::bail!("the corners of a face have to be wound the way its normal faces");
        }

        let edge1 = corners[1].pos - corners[0].pos;
        let edge2 = corners[2].pos - corners[0].pos;
        let duv1 = corners[1].uv - corners[0].uv;
        let duv2 = corners[2].uv - corners[0].uv;
        let determinant = duv1.x * duv2.y - duv2.x * duv1.y;

        if determinant.abs() <= 1e-6 {
            anyhow::bail!("the uv coordinates of a face have to span an area");
        }

        let tangent = (edge1 * duv2.y - edge2 * duv1.y) / determinant;
        let bitangent = (edge2 * duv1.x - edge1 * duv2.x) / determinant;
        let tangent = (tangent - norm * norm.dot(tangent)).normalize_or_zero();
        let bitangent = (bitangent - norm * norm.dot(bitangent) - tangent * tangent.dot(bitangent))
            .normalize_or_zero();

        if tangent == Vec3::ZERO || bitangent == Vec3::ZERO {
            anyhow::bail!("the uv mapping of a face has no usable tangent basis");
        }

        // A merge grows along an axis of the plane of the face, so a face holds at most
        // one merge per axis, and none along the normal.
        let mut held: Option<Axis> = None;
        // A slot that holds no merge keeps the axis no merge grows along.
        let mut spans = NO_MERGE | (NO_MERGE << 4);
        let mut merges = [None; 2];

        for (slot, merge) in self.spans.iter().enumerate() {
            let Some(axis) = *merge else {
                continue;
            };

            if held == Some(axis) {
                anyhow::bail!("a face merges along one axis once");
            }

            if norm.get(axis) != 0.0 {
                anyhow::bail!("a face merges along the axes of its plane, not along its normal");
            }

            held = Some(axis);

            // The corners a merge moves are the two on the far side of its axis, and
            // the face has to cover the whole block along that axis for its faces to
            // tile with the faces of the neighboring blocks.
            let far = (0..4)
                .filter(|&i| corners[i].pos.get(axis) == 1.0)
                .collect::<Vec<_>>();
            let near = (0..4)
                .filter(|&i| corners[i].pos.get(axis) == 0.0)
                .collect::<Vec<_>>();

            if far.len() != 2 || near.len() != 2 {
                anyhow::bail!("a face has to cover the whole block along the axis it merges along");
            }

            // A corner a merge moves advances its uv by the step between it and the
            // corner it moves towards, which is the one that shares its place on the
            // other axes of the plane.
            let mut step = None;

            for &f in far.iter() {
                let Some(&n) = near.iter().find(|&&n| {
                    Axis::ALL
                        .iter()
                        .filter(|&&other| other != axis)
                        .all(|&other| corners[n].pos.get(other) == corners[f].pos.get(other))
                }) else {
                    anyhow::bail!("a corner a face merges from has no corner it moves towards");
                };

                let moved = corners[f].uv - corners[n].uv;

                match step {
                    None => step = Some(moved),

                    Some(step) if step != moved => {
                        anyhow::bail!("the corners a merge moves have to advance their uv alike")
                    }

                    Some(_) => (),
                }
            }

            let Some(step) = step else {
                anyhow::bail!("a face holds a merge but no corners to move with it");
            };

            let Some(uv) = UV_STEP.iter().position(|step_unit| *step_unit == step) else {
                anyhow::bail!(
                    "the uv of a face has to advance by one block along the axis it merges along"
                );
            };

            // The slot of a merge replaces the axis it was left at, which is the axis no
            // merge grows along.
            let at = 4 * slot as u32;
            spans =
                (spans & !(0xF << at)) | ((axis.idx() as u32) << at) | ((uv as u32) << (at + 2));
            merges[slot] = Some(axis);
        }

        if let Some(dir) = self.cull {
            if dir.vector::<Vec3>() != norm {
                anyhow::bail!("a neighbor culls a face from the direction the face faces");
            }

            let border = if dir.positive() { 1.0 } else { 0.0 };
            if corners[0].pos.get(dir.axis()) != border {
                anyhow::bail!(
                    "a neighbor can only cull a face that lies on the border of the block"
                );
            }
        }

        Ok((
            TemplateModel {
                norm,
                spans,
                tangent,
                _pad_tangent: 0.0,
                bitangent,
                _pad_bitangent: 0.0,
                verts: corners,
            },
            TemplateProperties {
                spans: merges,
                cull: self.cull,
                norm,
                verts: corners,
            },
        ))
    }
}

impl From<Vec<BlockModelPart>> for BlockModel {
    fn from(value: Vec<BlockModelPart>) -> Self {
        Self::new(value)
    }
}

impl From<BlockModel> for Vec<BlockModelPart> {
    fn from(value: BlockModel) -> Self {
        value.parts
    }
}

impl From<RawBlockModelPart<'_>> for BlockModelPart {
    fn from(value: RawBlockModelPart) -> Self {
        Self {
            template: BLOCK_MODEL_TEMPLATES.id_of(value.template),
            texture: BLOCK_TEXTURE_HANDLES.id_of(value.texture),
            material: BLOCK_MATERIAL_HANDLES.id_of(value.material),
            no_cull: value.no_cull,
        }
    }
}

impl From<BlockModelPart> for RawBlockModelPart<'_> {
    fn from(value: BlockModelPart) -> Self {
        Self {
            template: BLOCK_MODEL_TEMPLATES.name_of(value.template),
            texture: BLOCK_TEXTURE_HANDLES.name_of(value.texture),
            material: BLOCK_MATERIAL_HANDLES.name_of(value.material),
            no_cull: value.no_cull,
        }
    }
}

impl BlockModel {
    pub fn new(parts: Vec<BlockModelPart>) -> Self {
        let mut cull = [None; 6];

        for part in parts.iter() {
            let template = BLOCK_MODEL_TEMPLATES.get(part.template);
            let &translucent = BLOCK_TEXTURE_HANDLES.get(part.texture);
            let &emissive = BLOCK_MATERIAL_HANDLES.get(part.material);
            if !part.no_cull
                && let Some(dir) = template.cull
            {
                cull[dir.idx()] = Some(cull[dir.idx()].unwrap_or(false) || translucent || emissive);
            }
        }

        Self { parts, cull }
    }

    /// The mesh the model is drawn with, in block coordinates, with the texture
    /// of every part.
    ///
    /// The faces are neither culled nor merged, unlike the meshes of a chunk,
    /// because a single block has no neighbors to cull them against.
    pub(crate) fn mesh(&self) -> Mesh<NormTexVertex> {
        self.parts
            .iter()
            .map(|part| {
                BLOCK_MODEL_TEMPLATES
                    .get(part.template)
                    .mesh()
                    .with_texture(part.texture)
            })
            .collect::<Vec<_>>()
            .merge()
    }

    pub fn empty() -> Self {
        Self::default()
    }

    fn culls(&self, dir: Direction, translucent: bool) -> bool {
        self.cull[dir.idx()].is_some_and(|t| !t || translucent)
    }
}

/// One rectangle of the faces of a chunk, as the mesher produces them.
///
/// This is what a merge leaves: the corner of the chunk the rectangle starts at, how far
/// it grows along each merge of its template in the slot that merge holds, and the part
/// whose texture it draws. The region turns these into the instances the passes draw, so
/// a rectangle never leaves the thread that meshed it.
#[derive(Copy, Clone, Debug)]
struct BlockRect {
    pos: RelPos,
    extent: [u8; 2],
    part: BlockModelPart,
}

impl BlockRect {
    /// The rectangle as the instance the passes draw, at the level of detail it was
    /// meshed at: its position is in blocks of the world, and the vertex shader scales
    /// the corners of its template by the same level.
    fn instance(&self, origin: BlockPos, lod: u8) -> BlockRectInst {
        let scale = 1 << lod;

        BlockRectInst {
            pos: origin + self.pos.as_ivec3() * scale,
            template: self.part.template,
            tex: self.part.texture,
            material: self.part.material,
            extent: [self.extent[0] as u32, self.extent[1] as u32],
            lod: lod as u32,
        }
    }
}

/// One merged rectangle of the faces of a chunk, as the vertex shaders of the passes that
/// draw blocks read it.
///
/// A rectangle is drawn out of the corners of its template, so an instance holds what a
/// merge left — where the rectangle stands, how far it grows along each merge in the slot
/// that merge holds, and the level of detail it was meshed at — instead of the vertices
/// those corners take once they are stretched.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockRectInst {
    /// World position of the corner the rectangle starts at, in blocks.
    pub pos: IVec3,
    /// Index of the template the rectangle is cut from.
    pub template: u32,
    /// Layer of the block texture array, or of the occlusion pattern array, the
    /// rectangle samples.
    pub tex: u32,
    /// Layer of the material texture array the rectangle samples.
    pub material: u32,
    /// How far the rectangle grows along each merge of its template, in the slot that
    /// merge holds, in blocks of the level it was meshed at.
    pub extent: [u32; 2],
    /// The level of detail the rectangle was meshed at.
    pub lod: u32,
}

impl Inst for BlockRectInst {
    fn layout<'a, V: Vertex>() -> Option<VertexBufferLayout<'a>> {
        Some(VertexBufferLayout {
            array_stride: size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: offset_of!(Self, pos) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT,
                    format: VertexFormat::Sint32x3,
                },
                VertexAttribute {
                    offset: offset_of!(Self, template) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 1,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: offset_of!(Self, tex) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 2,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: offset_of!(Self, extent) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 3,
                    format: VertexFormat::Uint32x2,
                },
                VertexAttribute {
                    offset: offset_of!(Self, lod) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 4,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: offset_of!(Self, material) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 5,
                    format: VertexFormat::Uint32,
                },
            ],
        })
    }
}

/// A request to mesh a chunk: a single sub-region when `pos` is set, otherwise
/// the whole region.
pub struct MeshingTask {
    pub lod: u8,
    pub pos: Option<SubRegionPos>,
    pub chunk: BlockChunk,
    /// World position of the region the chunk belongs to, so that the rectangles of the
    /// mesh can be turned into instances of the whole world without the region having to
    /// touch them again.
    pub origin: BlockPos,
    pub tx: Sender<MeshingResult>,
    /// The revision of the blocks the task was built from, which lets the
    /// region discard a mesh that arrived after the blocks changed again.
    pub revision: u64,
}

pub struct MeshingResult {
    pub lod: u8,
    pub pos: Option<SubRegionPos>,
    /// The rectangles of the blocks of the chunk, as the passes that draw them read them.
    /// They are in the world, at the level of detail the task asked for.
    pub blocks: Vec<BlockRectInst>,
    /// Visible rectangles of translucent materials.
    pub translucent: Vec<BlockRectInst>,
    /// The ambient occlusion of the faces of the chunk, which a pass of its own
    /// draws over the blocks.
    pub occlusion: Vec<BlockRectInst>,
    /// Every translucent face needed by the shadow transmission pass, including
    /// faces that the camera pass culls.
    pub optical: Vec<BlockRectInst>,
    pub revision: u64,
}

pub struct ChunkMesher {
    task_rx: Receiver<MeshingTask>,
}

/// Merges the equally shaped faces of one template into larger rectangles.
///
/// The blocks are stored as one bitmask per line: `lines` is indexed by two of
/// the three axes, and a set bit marks a block along the merge axis
/// `axes[0]`. `two` tells whether the template may also be merged along a second
/// axis, which is what turns a row into a rectangle.
struct MeshMerger {
    lines: Vec<u64>,
    side: u8,
    axes: [Axis; 3],
    /// The slot each merge axis took, which is where the extent of a rectangle it
    /// produces is written.
    slots: [u8; 2],
    two: bool,
    current: (usize, u8),
}

impl ChunkMesher {
    pub fn new(task_rx: Receiver<MeshingTask>) -> Self {
        Self { task_rx }
    }

    /// Sub-regions of the near levels of detail are meshed by the near pool, so
    /// that a queue of far regions cannot delay the geometry around the player.
    pub fn update(&mut self, threads: &WorldThreads) {
        let WorldThreads {
            near: near_threads,
            far: far_threads,
        } = threads;

        let mut near_tasks = Vec::new();
        let mut far_tasks = Vec::new();
        for task in self.task_rx.try_iter() {
            match task.pos {
                None => far_tasks.push(task),
                Some(_) => near_tasks.push(task),
            }
        }

        near_threads.spawn(move || {
            near_tasks.into_par_iter().for_each(|task| {
                Self::perform(task);
            })
        });

        far_threads.spawn(move || {
            far_tasks.into_par_iter().for_each(|task| {
                Self::perform(task);
            })
        });
    }

    fn perform(task: MeshingTask) {
        let (blocks, translucent, occlusion, optical) = Self::build_meshes(&task.chunk);

        // The rectangles of a sub-region are in the coordinates of the chunk it was meshed
        // from, so the ones of the region as a whole start where the sub-region stands in
        // it; the shader scales the rest by the level of detail.
        let origin = task.origin + task.pos.unwrap_or_default().as_ivec3() * SUBREGION_SIZE as i32;
        let instances = |rects: &[BlockRect]| -> Vec<BlockRectInst> {
            rects
                .iter()
                .map(|rect| rect.instance(origin, task.lod))
                .collect()
        };

        let result = MeshingResult {
            lod: task.lod,
            pos: task.pos,
            blocks: instances(&blocks),
            translucent: instances(&translucent),
            occlusion: instances(&occlusion),
            optical: instances(&optical),
            revision: task.revision,
        };

        if let Err(e) = task.tx.try_send(result) {
            error!("failed to send meshes to region: {}", e);
        }
    }

    /// The occlusion of the corners of the face of `template` at `real_pos`, as
    /// the number of the blocks that shade each one, in the order the corners
    /// appear in the vertices of the template.
    ///
    /// Only the opaque blocks that touch a corner shade it: the two beside the
    /// face of a culled template, which lie in the plane of that face, and the
    /// three around the vertex of an open one. Two blocks beside a corner shade
    /// it fully, so the block diagonal to it is only read while they do not, and
    /// no corner is ever shaded more than fully.
    fn occlusion(chunk: &BlockChunk, real_pos: U8Vec3, template: &TemplateProperties) -> [u8; 4] {
        let mut occluded = [0u8; 4];

        for (i, vert) in template.verts.iter().enumerate() {
            let step = |axis: Axis| axis.direction(vert.pos.get(axis) > 0.5);
            let opaque = |pos: U8Vec3, dir: Direction| {
                let adjacent = chunk.get_block(pos.step(dir));
                (adjacent.model().cull[dir.opposite().idx()] == Some(false)) as u8
            };

            let shaded = match template.cull {
                Some(dir) => {
                    // The face lies between the block and its neighbor, so the
                    // blocks beside a corner of it are the neighbors of that
                    // block along the axes of the face, and the block diagonal
                    // to the corner sits one step along both of them.
                    let adj_pos = real_pos.step(dir);
                    let axis = dir.axis();
                    let others = axis.others();
                    let (du, dv) = (step(others[0]), step(others[1]));
                    let sides = opaque(adj_pos, du) + opaque(adj_pos, dv);

                    match sides {
                        2 => 2,
                        _ => {
                            sides
                                + (opaque(adj_pos.step(du), dv) + opaque(adj_pos.step(dv), du)) / 2
                        }
                    }
                }

                None => Axis::ALL
                    .iter()
                    .map(|&axis| opaque(real_pos, step(axis)))
                    .sum(),
            };

            occluded[i] = shaded.min(2);
        }

        occluded
    }

    /// The rectangles of a chunk: the faces of its blocks, merged along the axes their
    /// templates merge along, and the ambient occlusion of those faces, which a pass of
    /// its own draws over them.
    fn build_meshes(
        chunk: &BlockChunk,
    ) -> (
        Vec<BlockRect>,
        Vec<BlockRect>,
        Vec<BlockRect>,
        Vec<BlockRect>,
    ) {
        // Merging needs to see every block of a template, so the positions are
        // collected first and merged afterwards.
        let mut opaque_mergers = HashMap::new();
        let mut translucent_mergers = HashMap::new();
        let mut occlusion_mergers = HashMap::new();
        let mut optical_mergers = HashMap::new();
        let [w, h, d] = (chunk.size - 2).to_array();

        let mut opaque = Vec::new();
        let mut translucent = Vec::new();
        let mut occlusion = Vec::new();
        let mut optical = Vec::new();

        for x in 0..w {
            for y in 0..h {
                for z in 0..d {
                    let pos = RelPos::new(x, y, z);
                    let real_pos: LocalPos = pos + 1;
                    let block = Block::from_meta(*chunk.get(real_pos));

                    for part in block.model().parts.iter() {
                        let template = BLOCK_MODEL_TEMPLATES.get(part.template);
                        let &is_translucent = BLOCK_TEXTURE_HANDLES.get(part.texture);
                        let &is_emissive = BLOCK_MATERIAL_HANDLES.get(part.material);

                        let mut culled = !part.no_cull;
                        let mut optic = is_translucent;
                        if let Some(dir) = template.cull {
                            let adjacent = Block::from_meta(*chunk.get(real_pos.step(dir)));
                            culled &= adjacent.model().culls(dir.opposite(), is_translucent);
                            optic &= !(culled && adjacent == block);
                        }

                        // Optical faces include camera-culled translucent surfaces because
                        // they can still tint light along a different view direction.
                        if optic {
                            if template.spans == [None, None] {
                                optical.push(BlockRect {
                                    pos,
                                    extent: [0, 0],
                                    part: *part,
                                });
                            } else {
                                optical_mergers
                                    .entry(*part)
                                    .or_insert_with(|| MeshMerger::new(w, template.spans))
                                    .add(pos);
                            }
                        }

                        if culled {
                            continue;
                        }

                        let (rects, mergers) = if is_translucent {
                            (&mut translucent, &mut translucent_mergers)
                        } else {
                            (&mut opaque, &mut opaque_mergers)
                        };

                        if template.spans == [None, None] {
                            rects.push(BlockRect {
                                pos,
                                extent: [0, 0],
                                part: *part,
                            });
                        } else {
                            mergers
                                .entry(*part)
                                .or_insert_with(|| MeshMerger::new(w, template.spans))
                                .add(pos);
                        }

                        // AO is only for opaque, non-emissive materials.
                        if is_translucent || is_emissive {
                            continue;
                        }

                        let occluded = Self::occlusion(chunk, real_pos, template);
                        let code = occluded
                            .iter()
                            .enumerate()
                            .map(|(i, &level)| level as u16 * 3u16.pow(i as u32))
                            .sum::<u16>();

                        if code == 0 {
                            continue;
                        }

                        let part = BlockModelPart {
                            template: part.template,
                            texture: code as Id,
                            material: 0,
                            no_cull: true,
                        };

                        if template.spans == [None, None] {
                            occlusion.push(BlockRect {
                                pos,
                                extent: [0, 0],
                                part,
                            });
                        } else {
                            occlusion_mergers
                                .entry(part)
                                .or_insert_with(|| MeshMerger::new(w, template.spans))
                                .add(pos);
                        }
                    }
                }
            }
        }

        opaque.extend(
            opaque_mergers
                .into_par_iter()
                .flat_map_iter(|(part, merges)| {
                    merges.map(move |(pos, extent)| BlockRect { pos, extent, part })
                })
                .collect::<Vec<_>>(),
        );
        translucent.extend(
            translucent_mergers
                .into_par_iter()
                .flat_map_iter(|(part, merges)| {
                    merges.map(move |(pos, extent)| BlockRect { pos, extent, part })
                })
                .collect::<Vec<_>>(),
        );
        occlusion.extend(
            occlusion_mergers
                .into_par_iter()
                .flat_map_iter(|(part, merges)| {
                    merges.map(move |(pos, extent)| BlockRect { pos, extent, part })
                })
                .collect::<Vec<_>>(),
        );
        optical.extend(
            optical_mergers
                .into_par_iter()
                .flat_map_iter(|(part, merges)| {
                    merges.map(move |(pos, extent)| BlockRect { pos, extent, part })
                })
                .collect::<Vec<_>>(),
        );

        (opaque, translucent, occlusion, optical)
    }
}

impl Iterator for MeshMerger {
    type Item = (U8Vec3, [u8; 2]);

    /// Takes the run of set bits at the current position, then grows the
    /// rectangle over the following lines for as long as they have a run of the
    /// same length. The item is the corner of the rectangle and how far it
    /// extends along each merge, in the slot the merge holds in the face.
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.side as usize;

        let (idx, bit) = &mut self.current;

        while *idx < n * n {
            *bit += (self.lines[*idx] >> (*bit)).trailing_zeros() as u8;
            if *bit >= self.side {
                *bit = 0;
                *idx += 1;
            } else {
                break;
            }
        }

        let (idx, bit) = self.current;

        if idx < n * n {
            let pos = self.pos_of_bit(self.current);
            let mut extent = [0; 2];

            // The run of set bits is the length of the rectangle along the first
            // merge axis; `trailing_ones` counts it, `dist` is the distance the
            // far vertices have to be moved.
            let dist = (self.lines[idx] >> bit).trailing_ones() as u8 - 1;
            let removal = !(((1u64 << (dist + 1)) - 1) << bit);
            extent[self.slots[0] as usize] = dist;
            self.lines[idx] &= removal;

            if self.two {
                let mut dist1 = 0;

                // Only the lines of the same row can extend the rectangle, and
                // they have to cover the run that was just consumed.
                for idx1 in idx + 1..((idx / n) + 1) * n {
                    if (self.lines[idx1] >> bit).trailing_ones() as u8 > dist {
                        dist1 += 1;
                        self.lines[idx1] &= removal;
                    } else {
                        break;
                    }
                }

                extent[self.slots[1] as usize] = dist1;
            }

            Some((pos, extent))
        } else {
            None
        }
    }
}

impl MeshMerger {
    fn new(side: u8, merges: [Option<Axis>; 2]) -> Self {
        // The merge axes take the first slots of `axes`, so that the bit of a position
        // is its coordinate along the merge axis, and each keeps the slot it holds in
        // the face, which is where the extent of the rectangle it produces is written.
        let masks = vec![0; (side as usize).pow(2)];
        let mut axes = *Axis::ALL;
        let mut slots = [0; 2];
        let mut count = 0;

        for (slot, merge) in merges.iter().enumerate() {
            let Some(merge) = merge else {
                continue;
            };

            // Move the axis of the merge in front of the axes that are not merged,
            // keeping those in order.
            let at = axes
                .iter()
                .position(|axis| axis == merge)
                .expect("the axes of a block are the three axes");
            axes[count..=at].rotate_right(1);

            slots[count] = slot as u8;
            count += 1;
        }

        Self {
            lines: masks,
            side,
            axes,
            slots,
            two: count > 1,
            current: (0, 0),
        }
    }

    #[inline]
    fn pos_of_bit(&self, bit: (usize, u8)) -> U8Vec3 {
        let (idx, bit) = bit;
        let n = self.side as usize;

        U8Vec3::ZERO
            .with(self.axes[0], bit)
            .with(self.axes[1], (idx % n) as u8)
            .with(self.axes[2], (idx / n) as u8)
    }

    #[inline]
    fn bit_of_pos(&self, pos: U8Vec3) -> (usize, u8) {
        debug_assert!(pos.x < self.side && pos.y < self.side && pos.z < self.side);

        let n = self.side as usize;
        let idx = pos.get(self.axes[1]) as usize + pos.get(self.axes[2]) as usize * n;
        (idx, pos.get(self.axes[0]))
    }

    fn add(&mut self, pos: U8Vec3) {
        let (idx, bit) = self.bit_of_pos(pos);
        self.lines[idx] |= 1u64 << bit;
    }
}
