// Shades the corners of the blocks that the light of the sky does not reach.
//
// The faces of the blocks are drawn a second time, over the pass that draws
// them, and every face samples the pattern of its corners: a pattern holds the
// occlusion of the four corners of a face, the share of it that a corner takes
// is the alpha of the texel the corner lies on, and the pass draws that share as
// a black cover over the face, which the blending mixes into the block below.
//
// The faces are the rectangles a draw call holds, and the vertex stage cuts their corners
// out of the template each rectangle names, exactly as the pass that drew the block did.

// The corners of a face of a block model, as the table of the templates holds them.
//#include "block_template.wgsl"

// The rectangle of a face the pass draws: where it stands in the world, the template its
// corners are cut from, the layer of the pattern array it samples, how far it grows along
// each merge of that template, and the level of detail it was meshed at.
struct VertexIn {
    @location(0) inst_pos: vec3i,
    @location(1) inst_template: u32,
    @location(2) inst_tex: u32,
    @location(3) inst_extent: vec2<u32>,
    @location(4) inst_lod: u32,
    @builtin(vertex_index) corner: u32,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) tex: u32,
    @location(1) uv: vec2f,
}

struct FragmentOut {
    @location(0) color: vec4f,
}

// The pattern of a face, whose layer is the occlusion of its four corners
// counted up in base three.
@group(0) @binding(0)
var patterns: texture_2d_array<f32>;
@group(0) @binding(1)
var patternsampler: sampler;

// The transform of the camera and the position it stands at.
struct Camera {
    matrix: mat4x4<f32>,
    position: vec4f,
}

@group(1) @binding(0)
var<uniform> camera: Camera;

// The templates the rectangles of a draw call are cut from, one entry per template id.
@group(2) @binding(0)
var<storage, read> templates: array<TemplateModel>;

/// The unit vector of the axis of the block at `axis`.
fn axis_vector(axis: u32) -> vec3f {
    if (axis == 0u) {
        return vec3f(1.0, 0.0, 0.0);
    }

    if (axis == 1u) {
        return vec3f(0.0, 1.0, 0.0);
    }

    return vec3f(0.0, 0.0, 1.0);
}

/// How far the uv of the corners a merge moves advances per block, in the order the
/// directions are counted up from zero: `+u`, `-u`, `+v` and `-v`.
fn uv_step(step: u32) -> vec2f {
    if (step == 0u) {
        return vec2f(1.0, 0.0);
    }

    if (step == 1u) {
        return vec2f(-1.0, 0.0);
    }

    if (step == 2u) {
        return vec2f(0.0, 1.0);
    }

    return vec2f(0.0, -1.0);
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let face = templates[in.inst_template];
    let vert = face.verts[in.corner];
    let scale = f32(1u << in.inst_lod);
    var extent = in.inst_extent;
    var local = vert.pos;
    var uv = vert.uv;

    // A merge moves the corners on the far side of its axis, and the slot it holds in the
    // face is the slot of its extent in the rectangle.
    for (var slot = 0u; slot < 2u; slot += 1u) {
        let merge = (face.spans >> (4u * slot)) & 15u;
        // An axis of 3 is a slot that holds no merge; its extent is zero, so the clamp
        // only keeps the vector of the axis defined.
        let axis = min(merge & 3u, 2u);
        let moved = select(0.0, f32(extent[slot]), dot(vert.pos, axis_vector(axis)) > 0.5);

        local += axis_vector(axis) * moved;
        uv += uv_step(merge >> 2u) * moved;
    }

    var pos = camera.matrix * vec4f(vec3f(in.inst_pos) + local * scale, 1.0);
    // The face is drawn where the pass that draws the block left the depth of
    // it, so it is moved towards the camera by a small share of its own depth to
    // win the comparison; a larger depth is the nearer one under inverted depth.
    pos.z *= 1.0009765625;

    return VertexOut(pos, in.inst_tex, uv);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    // A merged face advances its uv by one block at a time, and the sampler
    // repeats the pattern of a block, so the uv reads the same texel of the
    // pattern as the texture of the block reads of itself.
    let occluded = textureSample(patterns, patternsampler, in.uv, in.tex).a;

    return FragmentOut(vec4f(0.0, 0.0, 0.0, occluded));
}
