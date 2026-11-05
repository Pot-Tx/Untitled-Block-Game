// Draws the blocks of the world into one cascade of the shadow map.
//
// The pass writes depth alone. Its fragment stage only rejects texels below the
// alpha cutoff, so rejected parts of a face do not write depth.

// The corners of a face of a block model, as the table of the templates holds them.
//#include "block_template.wgsl"

// The rectangle of a face the pass draws: where it stands in the world, the template its
// corners are cut from, how far it grows along each merge of that template, and the level
// of detail it was meshed at.
struct VertexIn {
    @location(0) inst_pos: vec3i,
    @location(1) inst_template: u32,
    @location(2) inst_tex: u32,
    @location(3) inst_extent: vec2<u32>,
    @location(4) inst_lod: u32,
    @location(5) inst_material: u32,
    @builtin(vertex_index) corner: u32,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) tex: u32,
}

@group(0) @binding(0)
var colors: texture_2d_array<f32>;
@group(0) @binding(1)
var texsampler: sampler;

@group(1) @binding(0)
var<uniform> cascade: mat4x4<f32>;

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

    let world = vec3f(in.inst_pos) + local * scale;
    return VertexOut(cascade * vec4f(world, 1.0), uv * scale, in.inst_tex);
}

@fragment
fn fs_main(in: VertexOut) {
    let color = textureSample(colors, texsampler, in.uv, in.tex);
    if (color.a < 0.5) {
        discard;
    }
}
