// Draws the blocks of the world with the light that reaches them.
//
// A draw call is a list of rectangles: the vertex stage cuts the corners of each one
// out of the template it names and stretches them along the merges of that template, which is
// what keeps the geometry that a change to a block uploads small.

//#include "block_template.wgsl"

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
    @location(0) @interpolate(flat) tex: u32,
    @location(1) @interpolate(flat) material: u32,
    @location(2) uv: vec2f,
    @location(3) world: vec3f,
    @location(4) @interpolate(flat) norm: vec3f,
    @location(5) @interpolate(flat) tangent: vec3f,
    @location(6) @interpolate(flat) bitangent: vec3f,
    @location(7) @interpolate(flat) scale: f32,
}

struct FragmentOut {
    @location(0) color: vec4f,
}

@group(0) @binding(0)
var colors: texture_2d_array<f32>;
@group(0) @binding(1)
var materials: texture_2d_array<f32>;
@group(0) @binding(2)
var texsampler: sampler;

//#include "lighting.wgsl"

// The templates the rectangles of a draw call are cut from, one entry per template id.
@group(3) @binding(0)
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

    for (var slot = 0u; slot < 2u; slot += 1u) {
        let merge = (face.spans >> (4u * slot)) & 15u;
        let axis = min(merge & 3u, 2u);
        let moved = select(0.0, f32(extent[slot]), dot(vert.pos, axis_vector(axis)) > 0.5);

        local += axis_vector(axis) * moved;
        uv += uv_step(merge >> 2u) * moved;
    }

    let world = vec3f(in.inst_pos) + local * scale;

    return VertexOut(
        camera.matrix * vec4f(world, 1.0),
        in.inst_tex,
        in.inst_material,
        uv * scale,
        world,
        face.norm,
        face.tangent,
        face.bitangent,
        scale,
    );
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    let color = textureSample(colors, texsampler, in.uv, in.tex);
    if color.a < 0.5 { discard; }
    let material = textureSampleLevel(materials, texsampler, in.uv, in.material, 0.0);

    let tangent = normalize(in.tangent - in.norm * dot(in.norm, in.tangent));
    let bitangent = normalize(
        in.bitangent
        - in.norm * dot(in.norm, in.bitangent)
        - tangent * dot(tangent, in.bitangent)
    );
    let sampled_normal = material.rgb * 2.0 - 1.0;
    let shading_normal = normalize(
        tangent * sampled_normal.x
        + bitangent * sampled_normal.y
        + in.norm * sampled_normal.z
    );

    let texture_size = vec2f(textureDimensions(colors, 0));
    let shadow_world = snap_world(
        in.world,
        in.uv,
        tangent,
        bitangent,
        texture_size,
        in.scale,
    );
    let light = lighting(shading_normal, in.norm, shadow_world, in.pos.xy);
    let sky = ambient_radiance(shading_normal);
    let emitted = color.rgb * material.a;

    return FragmentOut(vec4f(color.rgb * (sky + light) + emitted, 1.0));
}
