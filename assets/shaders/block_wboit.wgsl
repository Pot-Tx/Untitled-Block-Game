// Draws camera-visible translucent block faces into WBOIT buffers.

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
    @location(0) accum: vec4f,
    @location(1) reveal: f32,
}

@group(0) @binding(0)
var colors: texture_2d_array<f32>;
@group(0) @binding(1)
var materials: texture_2d_array<f32>;
@group(0) @binding(2)
var texsampler: sampler;

//#include "lighting.wgsl"

@group(3) @binding(0)
var<storage, read> templates: array<TemplateModel>;

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
    if (color.a <= 0.0) {
        discard;
    }

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
    let sample_world = snap_world(
        in.world,
        in.uv,
        tangent,
        bitangent,
        texture_size,
        in.scale,
    );
    let lit = color.rgb
        * (ambient_radiance(shading_normal)
            + lighting(shading_normal, in.norm, sample_world, in.pos.xy))
        + color.rgb * material.a;

    let z = clamp(in.pos.z, 0.0, 1.0);
    let weight = clamp(0.03 / (0.00001 + 0.01 * z * z), 0.01, 3000.0);
    return FragmentOut(
        vec4f(lit * color.a * weight, color.a * weight),
        color.a,
    );
}
