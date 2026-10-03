struct VertexIn {
    // The block mesh.
    @location(0) pos: vec3f,
    @location(1) tex: u32,
    @location(2) uv: vec2f,
    @location(3) norm: vec3f,
    // The instance: the rectangle of the element and the range of depths the
    // block itself is drawn in.
    @location(4) min: vec2f,
    @location(5) max: vec2f,
    @location(6) min_z: f32,
    @location(7) max_z: f32,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) tex: u32,
    @location(1) uv: vec2f,
    @location(2) illum: f32,
}

struct FragmentOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0)
var textures: texture_2d_array<f32>;
@group(0) @binding(1)
var texsampler: sampler;
@group(1) @binding(0)
var<uniform> block_camera: mat4x4<f32>;
@group(2) @binding(0)
var<uniform> viewport: mat4x4<f32>;

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    // The camera projects the block into -1..1 on both axes and keeps the depth
    // of its faces, which is what makes the front faces hide the back ones.
    let view = block_camera * vec4f(in.pos, 1.0);

    // A block is as wide as it is tall, so both axes are scaled by the same
    // factor, which keeps it square and centered in the rectangle.
    let center = (in.min + in.max) * 0.5;
    let scale = min(in.max.x - in.min.x, in.max.y - in.min.y) * 0.5;
    let ui = center + view.xy * scale;

    // The faces of the block are mapped between the depths of the instance, so
    // that the block keeps its own order without leaving the layer of its
    // element.
    let depth = mix(in.min_z, in.max_z, view.z);
    let pos = viewport * vec4f(ui, depth, 1.0);

    let p = max(in.norm, vec3f(0.0));
    let n = max(-in.norm, vec3f(0.0));
    let illum = dot(p, vec3f(0.875, 1.0, 0.75)) + dot(n, vec3f(0.625, 0.375, 0.5));

    return VertexOut(pos, in.tex, in.uv, illum);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var color = textureSample(textures, texsampler, in.uv, i32(in.tex));
    color.r *= in.illum;
    color.g *= in.illum;
    color.b *= in.illum;

    return FragmentOut(color, in.pos.z);
}
