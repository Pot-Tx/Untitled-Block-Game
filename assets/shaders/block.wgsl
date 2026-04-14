struct VertexIn {
    @location(0) pos: vec3f,
    @location(1) tex: u32,
    @location(2) uv: vec2f,
    @location(3) norm: vec3f,
    @location(4) inst: vec3i,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) tex: u32,
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
var<uniform> camera: mat4x4<f32>;

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let pos = camera * vec4f(in.pos + vec3f(in.inst), 1.0);

    let p = max(in.norm, vec3f(0.0));
    let n = max(-in.norm, vec3f(0.0));
    let illum = dot(p, vec3f(0.875, 1.0, 0.75)) + dot(n, vec3f(0.625, 0.375, 0.5));

    return VertexOut(pos, in.tex, in.uv, illum);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var color = textureSample(textures, texsampler, in.uv, in.tex);
    color.x *= in.illum;
    color.y *= in.illum;
    color.z *= in.illum;
    
    return FragmentOut(color, in.pos.z);
}
