struct VertexIn {
    @location(0) pos: vec3f,
    @location(1) tex: u32,
    @location(2) uv: vec2f,
    @location(3) inst: vec3f,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) tex: u32,
    @location(1) uv: vec2f,
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
    var pos = camera * vec4f(in.pos + in.inst, 1.0);

    return VertexOut(pos, in.tex, in.uv);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var color = textureSample(textures, texsampler, in.uv, in.tex);
    
    return FragmentOut(color, in.pos.z);
}
