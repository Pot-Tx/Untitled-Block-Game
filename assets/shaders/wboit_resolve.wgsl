@group(0) @binding(0)
var accum: texture_2d<f32>;
@group(0) @binding(1)
var revealage: texture_2d<f32>;
@group(0) @binding(2)
var wboit_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let uv = vec2f(f32(index & 2u), f32((index << 1u) & 2u));
    let pos = uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0);
    return VertexOut(vec4f(pos, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    let sum = textureSample(accum, wboit_sampler, in.uv);
    let reveal = textureSample(revealage, wboit_sampler, in.uv).r;
    let color = sum.rgb / max(sum.a, 0.00001);
    return vec4f(color, 1.0 - reveal);
}
