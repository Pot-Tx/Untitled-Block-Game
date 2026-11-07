struct VertexIn {
    // location 0 is the ui atlas layer, unused by text
    @location(1) min: vec2f,
    @location(2) max: vec2f,
    @location(3) z: f32,
    @location(4) min_uv: vec2f,
    @location(5) max_uv: vec2f,
    @location(6) color: vec4f,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
}

struct FragmentOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0)
var<uniform> viewport: mat4x4<f32>;
@group(1) @binding(0)
var texture: texture_2d<f32>;
@group(1) @binding(1)
var texsampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) id: u32, in: VertexIn) -> VertexOut {
    var pos = in.min;
    var uv = in.min_uv;
    switch (id) {
            case 1u: {
                pos.y = in.max.y;
                uv.y = in.max_uv.y;
            }
            case 2u: {
                pos = in.max;
                uv = in.max_uv;
            }
            case 3u: {
                pos.x = in.max.x;
                uv.x = in.max_uv.x;
            }
            default: {}
        };
    let transformed = viewport * vec4f(pos, in.z, 1.0);

    return VertexOut(transformed, uv, in.color);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    let a = select(0.0, 1.0, textureSample(texture, texsampler, in.uv).r > 0.5);
    return FragmentOut(vec4f(in.color.rgb, in.color.a * a), in.pos.z);
}
