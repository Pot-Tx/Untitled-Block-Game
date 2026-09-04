struct VertexIn {
    @location(0) pos: vec3f,
    @location(2) width: f32,
    @location(3) height: f32,
    @location(4) min_uv: vec2f,
    @location(5) max_uv: vec2f,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
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
    var pos = in.pos;
    var uv = in.min_uv;
    switch (id) {
        case 0u: {
            pos.y += in.height;
        }
        case 1u: {
            uv.y = in.max_uv.y;
        }
        case 2u: {
            pos.x += in.width;
            uv = in.max_uv;
        }
        case 3u: {
            pos.x += in.width;
            pos.y += in.height;
            uv.x = in.max_uv.x;
        }
        default: {}
    };
    let transformed = viewport * vec4f(pos, 1.0);

    return VertexOut(transformed, uv);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    /*var uv = in.pos.xy * 0.5 + vec2f(0.5, 0.5);
    uv.y = 1.0 - uv.y;
    let color = textureSample(surface, sursampler, uv);
    let alpha = textureSample(texture, texsampler, in.uv).a;
    let inverted = vec4((color.rgb + vec3f(alpha)) - 2.0 * alpha * color.rgb, color.a);*/

    return FragmentOut(textureSample(texture, texsampler, in.uv), in.pos.z);
}
