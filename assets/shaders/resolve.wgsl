// Maps the scene buffer onto the surface of the window.
//
// The world is drawn into a floating point buffer, so that its lighting can
// exceed the range of the eight bit surface; this is the pass where that
// headroom is given up. The surface is an sRGB image, so the value written here
// is linear and the hardware encodes it while the fragment is stored.

@group(0) @binding(0)
var scene: texture_2d<f32>;
@group(0) @binding(1)
var scenesampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOut {
    // The corners of a triangle that covers the whole viewport: the position
    // and the uv of a vertex both come from its index, like the quads of the
    // interface. The order of the corners keeps the triangle front facing.
    let uv = vec2f(f32(id & 2u), f32((id << 1u) & 2u));
    let pos = uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0);

    return VertexOut(vec4f(pos, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    return vec4f(textureSample(scene, scenesampler, in.uv).rgb, 1.0);
}
