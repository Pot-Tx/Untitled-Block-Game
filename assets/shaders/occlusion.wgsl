struct VertexIn {
    @location(0) pos: vec3f,
    @location(1) alpha: f32,
    @location(2) inst: vec3i,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) alpha: f32,
}

struct FragmentOut {
    @location(0) color: vec4f,
}

@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var pos = camera * vec4f(in.pos + vec3f(in.inst), 1.0);
    pos.z *= 1.0009765625;

    return VertexOut(pos, in.alpha);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var color = vec4f(0.0, 0.0, 0.0, in.alpha);

    return FragmentOut(color);
}
