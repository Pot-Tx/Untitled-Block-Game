struct VertexIn {
    @location(0) pos: vec3f,
    @location(1) inst: vec3f,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
}

struct FragmentOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var pos = camera * vec4f(in.pos + in.inst, 1.0);
    pos.z *= 1.0009765625;
    return VertexOut(pos);
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    return FragmentOut(vec4f(0.0, 0.0, 0.0, 1.0), in.pos.z);
}
