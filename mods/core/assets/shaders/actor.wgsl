struct VertexIn {
    // The instance: where the actor stands, the size the sprite is drawn at,
    // its local offset, the part of the frame it shows and the frame itself.
    @location(0) pos: vec3f,
    @location(1) size: vec2f,
    @location(2) offset: vec3f,
    @location(3) uv_min: vec2f,
    @location(4) uv_max: vec2f,
    @location(5) tex: u32,
    @location(6) color: vec4f,
}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) @interpolate(flat) tex: u32,
    @location(3) @interpolate(flat) normal: vec3f,
    @location(4) @interpolate(flat) light_world: vec3f,
}

struct FragmentOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0)
var frames: texture_2d_array<f32>;
@group(0) @binding(1)
var framesampler: sampler;

//#include "lighting.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) id: u32, in: VertexIn) -> VertexOut {
    // The corners of the sprite come from the index of the vertex, like the
    // quads of the user interface.
    var corner = vec2f(-0.5, 0.0);
    var uv = vec2f(in.uv_min.x, in.uv_max.y);
    switch (id) {
        case 1u: {
            corner = vec2f(0.5, 0.0);
            uv = in.uv_max;
        }
        case 2u: {
            corner = vec2f(0.5, 1.0);
            uv = vec2f(in.uv_max.x, in.uv_min.y);
        }
        case 3u: {
            corner = vec2f(-0.5, 1.0);
            uv = in.uv_min;
        }
        default: {}
    };

    // The sprite turns around the vertical axis, so that it always stands
    // upright and shows the side the camera is on.
    let to_camera = camera.position.xz - in.pos.xz;
    let flat = select(vec2f(0.0, 1.0), normalize(to_camera), length(to_camera) > 0.0001);
    let right = vec3f(flat.y, 0.0, -flat.x);
    let forward = vec3f(flat.x, 0.0, flat.y);
    let origin = in.pos
        + right * in.offset.x
        + vec3f(0.0, in.offset.y, 0.0)
        + forward * in.offset.z;
    let world = origin
        + right * (corner.x * in.size.x)
        + vec3f(0.0, corner.y * in.size.y, 0.0);

    // A billboard has one visible side, so its normal faces the camera. This is
    // an approximation of a volumetric body, not a second lighting model.
    let normal = normalize(select(
        vec3f(0.0, 0.0, 1.0),
        camera.position.xyz - in.pos,
        length(camera.position.xyz - in.pos) > 0.0001,
    ));
    let light_world = origin + vec3f(0.0, in.size.y * 0.5, 0.0);

    return VertexOut(
        camera.matrix * vec4f(world, 1.0),
        uv,
        in.color,
        in.tex,
        normal,
        light_world,
    );
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    let albedo = textureSample(frames, framesampler, in.uv, i32(in.tex)) * in.color;

    // The transparent part of a sprite must not write depth, so that what is
    // behind it stays visible.
    if (albedo.a < 0.5) {
        discard;
    }

    let light = lighting(in.normal, in.normal, in.light_world, in.pos.xy);
    let sky = ambient_radiance(in.normal);
    return FragmentOut(vec4f(albedo.rgb * (sky + light), albedo.a), in.pos.z);
}
