// Reduces one Hi-Z level into the next one.

@group(0) @binding(0)
var hi_z_src: texture_2d<f32>;

@group(1) @binding(0)
var hi_z_dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let src_size = textureDimensions(hi_z_src, 0);
    let dst_size = textureDimensions(hi_z_dst);

    if (gid.x >= dst_size.x || gid.y >= dst_size.y) {
        return;
    }

    let max_pos = src_size - vec2u(1u);
    let p = gid.xy * 2u;
    let a = textureLoad(hi_z_src, min(p, max_pos), 0).r;
    let b = textureLoad(hi_z_src, min(p + vec2u(1u, 0u), max_pos), 0).r;
    let c = textureLoad(hi_z_src, min(p + vec2u(0u, 1u), max_pos), 0).r;
    let d = textureLoad(hi_z_src, min(p + vec2u(1u, 1u), max_pos), 0).r;

    textureStore(
        hi_z_dst,
        vec2i(i32(gid.x), i32(gid.y)),
        vec4f(min(min(a, b), min(c, d)), 0.0, 0.0, 0.0),
    );
}
