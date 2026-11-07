// Builds the first level of the conservative depth pyramid.
//
// The scene depth is reverse depth: a larger value is closer to the camera.
// Every output texel therefore stores the smallest depth of its source
// footprint.

@group(0) @binding(0)
var scene_depth: texture_depth_2d;

@group(1) @binding(0)
var hi_z_dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let src_size = textureDimensions(scene_depth, 0);
    let dst_size = textureDimensions(hi_z_dst);

    if (gid.x >= dst_size.x || gid.y >= dst_size.y) {
        return;
    }

    let sx0 = gid.x * src_size.x / dst_size.x;
    let sx1 = max(sx0 + 1u, (gid.x + 1u) * src_size.x / dst_size.x);
    let sy0 = gid.y * src_size.y / dst_size.y;
    let sy1 = max(sy0 + 1u, (gid.y + 1u) * src_size.y / dst_size.y);

    var depth = 1.0;
    for (var y = sy0; y < sy1; y += 1u) {
        for (var x = sx0; x < sx1; x += 1u) {
            depth = min(
                depth,
                textureLoad(scene_depth, vec2i(i32(x), i32(y)), 0),
            );
        }
    }

    textureStore(
        hi_z_dst,
        vec2i(i32(gid.x), i32(gid.y)),
        vec4f(depth, 0.0, 0.0, 0.0),
    );
}
