// Bins every local light into screen and distance clusters.
//
// A light writes its index into every cluster its influence sphere can reach.
// Fragments read one cluster and only evaluate the sources it contains; no CPU
// pass computes which surfaces the light affects.

struct Camera {
    matrix: mat4x4<f32>,
    position: vec4f,
}

struct Light {
    origin_range: vec4f,
    direction_outer: vec4f,
    color_inner: vec4f,
    intensity: f32,
    shadow_first: u32,
    shadow_count: u32,
    flags: u32,
}

struct LightParams {
    light_count: u32,
    unshadowed_directional_count: u32,
    directional_count: u32,
    local_offset: u32,
    local_count: u32,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    max_per_cluster: u32,
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    min_distance: f32,
    max_distance: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(0) @binding(1)
var<storage, read> lights: array<Light>;
@group(0) @binding(2)
var<uniform> params: LightParams;
@group(0) @binding(3)
var<storage, read_write> cluster_counts: array<atomic<u32>>;
@group(0) @binding(4)
var<storage, read_write> cluster_lights: array<u32>;

fn slice_of(distance: f32) -> u32 {
    let span = max(params.max_distance - params.min_distance, 0.0001);
    let t = saturate((distance - params.min_distance) / span);
    return min(u32(floor(t * f32(params.grid_z))), params.grid_z - 1u);
}

fn sphere_intersects_plane(plane: vec4f, center: vec3f, radius: f32) -> bool {
    let normal = plane.xyz;
    return dot(normal, center) + plane.w >= -radius * length(normal);
}

/// Whether a light influence sphere overlaps the camera frustum at all.
///
/// The center of a light may be behind the camera while its sphere still lights
/// a visible surface, so testing the center against the clip volume is not
/// enough.
fn sphere_intersects_camera(center: vec3f, radius: f32) -> bool {
    let matrix = camera.matrix;
    let row0 = vec4f(matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]);
    let row1 = vec4f(matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]);
    let row2 = vec4f(matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]);
    let row3 = vec4f(matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]);

    return sphere_intersects_plane(row0 + row3, center, radius)
        && sphere_intersects_plane(row3 - row0, center, radius)
        && sphere_intersects_plane(row1 + row3, center, radius)
        && sphere_intersects_plane(row3 - row1, center, radius)
        && sphere_intersects_plane(row2, center, radius)
        && sphere_intersects_plane(row3 - row2, center, radius);
}

/// The tile range covered by a normalized screen interval.
///
/// The one-tile expansion keeps a conservative edge tolerance so a light is
/// never lost between neighboring clusters because of rounding.
fn tile_span(low: f32, high: f32, count: u32) -> vec2u {
    let padding = 1.0 / f32(count);
    let lo = clamp(low - padding, 0.0, 1.0);
    let hi = clamp(high + padding, 0.0, 1.0);
    let min_tile = min(u32(floor(lo * f32(count))), count - 1u);
    let max_tile = min(u32(floor(hi * f32(count))), count - 1u);
    return vec2u(min_tile, max_tile);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.local_count) {
        return;
    }

    let source_index = params.local_offset + id.x;
    if (source_index >= params.light_count || source_index >= arrayLength(&lights)) {
        return;
    }

    let light = lights[source_index];
    let position = light.origin_range.xyz;
    let range = max(light.origin_range.w, 0.0001);
    let distance = length(position - camera.position.xyz);

    if (distance > params.max_distance + range || distance + range < params.min_distance) {
        return;
    }

    if (!sphere_intersects_camera(position, range)) {
        return;
    }

    let center_clip = camera.matrix * vec4f(position, 1.0);
    let grid = vec2u(params.grid_x, params.grid_y);
    var min_tile = vec2u(0u);
    var max_tile = grid - vec2u(1u);

    let row0 = vec4f(
        camera.matrix[0][0],
        camera.matrix[1][0],
        camera.matrix[2][0],
        camera.matrix[3][0],
    );
    let row1 = vec4f(
        camera.matrix[0][1],
        camera.matrix[1][1],
        camera.matrix[2][1],
        camera.matrix[3][1],
    );
    let row3 = vec4f(
        camera.matrix[0][3],
        camera.matrix[1][3],
        camera.matrix[2][3],
        camera.matrix[3][3],
    );

    // The sphere's nearest point can be much closer than its center. Using the
    // center depth alone underestimates the screen footprint of nearby lights.
    let near_depth = center_clip.w - length(row3.xyz) * range;
    if (near_depth > 0.0001) {
        let inverse_depth = 1.0 / near_depth;
        let max_x = (abs(center_clip.x) + length(row0.xyz) * range) * inverse_depth;
        let max_y = (abs(center_clip.y) + length(row1.xyz) * range) * inverse_depth;
        let low = vec2f(0.5 - 0.5 * max_x, 0.5 - 0.5 * max_y);
        let high = vec2f(0.5 + 0.5 * max_x, 0.5 + 0.5 * max_y);

        let x_span = tile_span(low.x, high.x, params.grid_x);
        let y_span = tile_span(low.y, high.y, params.grid_y);
        min_tile = vec2u(x_span.x, y_span.x);
        max_tile = vec2u(x_span.y, y_span.y);
    }

    let near_distance = max(distance - range, params.min_distance);
    let far_distance = min(distance + range, params.max_distance);
    let min_slice = slice_of(near_distance);
    let max_slice = slice_of(far_distance);

    for (var z = min_slice; z <= max_slice; z += 1u) {
        for (var y = min_tile.y; y <= max_tile.y; y += 1u) {
            for (var x = min_tile.x; x <= max_tile.x; x += 1u) {
                let cluster = (z * params.grid_y + y) * params.grid_x + x;
                let slot = atomicAdd(&cluster_counts[cluster], 1u);

                if (slot < params.max_per_cluster) {
                    cluster_lights[cluster * params.max_per_cluster + slot] = source_index;
                }
            }
        }
    }
}
