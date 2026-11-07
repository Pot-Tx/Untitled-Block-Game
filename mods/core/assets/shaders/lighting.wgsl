// Unified lighting for sky, block, and entity sources.
//
// A light with zero origin range is directional. A light with positive range is
// local. Consecutive shadow views describe cascades, a cone, or six cube faces.

struct Camera {
    matrix: mat4x4<f32>,
    position: vec4f,
}

@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    origin_range: vec4f,
    direction_outer: vec4f,
    color_inner: vec4f,
    intensity: f32,
    shadow_first: u32,
    shadow_count: u32,
    flags: u32,
}

struct ShadowView {
    matrix: mat4x4f,
    texel: f32,
    range: f32,
    extent: f32,
    scale: f32,
    atlas: u32,
    layer: u32,
    bias_scale: f32,
    normal_offset: f32,
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

@group(2) @binding(0)
var<storage, read> lights: array<Light>;
@group(2) @binding(1)
var<storage, read> shadow_views: array<ShadowView>;
@group(2) @binding(2)
var<uniform> params: LightParams;
@group(2) @binding(3)
var sun_shadows: texture_depth_2d_array;
@group(2) @binding(4)
var local_shadows: texture_depth_2d_array;
@group(2) @binding(5)
var shadow_sampler: sampler_comparison;
@group(2) @binding(6)
var<storage, read> cluster_counts: array<u32>;
@group(2) @binding(7)
var<storage, read> cluster_lights: array<u32>;
@group(2) @binding(8)
var sun_transmission_accum: texture_2d_array<f32>;
@group(2) @binding(9)
var sun_transmission_reveal: texture_2d_array<f32>;
@group(2) @binding(10)
var local_transmission_accum: texture_2d_array<f32>;
@group(2) @binding(11)
var local_transmission_reveal: texture_2d_array<f32>;
@group(2) @binding(12)
var transmission_sampler: sampler;

const EDGE_FADE: f32 = 0.08;
const FILTER_REACH: f32 = 1.2;

struct LightDirection {
    to_light: vec3f,
    distance_sq: f32,
    range_sq: f32,
    local: bool,
}

fn light_direction(light: Light, world: vec3f) -> LightDirection {
    if (light.origin_range.w <= 0.0) {
        return LightDirection(
            normalize(light.direction_outer.xyz),
            0.0,
            0.0,
            false,
        );
    }

    let delta = light.origin_range.xyz - world;
    let distance_sq = dot(delta, delta);
    return LightDirection(
        delta * inverseSqrt(max(distance_sq, 0.000001)),
        distance_sq,
        light.origin_range.w * light.origin_range.w,
        true,
    );
}

fn sample_shadow_view(
    view_index: u32,
    world: vec3f,
    norm: vec3f,
    to_light: vec3f,
    fade: bool,
) -> vec3f {
    let view = shadow_views[view_index];
    let clip = view.matrix * vec4f(world + norm * view.normal_offset, 1.0);
    if (clip.w <= 0.0001) {
        return vec3f(1.0);
    }
    let uv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + vec2f(0.5);
    if (view.atlas != 0u
        && (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)))) {
        return vec3f(1.0);
    }
    let cosine = max(dot(norm, to_light), 0.05);
    let tangent = sqrt(max(1.0 - cosine * cosine, 0.0)) / cosine;
    let depth = clip.z / clip.w
        + view.bias_scale * (1.0 + 2.0 * FILTER_REACH * tangent);
    let atlas_uv = uv * view.scale;

    var lit = 1.0;
    if (view.atlas == 0u) {
        lit = textureSampleCompareLevel(
            sun_shadows,
            shadow_sampler,
            atlas_uv,
            i32(view.layer),
            depth,
        );
    } else {
        lit = textureSampleCompareLevel(
            local_shadows,
            shadow_sampler,
            atlas_uv,
            i32(view.layer),
            depth,
        );
    }

    if (fade) {
        let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        lit = mix(1.0, lit, smoothstep(0.0, EDGE_FADE, edge));
    }

    var accum: vec4f;
    var reveal: vec4f;
    if (view.atlas == 0u) {
        accum = textureSampleLevel(
            sun_transmission_accum,
            transmission_sampler,
            atlas_uv,
            i32(view.layer),
            0.0,
        );
        reveal = textureSampleLevel(
            sun_transmission_reveal,
            transmission_sampler,
            atlas_uv,
            i32(view.layer),
            0.0,
        );
    } else {
        accum = textureSampleLevel(
            local_transmission_accum,
            transmission_sampler,
            atlas_uv,
            i32(view.layer),
            0.0,
        );
        reveal = textureSampleLevel(
            local_transmission_reveal,
            transmission_sampler,
            atlas_uv,
            i32(view.layer),
            0.0,
        );
    }

    let average_tint = accum.rgb / max(accum.a, 0.0001);
    let coverage = 1.0 - clamp(reveal.r, 0.0, 1.0);
    let transmission = mix(vec3f(1.0), average_tint, coverage);
    return vec3f(lit) * transmission;
}

fn directional_shadow(light: Light, world: vec3f, norm: vec3f, to_light: vec3f) -> vec3f {
    let distance = length(world - camera.position.xyz);
    var selected = 0u;
    for (var c = 0u; c + 1u < light.shadow_count; c += 1u) {
        if (distance <= shadow_views[light.shadow_first + c].extent) {
            break;
        }
        selected = c + 1u;
    }
    let last = selected + 1u == light.shadow_count;
    let near = sample_shadow_view(
        light.shadow_first + selected,
        world,
        norm,
        to_light,
        last,
    );

    if (last) {
        return near;
    }

    let extent = shadow_views[light.shadow_first + selected].extent;
    let blend = 0.8 * extent;
    if (distance <= blend) {
        return near;
    }

    let far = sample_shadow_view(
        light.shadow_first + selected + 1u,
        world,
        norm,
        to_light,
        selected + 2u == light.shadow_count,
    );
    return mix(near, far, (distance - blend) / (extent - blend));
}

fn local_shadow(light: Light, world: vec3f, norm: vec3f, to_light: vec3f) -> vec3f {
    var view_index = light.shadow_first;
    if (light.shadow_count == 6u) {
        let direction = -to_light;
        let absolute = abs(direction);
        if (absolute.x >= absolute.y && absolute.x >= absolute.z) {
            view_index += select(1u, 0u, direction.x > 0.0);
        } else if (absolute.y >= absolute.z) {
            view_index += select(3u, 2u, direction.y > 0.0);
        } else {
            view_index += select(5u, 4u, direction.z > 0.0);
        }
    }
    return sample_shadow_view(view_index, world, norm, to_light, false);
}

fn shadow_visibility(
    light: Light,
    direction: LightDirection,
    world: vec3f,
    norm: vec3f,
) -> vec3f {
    if (light.shadow_count == 0u) {
        return vec3f(1.0);
    }
    if (direction.local) {
        return local_shadow(light, world, norm, direction.to_light);
    }
    return directional_shadow(light, world, norm, direction.to_light);
}

fn diffuse_light(
    light: Light,
    direction: LightDirection,
    norm: vec3f,
) -> vec3f {
    if (all(light.color_inner.rgb == vec3f(0.0))) {
        return vec3f(0.0);
    }

    let facing = max(dot(norm, direction.to_light), 0.0);
    if (facing <= 0.0) {
        return vec3f(0.0);
    }

    var attenuation = 1.0;
    var cone = 1.0;
    if (direction.local) {
        if (direction.distance_sq >= direction.range_sq) {
            return vec3f(0.0);
        }
        let fade = saturate(1.0 - direction.distance_sq / direction.range_sq);
        attenuation = fade * fade;

        if (light.direction_outer.w > -0.999) {
            let axis = normalize(light.direction_outer.xyz);
            cone = smoothstep(
                light.direction_outer.w,
                light.color_inner.w,
                dot(-direction.to_light, axis),
            );
        }
    }

    return light.color_inner.rgb
        * light.intensity
        * facing
        * attenuation
        * cone;
}

fn unshadowed_light(light: Light, norm: vec3f) -> vec3f {
    return diffuse_light(light, light_direction(light, vec3f(0.0)), norm);
}

fn evaluate_light(
    light: Light,
    world: vec3f,
    shading_normal: vec3f,
    geometric_normal: vec3f,
) -> vec3f {
    let direction = light_direction(light, world);
    let radiance = diffuse_light(light, direction, shading_normal);
    if (all(radiance == vec3f(0.0)) || light.shadow_count == 0u) {
        return radiance;
    }
    return radiance * shadow_visibility(light, direction, world, geometric_normal);
}

fn ambient_radiance(norm: vec3f) -> vec3f {
    var radiance = vec3f(0.0);
    let count = min(params.unshadowed_directional_count, arrayLength(&lights));
    for (var i = 0u; i < count; i += 1u) {
        radiance += unshadowed_light(lights[i], norm);
    }
    return radiance;
}

fn cast_radiance(
    world: vec3f,
    shading_normal: vec3f,
    geometric_normal: vec3f,
) -> vec3f {
    var radiance = vec3f(0.0);
    let count = min(params.directional_count, arrayLength(&lights));
    for (var i = params.unshadowed_directional_count; i < count; i += 1u) {
        radiance += evaluate_light(lights[i], world, shading_normal, geometric_normal);
    }
    return radiance;
}

fn local_radiance(
    world: vec3f,
    shading_normal: vec3f,
    geometric_normal: vec3f,
    pixel: vec2f,
) -> vec3f {
    var radiance = vec3f(0.0);
    if (params.local_count == 0u) {
        return radiance;
    }

    let grid = vec3u(params.grid_x, params.grid_y, params.grid_z);
    let screen = max(
        vec2f(f32(params.screen_width), f32(params.screen_height)),
        vec2f(1.0),
    );
    let uv = clamp(pixel / screen, vec2f(0.0), vec2f(0.999999));
    let x = min(u32(uv.x * f32(grid.x)), grid.x - 1u);
    let y = min(u32(uv.y * f32(grid.y)), grid.y - 1u);
    let distance = length(world - camera.position.xyz);
    let span = max(params.max_distance - params.min_distance, 0.0001);
    let z = min(
        u32(saturate((distance - params.min_distance) / span) * f32(grid.z)),
        grid.z - 1u,
    );
    let cluster = (z * grid.y + y) * grid.x + x;
    let count = min(cluster_counts[cluster], params.max_per_cluster);

    for (var slot = 0u; slot < count; slot += 1u) {
        let index = cluster_lights[cluster * params.max_per_cluster + slot];
        radiance += evaluate_light(lights[index], world, shading_normal, geometric_normal);
    }
    return radiance;
}

fn lighting(
    shading_normal: vec3f,
    geometric_normal: vec3f,
    world: vec3f,
    pixel: vec2f,
) -> vec3f {
    return cast_radiance(world, shading_normal, geometric_normal)
        + local_radiance(world, shading_normal, geometric_normal, pixel);
}
