struct TemplateVert {
    pos: vec3f,
    uv: vec2f,
}

struct TemplateModel {
    norm: vec3f,
    spans: u32,
    tangent: vec3f,
    bitangent: vec3f,
    verts: array<TemplateVert, 4>,
}

fn snap_world(
    world: vec3f,
    uv: vec2f,
    tangent: vec3f,
    bitangent: vec3f,
    texture_size: vec2f,
    scale: f32,
) -> vec3f {
    let snapped_uv = (floor(uv * texture_size) + 0.5) / texture_size;
    let delta = snapped_uv - uv;
    return world + (tangent * delta.x + bitangent * delta.y) * scale;
}
