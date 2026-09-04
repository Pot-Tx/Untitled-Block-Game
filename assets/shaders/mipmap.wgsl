struct Params {
    size: vec2<u32>,
    level: u32,
}

@group(0) @binding(0)
var textures: texture_2d_array<f32>;
@group(1) @binding(0)
var texstorage: texture_storage_2d_array<rgba8unorm, write>;
@group(1) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv_dst = vec2<u32>(id.xy);
    let size = params.size;
    if uv_dst.x >= size.x || uv_dst.y >= size.y {
        return;
    }
    let uv_src = uv_dst * 2;
    let idx = id.z;
    let level = params.level - 1;

    var rgb = vec3f(0.0);
    var a = 0.0;
    for (var i: u32 = 0; i < 2; i += 1) {
        for (var j: u32 = 0; j < 2; j += 1) {
            let offset = vec2<u32>(i, j);
            let src = textureLoad(textures, uv_src + offset, idx, level);
            rgb += src.rgb;
            a += src.a;
        }
    }

    rgb *= 0.25;
    a *= 0.25;
    rgb = select(rgb * 12.92, 1.055 * pow(rgb, vec3f(1.0 / 2.4)) - 0.055, rgb > vec3f(0.0031308));
    textureStore(texstorage, uv_dst, idx, vec4f(rgb, a));
}