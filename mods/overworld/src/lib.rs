use block_game_sdk::worldgen as wg;
use block_game_sdk::worldgen::abi;
use glam::{U8Vec3, Vec3, Vec3Swizzles};
use noise_functions::{CellDistanceSq, Noise, Perlin};
use std::alloc::{Layout, alloc as sys_alloc, dealloc as sys_dealloc};

#[unsafe(no_mangle)]
pub extern "C" fn abi_version() -> u32 {
    abi::ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(size: u32) -> u32 {
    let layout = Layout::from_size_align(size.max(1) as usize, 8).expect("valid layout");
    let ptr = unsafe { sys_alloc(layout) };

    if ptr.is_null() { 0 } else { ptr as u32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn dealloc(ptr: u32, size: u32) {
    if ptr == 0 {
        return;
    }

    let layout = Layout::from_size_align(size.max(1) as usize, 8).expect("valid layout");
    unsafe { sys_dealloc(ptr as *mut u8, layout) };
}

#[unsafe(no_mangle)]
pub extern "C" fn sample_base(raw: *const abi::SampleCtx) -> u32 {
    let mut ctx = unsafe { wg::SampleCtx::new(&*raw) };
    let climate = ctx.channel("climate").expect("climate channel");
    let density = ctx.channel("density").expect("density channel");
    let erosion = ctx.channel("erosion").expect("erosion channel");
    let positions = ctx.positions().collect::<Vec<_>>();

    for pos in positions {
        let world = ctx.world_pos(pos);
        let value = world.as_vec3();

        ctx.set_channel(climate, pos, &Vec3::ZERO.to_array());

        let h = Perlin::default().frequency(0.015625).sample2(value.xz()) * 32.0;
        let a = 0.125;
        let b = 0.0625;
        let d1 = (if h > 0.0 {
            ((h * a).exp() - 1.0) / a
        } else {
            (1.0 - (h * -b).exp()) / b
        } - value.y)
            * 0.03125;
        let d2 = Perlin::default().frequency(0.0625).sample3(value) * 0.75;

        ctx.set(density, pos, 0, (d1.tanh() + d2).tanh());

        let mut value = value;
        value.y *= 2.0;
        let v1 = CellDistanceSq::default()
            .jitter(0.75)
            .frequency(0.03125)
            .sample3(value);
        let v2 = CellDistanceSq::default()
            .jitter(1.25)
            .frequency(0.03125)
            .sample3(value);
        let v3 = -(value.y + 128.0) * 0.00390625;

        ctx.set(erosion, pos, 0, (v1 + v2 + v3).tanh());
    }

    0
}

#[unsafe(no_mangle)]
pub extern "C" fn ground(raw: *const abi::ChunkCtx) -> u32 {
    let mut ctx = unsafe { wg::ChunkCtx::new(&*raw) };
    let density = ctx.channel("density").expect("density channel");
    let erosion = ctx.channel("erosion").expect("erosion channel");
    let gradient = ctx.channel("density_gradient").expect("gradient channel");
    let stone = ctx.block("bricks");
    let dirt = ctx.block("dirt");
    let air = ctx.block("air");
    let positions = ctx.positions().collect::<Vec<_>>();

    for pos in positions {
        let sample = ctx.sample(pos);
        let meta = if sample.scalar(density) > 0.0 && sample.scalar(erosion) < 0.0 {
            if sample.scalar(density) < 0.125
                && sample.component(gradient, 1) < 0.0
                && sample.vector(gradient).xz().length_squared() < 0.00390625
            {
                dirt
            } else {
                stone
            }
        } else {
            air
        };

        ctx.set(pos, meta);
    }

    0
}

#[unsafe(no_mangle)]
pub extern "C" fn grass(raw: *const abi::ChunkCtx) -> u32 {
    let mut ctx = unsafe { wg::ChunkCtx::new(&*raw) };
    let dirt = ctx.block("dirt");
    let grass = ctx.block("grass");
    let air = ctx.block("air");
    let stride = ctx.stride();

    for x in 1..stride - 1 {
        for y in 1..stride - 1 {
            for z in 1..stride - 1 {
                let pos = U8Vec3::new(x, y, z);

                if ctx.get(pos) == dirt && ctx.get(pos + U8Vec3::Y) == air {
                    ctx.set(pos, grass);
                }
            }
        }
    }

    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tree_sites(raw: *const abi::PlaceCtx) -> u32 {
    let mut ctx = unsafe { wg::PlaceCtx::new(&*raw) };
    let density = ctx.channel("density").expect("density channel");
    let erosion = ctx.channel("erosion").expect("erosion channel");
    let anchor = U8Vec3::new(
        ctx.structure().anchor[0] as u8,
        ctx.structure().anchor[1] as u8,
        ctx.structure().anchor[2] as u8,
    );

    for _ in 0..64 {
        let site = ctx.random_site();
        let base = site + anchor;
        let root = ctx.sample(base);

        if !(root.scalar(density) > 0.0
            && root.scalar(erosion) < 0.0
            && root.scalar(density) < 0.125)
        {
            continue;
        }

        let blocked = (3..6).any(|dy| {
            let sample = ctx.sample(base + U8Vec3::new(0, dy as u8, 0));
            sample.scalar(density) > 0.0 && sample.scalar(erosion) < 0.0
        });

        if !blocked {
            ctx.push_site(site);
        }
    }

    ctx.sites()
}

#[unsafe(no_mangle)]
pub extern "C" fn camp_sites(raw: *const abi::PlaceCtx) -> u32 {
    let mut ctx = unsafe { wg::PlaceCtx::new(&*raw) };
    let density = ctx.channel("density").expect("density channel");
    let erosion = ctx.channel("erosion").expect("erosion channel");
    let anchor = U8Vec3::new(
        ctx.structure().anchor[0] as u8,
        ctx.structure().anchor[1] as u8,
        ctx.structure().anchor[2] as u8,
    );

    for _ in 0..16 {
        let site = ctx.random_site();
        let center = site + anchor + U8Vec3::new(1, 0, 1);
        let root = ctx.sample(center);

        if !(root.scalar(density) > 0.0
            && root.scalar(erosion) < 0.0
            && root.scalar(density) < 0.125)
        {
            continue;
        }

        let clear = (1..3).all(|dy| {
            let sample = ctx.sample(center + U8Vec3::new(0, dy as u8, 0));
            sample.scalar(density) <= 0.0 || sample.scalar(erosion) >= 0.0
        });

        if clear {
            ctx.push_site(site);
        }
    }

    ctx.sites()
}
