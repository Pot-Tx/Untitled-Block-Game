//! Unified light sources and shadow views.
//!
//! Sky, block, and entity lights differ only in how their source data is
//! produced. Once prepared, all of them use the same GPU descriptors, the same
//! shadow-view format, and the same shader evaluator.

use crate::actor::{Position, Rotation};
use crate::components;
use crate::ecs::*;
use crate::game::{ASSET_OPTIONS, TotalTicks};
use crate::render::*;
use crate::util::bounding::AABB;
use crate::util::transform::Trans4;
use crate::world::{
    BLOCK_SURFACES, BLOCK_TEMPLATES, BLOCK_TEXTURES, BlockPos, BlockRectInst, BlockTemplates,
    BlockTextureSet, World,
};
use bytemuck::{Pod, Zeroable};
use glam::*;
use log::warn;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::mem::size_of;
use wgpu::*;

const LOCAL_SHADOW_FACES: usize = 6;
const BLOCK_LIGHT_FLAG: u32 = 1;
const CUBE_DIRECTIONS: [Vec3; LOCAL_SHADOW_FACES] = [
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
    Vec3::Z,
    Vec3::NEG_Z,
];

/// One cascade of the directional shadow atlas.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Cascade {
    pub extent: f32,
    pub resolution: u32,
}

impl Cascade {
    #[inline]
    fn texel(&self) -> f32 {
        2.0 * self.extent / self.resolution as f32
    }
}

/// User-tunable limits and quality settings for all lighting.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LightingSettings {
    pub directional_light_distance: f32,
    pub shadow_margin: f32,
    pub max_local_lights: u32,
    pub max_shadowed_local_lights: u32,
    pub local_shadow_resolution: u32,
    pub cluster_grid: [u32; 3],
    pub max_cluster_lights: u32,
    pub cluster_near: f32,
    pub cluster_far: f32,
    pub cascades: Vec<Cascade>,
}

impl LightingSettings {
    /// Reads `settings/lighting.ron`.
    pub fn load() -> anyhow::Result<Self> {
        let file = File::open("settings/lighting.ron")?;
        let settings: Self = ASSET_OPTIONS.from_reader(BufReader::new(file))?;
        settings.validate()?;
        Ok(settings)
    }

    fn validate(&self) -> anyhow::Result<()> {
        if !self.directional_light_distance.is_finite()
            || !self.shadow_margin.is_finite()
            || self.shadow_margin <= 0.0
            || self.shadow_margin >= self.directional_light_distance
        {
            anyhow::bail!(
                "shadow_margin must be positive and smaller than directional_light_distance"
            );
        }
        if self.max_local_lights == 0 {
            anyhow::bail!("max_local_lights must be positive");
        }
        if self.max_shadowed_local_lights == 0
            || self.max_shadowed_local_lights > self.max_local_lights
        {
            anyhow::bail!("max_shadowed_local_lights must be between one and max_local_lights");
        }
        if self.local_shadow_resolution == 0 {
            anyhow::bail!("local_shadow_resolution must be positive");
        }
        if self.cluster_grid.contains(&0) {
            anyhow::bail!("every cluster_grid dimension must be positive");
        }
        let cluster_count = (self.cluster_grid[0] as usize)
            .checked_mul(self.cluster_grid[1] as usize)
            .and_then(|count| count.checked_mul(self.cluster_grid[2] as usize))
            .ok_or_else(|| anyhow::anyhow!("cluster_grid is too large"))?;
        if cluster_count == 0 {
            anyhow::bail!("cluster_grid must contain at least one cluster");
        }
        if self.max_cluster_lights == 0 {
            anyhow::bail!("max_cluster_lights must be positive");
        }
        if !self.cluster_near.is_finite()
            || !self.cluster_far.is_finite()
            || self.cluster_near < 0.0
            || self.cluster_far <= self.cluster_near
        {
            anyhow::bail!("cluster_far must be greater than cluster_near");
        }
        if self.cascades.is_empty() {
            anyhow::bail!("the lighting settings need at least one cascade");
        }

        let mut extent = 0.0;
        for cascade in &self.cascades {
            if !cascade.extent.is_finite() || cascade.extent <= extent {
                anyhow::bail!("cascades must reach farther out one after another");
            }
            if cascade.resolution == 0 {
                anyhow::bail!("a cascade needs at least one texel");
            }
            extent = cascade.extent;
        }
        Ok(())
    }

    fn cascade_resolution(&self) -> u32 {
        self.cascades
            .iter()
            .map(|cascade| cascade.resolution)
            .max()
            .unwrap_or(1)
    }

    fn cluster_count(&self) -> usize {
        self.cluster_grid[0] as usize
            * self.cluster_grid[1] as usize
            * self.cluster_grid[2] as usize
    }

    fn local_shadow_layers(&self) -> usize {
        self.max_shadowed_local_lights as usize * LOCAL_SHADOW_FACES
    }
}

/// A sky light whose direction and radiance move with the tick.
pub struct Sun {
    rot_of_tick: fn(u64) -> Vec3,
    color_of_tick: fn(u64) -> Vec3,
    pub cur: (Vec3, Vec3),
    pub next: (Vec3, Vec3),
}

impl Sun {
    pub fn new(rot_of_tick: fn(u64) -> Vec3, color_of_tick: fn(u64) -> Vec3, tick: u64) -> Self {
        let mut sun = Self {
            rot_of_tick,
            color_of_tick,
            cur: (Vec3::ZERO, Vec3::ZERO),
            next: (Vec3::ZERO, Vec3::ZERO),
        };
        sun.update(tick);
        sun
    }

    pub fn update(&mut self, tick: u64) {
        self.cur = ((self.rot_of_tick)(tick), (self.color_of_tick)(tick));
        self.next = ((self.rot_of_tick)(tick + 1), (self.color_of_tick)(tick + 1));
    }
}

/// The CPU description of sky lights. Their GPU data is owned by [`Lighting`].
pub struct Sky {
    pub cast: Vec<Sun>,
    pub ambient: Vec<Sun>,
}

impl Resource for Sky {}

impl Sky {
    pub fn new(cast: Vec<Sun>, ambient: Vec<Sun>) -> Self {
        Self { cast, ambient }
    }

    pub fn update(&mut self, tick: u64) {
        for sun in self.cast.iter_mut().chain(self.ambient.iter_mut()) {
            sun.update(tick);
        }
    }

    fn direction(rotation: Vec3) -> Vec3 {
        Mat4::rotation(rotation.x, rotation.y, rotation.z).transform_vector3(Vec3::Z)
    }
}

components!(
    /// A light emitted by an entity. Its position and direction are relative to
    /// the actor and follow its placement and rotation.
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct Emission {
        pub offset: Vec3,
        pub direction: Vec3,
        pub color: Vec3,
        pub intensity: f32,
        pub range: f32,
        pub inner_cos: f32,
        pub outer_cos: f32,
    }: Cold, asset, save;
);

/// One light as every world shader reads it.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub(crate) struct Light {
    /// xyz is the position of a local light; w is zero for infinite directional
    /// lights and the positive reach of a local light.
    pub origin_range: Vec4,
    /// xyz points towards an infinite light or along a cone, and w is the outer
    /// cone cosine.
    pub direction_outer: Vec4,
    /// rgb is the radiance and w is the inner cone cosine.
    pub color_inner: Vec4,
    pub intensity: f32,
    pub shadow_first: u32,
    pub shadow_count: u32,
    pub flags: u32,
}

impl Light {
    fn directional(direction: Vec3, color: Vec3, shadow_first: u32, shadow_count: u32) -> Self {
        Self {
            origin_range: Vec4::ZERO,
            direction_outer: direction.normalize_or_zero().extend(-1.0),
            color_inner: color.extend(1.0),
            intensity: 1.0,
            shadow_first,
            shadow_count,
            flags: 0,
        }
    }

    fn from_block(pos: BlockPos, emission: Emission) -> Self {
        let direction = emission.direction.normalize_or_zero();
        let outer = if direction == Vec3::ZERO {
            -1.0
        } else {
            emission.outer_cos
        };

        Self {
            origin_range: (pos.as_vec3() + emission.offset).extend(emission.range),
            direction_outer: direction.extend(outer),
            color_inner: emission.color.extend(emission.inner_cos),
            intensity: emission.intensity,
            shadow_first: 0,
            shadow_count: 0,
            flags: BLOCK_LIGHT_FLAG,
        }
    }

    fn from_entity(light: &Emission, pos: Vec3, rotation: &Rotation) -> Self {
        let direction = if light.outer_cos > -0.999 {
            let given = rotation.transform(light.direction).normalize_or_zero();
            if given == Vec3::ZERO {
                rotation.direction()
            } else {
                given
            }
        } else {
            Vec3::ZERO
        };

        Self {
            origin_range: (pos + rotation.transform(light.offset)).extend(light.range),
            direction_outer: direction.extend(light.outer_cos),
            color_inner: light.color.extend(light.inner_cos),
            intensity: light.intensity,
            shadow_first: 0,
            shadow_count: 0,
            flags: 0,
        }
    }

    #[inline]
    fn is_local(&self) -> bool {
        self.origin_range.w > 0.0
    }
}

/// One shadow lookup view.
///
/// A directional light owns consecutive cascade views, a spot owns one view, and
/// an omnidirectional light owns six views.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct ShadowView {
    matrix: Mat4,
    texel: f32,
    range: f32,
    extent: f32,
    scale: f32,
    atlas: u32,
    layer: u32,
    bias_scale: f32,
    normal_offset: f32,
}

/// The light and cluster parameters shared by all passes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
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

/// The uniform transform handed to one shadow render pass.
pub(crate) struct ShadowTransform;

impl BindSignature for ShadowTransform {
    const NAME: &'static str = "shadow";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    type Content<'a> = &'a Buffer;
}

struct ShadowPass {
    view: TextureView,
    accum: TextureView,
    reveal: TextureView,
    transform: Buffer,
    pass: BindSet<ShadowTransform>,
}

impl ShadowPass {
    fn new(
        canvas: &Canvas,
        shadow: &Texture,
        accum: &Texture,
        reveal: &Texture,
        layer: u32,
    ) -> Self {
        let view = TextureView::new(
            shadow,
            &LayerViewConfig {
                name: "shadow",
                layer,
            },
        );
        let transform = Buffer::new(
            canvas,
            &BufferConfig {
                name: "shadow",
                init: BufferInit::Content(&[Mat4::IDENTITY]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        );
        let pass = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "shadow",
                content: &transform,
            },
        );
        let accum = TextureView::new(
            accum,
            &LayerViewConfig {
                name: "transmission_accum",
                layer,
            },
        );
        let reveal = TextureView::new(
            reveal,
            &LayerViewConfig {
                name: "transmission_reveal",
                layer,
            },
        );
        Self {
            view,
            accum,
            reveal,
            transform,
            pass,
        }
    }
}

/// One shadow view that the next depth pass renders.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShadowDraw {
    pub view: usize,
    pub resolution: u32,
    pub bound: AABB<Vec3>,
}

/// Every GPU light, shadow view, cluster, and bind group in the world.
pub(crate) struct Lighting {
    settings: LightingSettings,
    lights: Buffer,
    light_capacity: usize,
    views: Buffer,
    params: Buffer,
    cluster_counts: Buffer,
    cluster_clear: Vec<u32>,
    cluster_batch: ComputeBatch<LightClusterLayout>,
    cluster_bind: BindSet<LightClusterLayout>,

    _sun_shadow: Texture,
    _sun_shadow_view: TextureView,
    _sun_transmission_accum: Texture,
    _sun_transmission_accum_view: TextureView,
    _sun_transmission_reveal: Texture,
    _sun_transmission_reveal_view: TextureView,
    _local_shadow: Texture,
    _local_shadow_view: TextureView,
    _local_transmission_accum: Texture,
    _local_transmission_accum_view: TextureView,
    _local_transmission_reveal: Texture,
    _local_transmission_reveal_view: TextureView,
    shadow_passes: Vec<ShadowPass>,
    sun_layers: usize,
    sun_centers: Vec<Option<Vec3>>,
    shadow_draws: Vec<ShadowDraw>,
    local_overflowed: bool,

    pub bind_group: BindSet<WorldLightsLayout>,
}

impl Resource for Lighting {}

impl Lighting {
    pub(crate) fn new(
        canvas: &Canvas,
        camera: &Camera,
        sky: &Sky,
        settings: LightingSettings,
    ) -> Self {
        let local_shadow_layers = settings.local_shadow_layers();
        assert!(
            local_shadow_layers <= canvas.device.limits().max_texture_array_layers as usize,
            "local shadow layers exceed the device limit"
        );
        assert!(
            settings.local_shadow_resolution <= canvas.device.limits().max_texture_dimension_2d,
            "local shadow resolution exceeds the device limit"
        );

        let real_sun_layers = sky.cast.len() * settings.cascades.len();
        let sun_layers = real_sun_layers.max(1);
        assert!(
            sun_layers <= canvas.device.limits().max_texture_array_layers as usize,
            "sun shadow layers exceed the device limit"
        );
        let view_capacity = sun_layers + local_shadow_layers;
        let light_capacity =
            sky.ambient.len() + sky.cast.len() + settings.max_local_lights as usize;
        let light_capacity = light_capacity.max(1);
        let sun_resolution = settings.cascade_resolution();
        let cluster_count = settings.cluster_count();

        let lights = Buffer::new(
            canvas,
            &BufferConfig {
                name: "lights",
                init: BufferInit::<Light>::Size(
                    light_capacity as BufferAddress * size_of::<Light>() as BufferAddress,
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        );
        let views = Buffer::new(
            canvas,
            &BufferConfig {
                name: "shadow_views",
                init: BufferInit::<ShadowView>::Size(
                    view_capacity as BufferAddress * size_of::<ShadowView>() as BufferAddress,
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        );
        let params = Buffer::new(
            canvas,
            &BufferConfig {
                name: "light_params",
                init: BufferInit::Content(&[LightParams::default()]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        );
        let cluster_counts = Buffer::new(
            canvas,
            &BufferConfig {
                name: "light_cluster_counts",
                init: BufferInit::<u32>::Size(
                    cluster_count as BufferAddress * size_of::<u32>() as BufferAddress,
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        );
        let cluster_lights = Buffer::new(
            canvas,
            &BufferConfig {
                name: "light_cluster_indices",
                init: BufferInit::<u32>::Size(
                    (cluster_count * settings.max_cluster_lights as usize) as BufferAddress
                        * size_of::<u32>() as BufferAddress,
                ),
                usage: BufferUsages::STORAGE,
            },
        );

        let sun_shadow = Self::render_texture(
            canvas,
            "sun_shadow",
            sun_resolution,
            sun_layers as u32,
            TextureFormat::Depth32Float,
        );
        let local_shadow = Self::render_texture(
            canvas,
            "local_shadow",
            settings.local_shadow_resolution,
            local_shadow_layers as u32,
            TextureFormat::Depth32Float,
        );
        let sun_transmission_accum = Self::render_texture(
            canvas,
            "sun_transmission_accum",
            sun_resolution,
            sun_layers as u32,
            TextureFormat::Rgba16Float,
        );
        let sun_transmission_reveal = Self::render_texture(
            canvas,
            "sun_transmission_reveal",
            sun_resolution,
            sun_layers as u32,
            TextureFormat::R16Float,
        );
        let local_transmission_accum = Self::render_texture(
            canvas,
            "local_transmission_accum",
            settings.local_shadow_resolution,
            local_shadow_layers as u32,
            TextureFormat::Rgba16Float,
        );
        let local_transmission_reveal = Self::render_texture(
            canvas,
            "local_transmission_reveal",
            settings.local_shadow_resolution,
            local_shadow_layers as u32,
            TextureFormat::R16Float,
        );
        let sun_shadow_view = TextureView::new(
            &sun_shadow,
            &TextureViewConfig {
                name: "sun_shadow",
                dimension: TextureViewDimension::D2Array,
                mip_level: None,
            },
        );
        let local_shadow_view = TextureView::new(
            &local_shadow,
            &TextureViewConfig {
                name: "local_shadow",
                dimension: TextureViewDimension::D2Array,
                mip_level: None,
            },
        );
        let sun_transmission_accum_view = TextureView::new(
            &sun_transmission_accum,
            &TextureViewConfig {
                name: "sun_transmission_accum",
                dimension: TextureViewDimension::D2Array,
                mip_level: None,
            },
        );
        let sun_transmission_reveal_view = TextureView::new(
            &sun_transmission_reveal,
            &TextureViewConfig {
                name: "sun_transmission_reveal",
                dimension: TextureViewDimension::D2Array,
                mip_level: None,
            },
        );
        let local_transmission_accum_view = TextureView::new(
            &local_transmission_accum,
            &TextureViewConfig {
                name: "local_transmission_accum",
                dimension: TextureViewDimension::D2Array,
                mip_level: None,
            },
        );
        let local_transmission_reveal_view = TextureView::new(
            &local_transmission_reveal,
            &TextureViewConfig {
                name: "local_transmission_reveal",
                dimension: TextureViewDimension::D2Array,
                mip_level: None,
            },
        );
        let mut shadow_passes = Vec::with_capacity(view_capacity);
        shadow_passes.extend((0..sun_layers as u32).map(|layer| {
            ShadowPass::new(
                canvas,
                &sun_shadow,
                &sun_transmission_accum,
                &sun_transmission_reveal,
                layer,
            )
        }));
        shadow_passes.extend((0..local_shadow_layers as u32).map(|layer| {
            ShadowPass::new(
                canvas,
                &local_shadow,
                &local_transmission_accum,
                &local_transmission_reveal,
                layer,
            )
        }));

        let cluster_batch = ComputeBatch::new(
            canvas,
            &ComputeBatchConfig {
                name: "light_clusters",
                shader: "light_cluster",
            },
        );
        let cluster_bind = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "light_clusters",
                content: (
                    &camera.buffer,
                    &lights,
                    &params,
                    &cluster_counts,
                    &cluster_lights,
                ),
            },
        );
        let sampler = Sampler::new(
            canvas,
            &SamplerConfig {
                name: "world_shadow",
                filter: FilterMode::Linear,
                address_mode: AddressMode::ClampToEdge,
                mipmap_filter: MipmapFilterMode::Nearest,
                mip_level_count: 1,
                compare: Some(CompareFunction::Greater),
            },
        );
        let transmission_sampler = Sampler::new(
            canvas,
            &SamplerConfig {
                name: "transmission",
                filter: FilterMode::Linear,
                address_mode: AddressMode::ClampToEdge,
                mipmap_filter: MipmapFilterMode::Nearest,
                mip_level_count: 1,
                compare: None,
            },
        );
        let bind_group = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "world",
                content: WorldLightsContent {
                    lights: &lights,
                    views: &views,
                    params: &params,
                    sun_shadow: &sun_shadow_view,
                    local_shadow: &local_shadow_view,
                    shadow_sampler: &sampler,
                    cluster_counts: &cluster_counts,
                    cluster_lights: &cluster_lights,
                    sun_transmission_accum: &sun_transmission_accum_view,
                    sun_transmission_reveal: &sun_transmission_reveal_view,
                    local_transmission_accum: &local_transmission_accum_view,
                    local_transmission_reveal: &local_transmission_reveal_view,
                    transmission_sampler: &transmission_sampler,
                },
            },
        );

        Self {
            settings,
            lights,
            light_capacity,
            views,
            params,
            cluster_counts,
            cluster_clear: vec![0; cluster_count],
            cluster_batch,
            cluster_bind,
            _sun_shadow: sun_shadow,
            _sun_shadow_view: sun_shadow_view,
            _sun_transmission_accum: sun_transmission_accum,
            _sun_transmission_accum_view: sun_transmission_accum_view,
            _sun_transmission_reveal: sun_transmission_reveal,
            _sun_transmission_reveal_view: sun_transmission_reveal_view,
            _local_shadow: local_shadow,
            _local_shadow_view: local_shadow_view,
            _local_transmission_accum: local_transmission_accum,
            _local_transmission_accum_view: local_transmission_accum_view,
            _local_transmission_reveal: local_transmission_reveal,
            _local_transmission_reveal_view: local_transmission_reveal_view,
            shadow_passes,
            sun_layers,
            sun_centers: vec![None; sun_layers],
            shadow_draws: Vec::new(),
            local_overflowed: false,
            bind_group,
        }
    }

    fn prepare(
        &mut self,
        canvas: &Canvas,
        camera: &Camera,
        sky: &Sky,
        partial: f32,
        mut local: Vec<Light>,
        frame: &mut Frame,
    ) {
        self.shadow_draws.clear();
        let mut lights = Vec::with_capacity(self.light_capacity);
        let mut views =
            vec![ShadowView::default(); self.sun_layers + self.settings.local_shadow_layers()];

        for sun in &sky.ambient {
            let direction = Sky::direction(sun.cur.0.lerp(sun.next.0, partial));
            let color = sun.cur.1.lerp(sun.next.1, partial);
            lights.push(Light::directional(direction, color, 0, 0));
        }
        let unshadowed_directional_count = lights.len() as u32;

        for (light_index, sun) in sky.cast.iter().enumerate() {
            let rotation = sun.cur.0.lerp(sun.next.0, partial);
            let direction = Sky::direction(rotation);
            let color = sun.cur.1.lerp(sun.next.1, partial);
            let first = light_index * self.settings.cascades.len();
            lights.push(Light::directional(
                direction,
                color,
                first as u32,
                self.settings.cascades.len() as u32,
            ));

            for cascade_index in 0..self.settings.cascades.len() {
                let cascade = self.settings.cascades[cascade_index];
                let view_index = first + cascade_index;
                let center = self.sun_center(camera.eye, rotation, cascade.texel(), view_index);
                let eye = center + direction * self.settings.directional_light_distance;
                let transform = self.fit(eye, rotation, cascade.extent);
                canvas.queue.write_buffer(
                    &self.shadow_passes[view_index].transform,
                    0,
                    bytemuck::cast_slice(&[transform]),
                );
                views[view_index] = ShadowView {
                    matrix: transform,
                    texel: cascade.texel(),
                    range: 2.0 * self.settings.shadow_margin,
                    extent: cascade.extent,
                    scale: cascade.resolution as f32 / self.settings.cascade_resolution() as f32,
                    atlas: 0,
                    layer: view_index as u32,
                    bias_scale: cascade.texel() / (2.0 * self.settings.shadow_margin),
                    normal_offset: 0.0,
                };

                if color != Vec3::ZERO {
                    self.shadow_draws.push(ShadowDraw {
                        view: view_index,
                        resolution: cascade.resolution,
                        bound: AABB {
                            min: camera.eye - Vec3::splat(cascade.extent * 2.0),
                            max: camera.eye + Vec3::splat(cascade.extent * 2.0),
                        },
                    });
                }
            }
        }
        let directional_count = lights.len() as u32;

        local.sort_by(|a, b| {
            a.origin_range
                .truncate()
                .distance_squared(camera.eye)
                .total_cmp(&b.origin_range.truncate().distance_squared(camera.eye))
                .then_with(|| a.origin_range.x.total_cmp(&b.origin_range.x))
                .then_with(|| a.origin_range.y.total_cmp(&b.origin_range.y))
                .then_with(|| a.origin_range.z.total_cmp(&b.origin_range.z))
        });
        let max_local_lights = self.settings.max_local_lights as usize;
        let dropped = local.len().saturating_sub(max_local_lights);
        local.truncate(max_local_lights);
        self.prepare_local_shadows(canvas, &mut local, &mut views);
        let local_count = local.len() as u32;
        lights.extend(local);

        let params = LightParams {
            light_count: lights.len() as u32,
            unshadowed_directional_count,
            directional_count,
            local_offset: directional_count,
            local_count,
            grid_x: self.settings.cluster_grid[0],
            grid_y: self.settings.cluster_grid[1],
            grid_z: self.settings.cluster_grid[2],
            max_per_cluster: self.settings.max_cluster_lights,
            screen_width: canvas.surface_config.width,
            screen_height: canvas.surface_config.height,
            _pad0: 0,
            min_distance: self.settings.cluster_near,
            max_distance: self.settings.cluster_far,
            _pad1: 0,
            _pad2: 0,
        };

        if !lights.is_empty() {
            canvas
                .queue
                .write_buffer(&self.lights, 0, bytemuck::cast_slice(&lights));
        }
        canvas
            .queue
            .write_buffer(&self.views, 0, bytemuck::cast_slice(&views));
        canvas
            .queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&[params]));
        canvas.queue.write_buffer(
            &self.cluster_counts,
            0,
            bytemuck::cast_slice(&self.cluster_clear),
        );

        if local_count > 0 {
            frame.compute("light_clusters", |mut pass| {
                self.cluster_batch.begin(&mut pass);
                self.cluster_batch.push(&mut pass, &self.cluster_bind);
                self.cluster_batch
                    .dispatch(&mut pass, local_count.div_ceil(64), 1, 1);
            });
        }

        if dropped > 0 && !self.local_overflowed {
            warn!(
                "dropped {dropped} local light source(s) beyond the limit of {}",
                self.settings.max_local_lights
            );
        }
        self.local_overflowed = dropped > 0;
    }

    /// Creates an empty square texture array used by a lighting pass.
    fn render_texture(
        canvas: &Canvas,
        name: &str,
        resolution: u32,
        layers: u32,
        format: TextureFormat,
    ) -> Texture {
        Texture::new(
            canvas,
            &TextureConfig {
                name,
                texs: vec![None; layers as usize],
                width: resolution,
                height: resolution,
                mip_level_count: 1,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            },
        )
    }

    fn prepare_local_shadows(
        &mut self,
        canvas: &Canvas,
        local: &mut [Light],
        views: &mut [ShadowView],
    ) {
        for (slot, light) in local
            .iter_mut()
            .take(self.settings.max_shadowed_local_lights as usize)
            .enumerate()
        {
            let radius = light.origin_range.w;
            if !light.is_local() || radius <= 0.0 {
                continue;
            }

            let position = light.origin_range.truncate();
            let axis = light.direction_outer.truncate().normalize_or_zero();
            let base = self.sun_layers + slot * LOCAL_SHADOW_FACES;
            let bound = AABB {
                min: position - Vec3::splat(radius),
                max: position + Vec3::splat(radius),
            };
            let omni = axis == Vec3::ZERO || light.direction_outer.w <= -0.999;

            light.shadow_first = base as u32;
            light.shadow_count = if omni { 6 } else { 1 };

            if omni {
                for (face, direction) in CUBE_DIRECTIONS.iter().enumerate() {
                    let transform = Self::cube_shadow_transform(position, *direction, radius);
                    self.set_local_shadow(
                        canvas,
                        views,
                        base + face,
                        (slot * LOCAL_SHADOW_FACES + face) as u32,
                        transform,
                        radius,
                    );
                    self.shadow_draws.push(ShadowDraw {
                        view: base + face,
                        resolution: self.settings.local_shadow_resolution,
                        bound,
                    });
                }
            } else {
                let transform =
                    Self::spot_shadow_transform(position, axis, light.direction_outer.w, radius);
                self.set_local_shadow(
                    canvas,
                    views,
                    base,
                    (slot * LOCAL_SHADOW_FACES) as u32,
                    transform,
                    radius,
                );
                self.shadow_draws.push(ShadowDraw {
                    view: base,
                    resolution: self.settings.local_shadow_resolution,
                    bound,
                });
            }
        }
    }

    fn set_local_shadow(
        &mut self,
        canvas: &Canvas,
        views: &mut [ShadowView],
        view_index: usize,
        layer: u32,
        transform: Mat4,
        range: f32,
    ) {
        canvas.queue.write_buffer(
            &self.shadow_passes[view_index].transform,
            0,
            bytemuck::cast_slice(&[transform]),
        );
        views[view_index] = ShadowView {
            matrix: transform,
            texel: 2.0 * range / self.settings.local_shadow_resolution as f32,
            range,
            extent: 0.0,
            scale: 1.0,
            atlas: 1,
            layer,
            bias_scale: 0.00025,
            normal_offset: 0.03,
        };
    }

    fn sun_center(&mut self, eye: Vec3, rotation: Vec3, texel: f32, index: usize) -> Vec3 {
        let center = *self.sun_centers[index].get_or_insert(eye);
        let view = Self::light_view(center, rotation);
        let local = view.transform_point3(eye);
        let stepped = Vec3::new(
            (local.x / texel).round() * texel,
            (local.y / texel).round() * texel,
            local.z,
        );
        let center = view.inverse().transform_point3(stepped);
        self.sun_centers[index] = Some(center);
        center
    }

    fn spot_shadow_transform(position: Vec3, axis: Vec3, outer_cos: f32, radius: f32) -> Mat4 {
        let fov =
            (2.0 * outer_cos.clamp(-1.0, 1.0).acos()).clamp(0.01, std::f32::consts::PI - 0.01);
        Self::oriented_shadow_transform(position, axis, radius, 0.0625, fov)
    }

    fn cube_shadow_transform(position: Vec3, axis: Vec3, radius: f32) -> Mat4 {
        Self::oriented_shadow_transform(
            position,
            axis,
            radius,
            0.0625,
            std::f32::consts::FRAC_PI_2 * 1.0625,
        )
    }

    fn oriented_shadow_transform(
        position: Vec3,
        axis: Vec3,
        radius: f32,
        near: f32,
        fov: f32,
    ) -> Mat4 {
        let up_hint = if axis.dot(Vec3::Y).abs() > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = axis.cross(up_hint).normalize_or_zero();
        let up = right.cross(axis).normalize_or_zero();
        let rotation = Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            (-axis).extend(0.0),
            Vec4::W,
        );
        let view = rotation.transpose() * Mat4::translation(-position.x, -position.y, -position.z);
        let projection = Mat4::projection(near, radius, fov, 1.0);
        projection * view
    }

    fn fit(&self, eye: Vec3, rotation: Vec3, extent: f32) -> Mat4 {
        Mat4::orthographic(
            -extent,
            extent,
            -extent,
            extent,
            self.settings.directional_light_distance - self.settings.shadow_margin,
            self.settings.directional_light_distance + self.settings.shadow_margin,
        ) * Self::light_view(eye, rotation)
    }

    fn light_view(eye: Vec3, rotation: Vec3) -> Mat4 {
        Mat4::rotation(rotation.x, rotation.y, rotation.z).transpose()
            * Mat4::translation(-eye.x, -eye.y, -eye.z)
    }

    pub(crate) fn shadow_draws(&self) -> &[ShadowDraw] {
        &self.shadow_draws
    }
}

/// The bind group shared by block and actor rendering.
pub(crate) struct WorldLightsLayout;

impl BindSignature for WorldLightsLayout {
    const NAME: &'static str = "lights";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[
        storage_entry(0, ShaderStages::VERTEX_FRAGMENT, true),
        storage_entry(1, ShaderStages::FRAGMENT, true),
        uniform_entry(2, ShaderStages::VERTEX_FRAGMENT),
        depth_entry(3),
        depth_entry(4),
        BindGroupLayoutEntry {
            binding: 5,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Comparison),
            count: None,
        },
        storage_entry(6, ShaderStages::FRAGMENT, true),
        storage_entry(7, ShaderStages::FRAGMENT, true),
        float_texture_entry(8),
        float_texture_entry(9),
        float_texture_entry(10),
        float_texture_entry(11),
        BindGroupLayoutEntry {
            binding: 12,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
    ];
    type Content<'a> = WorldLightsContent<'a>;
}

const fn storage_entry(
    binding: u32,
    visibility: ShaderStages,
    read_only: bool,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn uniform_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn depth_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

const fn float_texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

pub(crate) struct WorldLightsContent<'a> {
    lights: &'a Buffer,
    views: &'a Buffer,
    params: &'a Buffer,
    sun_shadow: &'a TextureView,
    local_shadow: &'a TextureView,
    shadow_sampler: &'a Sampler,
    cluster_counts: &'a Buffer,
    cluster_lights: &'a Buffer,
    sun_transmission_accum: &'a TextureView,
    sun_transmission_reveal: &'a TextureView,
    local_transmission_accum: &'a TextureView,
    local_transmission_reveal: &'a TextureView,
    transmission_sampler: &'a Sampler,
}

impl<'a> BindContent<'a> for WorldLightsContent<'a> {
    fn to_bindings(&self) -> Vec<BindGroupEntry<'a>> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.lights.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.views.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(self.sun_shadow),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(self.local_shadow),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::Sampler(self.shadow_sampler),
            },
            BindGroupEntry {
                binding: 6,
                resource: self.cluster_counts.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: self.cluster_lights.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::TextureView(self.sun_transmission_accum),
            },
            BindGroupEntry {
                binding: 9,
                resource: BindingResource::TextureView(self.sun_transmission_reveal),
            },
            BindGroupEntry {
                binding: 10,
                resource: BindingResource::TextureView(self.local_transmission_accum),
            },
            BindGroupEntry {
                binding: 11,
                resource: BindingResource::TextureView(self.local_transmission_reveal),
            },
            BindGroupEntry {
                binding: 12,
                resource: BindingResource::Sampler(self.transmission_sampler),
            },
        ]
    }
}

struct LightClusterLayout;

impl BindSignature for LightClusterLayout {
    const NAME: &'static str = "light_cluster";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[
        uniform_entry(0, ShaderStages::COMPUTE),
        storage_entry(1, ShaderStages::COMPUTE, true),
        uniform_entry(2, ShaderStages::COMPUTE),
        storage_entry(3, ShaderStages::COMPUTE, false),
        storage_entry(4, ShaderStages::COMPUTE, false),
    ];
    type Content<'a> = (&'a Buffer, &'a Buffer, &'a Buffer, &'a Buffer, &'a Buffer);
}

/// Moves the sky lights to the current tick.
pub(super) struct SkyUpdater;

impl System<1> for SkyUpdater {
    type CompQueries = ((),);
    type ResQuery = (ResWrite<Sky>, ResRead<TotalTicks>);

    fn update(
        &mut self,
        _: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        let (sky, ticks) = res;
        sky.update(ticks.0);
    }
}

/// Builds all frame sources and dispatches the local-light cluster.
pub(crate) struct LightingUpdater;

impl System<2> for LightingUpdater {
    type CompQueries = (
        (CompRead<Emission>, CompRead<Position>, CompRead<Rotation>),
        (),
    );
    type ResQuery = (
        ResRead<Canvas>,
        ResRead<Camera>,
        ResRead<World>,
        ResRead<Sky>,
        ResWrite<Lighting>,
        ResRead<PartialTick>,
        ResWrite<FrameSlot>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, camera, world, sky, lighting, partial, frame) = res;
        let Some(frame) = frame.as_mut() else {
            return;
        };
        let mut local = world
            .light_sources()
            .map(|(pos, emission)| Light::from_block(pos, emission))
            .collect::<Vec<_>>();
        for (_, emission, position, rotation) in comp.0.iter() {
            local.push(Light::from_entity(
                emission,
                position.interpolated(partial.0),
                rotation,
            ));
        }
        lighting.prepare(canvas, camera, sky, partial.0, local, frame);
    }
}

/// Draws every shadow view prepared by [`Lighting`].
pub(crate) struct ShadowRenderer {
    depth_batch: RenderBatch<
        (TextureArraySampler, ShadowTransform, BlockTemplates),
        (),
        BlockRectInst,
        (),
        DepthDraw,
    >,
    alpha_depth_batch: RenderBatch<
        (BlockTextureSet, ShadowTransform, BlockTemplates),
        (),
        BlockRectInst,
        ColorDraw<2>,
        DepthDraw,
    >,
    transmission_batch: RenderBatch<
        (BlockTextureSet, ShadowTransform, BlockTemplates),
        (),
        BlockRectInst,
        ColorDraw<2>,
        DepthDraw,
    >,
}

impl System<2> for ShadowRenderer {
    type CompQueries = ((), ());
    type ResQuery = (ResWrite<FrameSlot>, ResRead<World>, ResRead<Lighting>);

    fn update(
        &mut self,
        _: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (frame, world, lighting) = res;
        let Some(frame) = frame.as_mut() else {
            return;
        };

        for draw in lighting.shadow_draws() {
            let Some(view) = lighting.shadow_passes.get(draw.view) else {
                continue;
            };
            let target = DepthTarget {
                view: &view.view,
                load: LoadOp::Clear(0.0),
            };

            frame.render::<(), DepthDraw>("shadow", (), target, |mut pass| {
                pass.set_viewport(
                    0.0,
                    0.0,
                    draw.resolution as f32,
                    draw.resolution as f32,
                    0.0,
                    1.0,
                );
                self.depth_batch.begin(&mut pass);
                self.depth_batch
                    .push(&mut pass, (&BLOCK_TEXTURES, &view.pass, &BLOCK_TEMPLATES));

                for region in world.regions.volume.vec.iter().filter_map(|r| r.as_ref()) {
                    if draw.bound.intersects_with(region.bound())
                        && let Some(rects) = region.model().blocks()
                    {
                        self.depth_batch.draw(&mut pass, &rects);
                    }
                }
            });

            let alpha_colors = [
                ColorTarget {
                    view: &view.accum,
                    load: LoadOp::Clear(Color::TRANSPARENT),
                },
                ColorTarget {
                    view: &view.reveal,
                    load: LoadOp::Clear(Color {
                        r: 1.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                },
            ];
            let alpha_depth = DepthTarget {
                view: &view.view,
                load: LoadOp::Load,
            };
            frame.render::<ColorDraw<2>, DepthDraw>(
                "shadow_alpha_depth",
                alpha_colors,
                alpha_depth,
                |mut pass| {
                    pass.set_viewport(
                        0.0,
                        0.0,
                        draw.resolution as f32,
                        draw.resolution as f32,
                        0.0,
                        1.0,
                    );
                    self.alpha_depth_batch.begin(&mut pass);
                    self.alpha_depth_batch
                        .push(&mut pass, (&BLOCK_SURFACES, &view.pass, &BLOCK_TEMPLATES));
                    for region in world.regions.volume.vec.iter().filter_map(|r| r.as_ref()) {
                        if draw.bound.intersects_with(region.bound())
                            && let Some(rects) = region.model().optical()
                        {
                            self.alpha_depth_batch.draw(&mut pass, &rects);
                        }
                    }
                },
            );

            let transmission_colors = [
                ColorTarget {
                    view: &view.accum,
                    load: LoadOp::Load,
                },
                ColorTarget {
                    view: &view.reveal,
                    load: LoadOp::Load,
                },
            ];
            let transmission_depth = DepthTarget {
                view: &view.view,
                load: LoadOp::Load,
            };
            frame.render::<ColorDraw<2>, DepthDraw>(
                "shadow_transmission",
                transmission_colors,
                transmission_depth,
                |mut pass| {
                    pass.set_viewport(
                        0.0,
                        0.0,
                        draw.resolution as f32,
                        draw.resolution as f32,
                        0.0,
                        1.0,
                    );
                    self.transmission_batch.begin(&mut pass);
                    self.transmission_batch
                        .push(&mut pass, (&BLOCK_SURFACES, &view.pass, &BLOCK_TEMPLATES));
                    for region in world.regions.volume.vec.iter().filter_map(|r| r.as_ref()) {
                        if draw.bound.intersects_with(region.bound())
                            && let Some(rects) = region.model().optical()
                        {
                            self.transmission_batch.draw(&mut pass, &rects);
                        }
                    }
                },
            );
        }
    }
}

impl ShadowRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            depth_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "shadow",
                    shader: "depth",
                    color_formats: (),
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: None,
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            alpha_depth_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "shadow_alpha_depth",
                    shader: "shadow_alpha_depth",
                    color_formats: [TextureFormat::Rgba16Float, TextureFormat::R16Float],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Front),
                    blend: None,
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            transmission_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "shadow_transmission",
                    shader: "shadow_transmission",
                    color_formats: [TextureFormat::Rgba16Float, TextureFormat::R16Float],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Front),
                    blend: Some(BlendState::ADDITIVE),
                    secondary_blend: Some(WBOIT_REVEAL_BLEND),
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: false,
                },
            ),
        }
    }
}
