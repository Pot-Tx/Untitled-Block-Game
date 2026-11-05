//! Conservative visibility for the geometry of the current world view.

use crate::ecs::*;
use crate::render::*;
use crate::util::bounding::{AABB, Plane, PlaneGroup};
use glam::{Mat4, Vec2, Vec3};
use log::{trace, warn};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use wgpu::*;

/// The largest edge of the pyramid that is copied back to the CPU.
///
/// This is an internal query budget. It is independent of the scene depth
/// resolution, which remains sized to the window.
const MAX_QUERY_EXTENT: u32 = 256;
/// Number of staging buffers that can be in flight at once.
const READBACK_COUNT: usize = 3;
/// The map callback writes this until the GPU work has completed.
const MAPPING_PENDING: u8 = 0;
const MAPPING_READY: u8 = 1;
const MAPPING_FAILED: u8 = 2;

/// The current frustum and the latest completed depth query.
pub(crate) struct Visibility {
    frustum: [Plane<Vec3>; 5],
    snapshot: Option<VisibilitySnapshot>,
    hi_z: HiZ,
}

impl Resource for Visibility {}

impl Visibility {
    pub(crate) fn new() -> Self {
        Self {
            frustum: [Plane::default(); 5],
            snapshot: None,
            hi_z: HiZ::new(),
        }
    }

    /// Rebuilds the view frustum from the camera of the current frame.
    pub(crate) fn update(&mut self, canvas: &Canvas, camera: &Camera) {
        let aspect = canvas.surface_config.width as f32 / canvas.surface_config.height as f32;
        let dy = camera.far * (camera.fov / 2.0).tan();
        let dx = dy * aspect;
        let eye = camera.eye;

        let bl = eye + camera.right * dx - camera.up * dy + camera.forward * camera.far;
        let br = eye - camera.right * dx - camera.up * dy + camera.forward * camera.far;
        let tl = eye + camera.right * dx + camera.up * dy + camera.forward * camera.far;
        let tr = eye - camera.right * dx + camera.up * dy + camera.forward * camera.far;
        let back = eye;
        let orient = eye + camera.forward * camera.near;

        self.frustum = [
            Plane::from_points(back, bl, tl, orient),
            Plane::from_points(back, tr, br, orient),
            Plane::from_points(back, tl, tr, orient),
            Plane::from_points(back, br, bl, orient),
            Plane::from_points(bl, br, tl, orient),
        ];
    }

    /// Tests the current frustum and the previous frame's depth pyramid.
    pub(crate) fn is_visible(&self, bound: AABB<Vec3>, geometry_revision: u64) -> bool {
        self.frustum.is_aabb_inside(bound)
            && !self
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.is_occluded(bound, geometry_revision))
    }

    /// Polls the completed Hi-Z readback without waiting for the GPU.
    pub(crate) fn prepare(&mut self, canvas: &Canvas) {
        if self.hi_z.ensure_size(canvas) {
            self.snapshot = None;
        }

        if let Some(snapshot) = self.hi_z.poll(canvas)
            && self
                .snapshot
                .as_ref()
                .is_none_or(|current| current.sequence < snapshot.sequence)
        {
            self.snapshot = Some(snapshot);
        }
    }

    /// Captures the opaque scene depth into the Hi-Z pyramid.
    pub(crate) fn capture(
        &mut self,
        canvas: &Canvas,
        frame: &mut Frame,
        camera: &Camera,
        geometry_revision: u64,
    ) {
        self.hi_z
            .capture(canvas, frame, camera.view_projection, geometry_revision);
    }
}

/// The GPU resources that produce and read back the depth pyramid.
struct HiZ {
    targets: Option<HiZTargets>,
    readbacks: Vec<Readback>,
    next_sequence: u64,
}

struct HiZTargets {
    viewport: [u32; 2],
    texture: Texture,
    levels: Vec<HiZLevel>,
    depth_source: BindSet<HiZDepthSource>,
    depth_batch: ComputeBatch<(HiZDepthSource, HiZStorage)>,
    mip_batch: ComputeBatch<(HiZTextureSource, HiZStorage)>,
}

struct HiZLevel {
    width: u32,
    height: u32,
    source: Option<BindSet<HiZTextureSource>>,
    storage: BindSet<HiZStorage>,
}

struct Readback {
    buffer: Buffer,
    state: Arc<AtomicU8>,
    in_flight: bool,
    sequence: u64,
    view_projection: Mat4,
    geometry_revision: u64,
    mips: Vec<ReadbackMip>,
}

#[derive(Clone)]
struct ReadbackMip {
    width: u32,
    height: u32,
    offset: u64,
    padded_row: u64,
}

struct HiZDepthSource;
struct HiZTextureSource;
struct HiZStorage;

impl BindSignature for HiZDepthSource {
    const NAME: &'static str = "hi_z_depth_source";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    type Content<'a> = &'a TextureView;
}

impl BindSignature for HiZTextureSource {
    const NAME: &'static str = "hi_z_texture_source";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    type Content<'a> = &'a TextureView;
}

impl BindSignature for HiZStorage {
    const NAME: &'static str = "hi_z_storage";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::R32Float,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }];
    type Content<'a> = &'a TextureView;
}

impl HiZ {
    fn new() -> Self {
        Self {
            targets: None,
            readbacks: Vec::new(),
            next_sequence: 0,
        }
    }

    /// Creates the pyramid and staging buffers for a new window size, or
    /// returns `true` when a resize discarded the previous generation.
    fn ensure_size(&mut self, canvas: &Canvas) -> bool {
        let viewport = [canvas.surface_config.width, canvas.surface_config.height];

        if self
            .targets
            .as_ref()
            .is_some_and(|targets| targets.viewport == viewport)
        {
            return false;
        }

        let targets = HiZTargets::new(canvas, viewport);
        let mut readbacks = Vec::with_capacity(READBACK_COUNT);

        for _ in 0..READBACK_COUNT {
            readbacks.push(Readback::new(canvas, &targets));
        }

        self.targets = Some(targets);
        self.readbacks = readbacks;
        true
    }

    /// Polls the GPU once and returns the newest completed pyramid.
    fn poll(&mut self, canvas: &Canvas) -> Option<VisibilitySnapshot> {
        let _ = canvas.device.poll(PollType::Poll);
        let mut newest = None;

        for readback in &mut self.readbacks {
            if !readback.in_flight {
                continue;
            }

            match readback.state.load(Ordering::Acquire) {
                MAPPING_PENDING => continue,

                MAPPING_FAILED => {
                    warn!("failed to map the Hi-Z readback buffer");
                    readback.in_flight = false;
                    readback.state.store(MAPPING_PENDING, Ordering::Release);
                }

                MAPPING_READY => {
                    let Ok(mapped) = readback.buffer.slice(..).get_mapped_range() else {
                        warn!("failed to read the mapped Hi-Z buffer");
                        readback.buffer.unmap();
                        readback.in_flight = false;
                        readback.state.store(MAPPING_PENDING, Ordering::Release);
                        continue;
                    };
                    let mips = readback
                        .mips
                        .iter()
                        .map(|mip| mip.decode(&mapped))
                        .collect::<Vec<_>>();
                    drop(mapped);

                    readback.buffer.unmap();
                    readback.in_flight = false;
                    readback.state.store(MAPPING_PENDING, Ordering::Release);
                    trace!("Hi-Z snapshot {} is ready", readback.sequence);

                    if newest
                        .as_ref()
                        .is_none_or(|(sequence, _)| *sequence < readback.sequence)
                    {
                        newest = Some((
                            readback.sequence,
                            VisibilitySnapshot {
                                sequence: readback.sequence,
                                mips,
                                view_projection: readback.view_projection,
                                geometry_revision: readback.geometry_revision,
                            },
                        ));
                    }
                }

                _ => unreachable!("unknown Hi-Z mapping state"),
            }
        }

        newest.map(|(_, snapshot)| snapshot)
    }

    /// Records the depth pyramid and schedules its readback after submission.
    fn capture(
        &mut self,
        canvas: &Canvas,
        frame: &mut Frame,
        view_projection: Mat4,
        geometry_revision: u64,
    ) {
        self.ensure_size(canvas);
        let Some(index) = self
            .readbacks
            .iter()
            .position(|readback| !readback.in_flight)
        else {
            return;
        };

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);

        let targets = self
            .targets
            .as_ref()
            .expect("Hi-Z targets were just ensured");

        frame.compute("hi_z_depth", |mut pass| {
            targets.depth_batch.begin(&mut pass);
            targets.depth_batch.push(
                &mut pass,
                (&targets.depth_source, &targets.levels[0].storage),
            );
            targets.depth_batch.dispatch(
                &mut pass,
                targets.levels[0].width.div_ceil(8),
                targets.levels[0].height.div_ceil(8),
                1,
            );
        });

        for level in targets.levels.iter().skip(1) {
            let Some(source) = &level.source else {
                continue;
            };

            frame.compute("hi_z_mip", |mut pass| {
                targets.mip_batch.begin(&mut pass);
                targets.mip_batch.push(&mut pass, (source, &level.storage));
                targets.mip_batch.dispatch(
                    &mut pass,
                    level.width.div_ceil(8),
                    level.height.div_ceil(8),
                    1,
                );
            });
        }

        let readback = &mut self.readbacks[index];
        readback.mips = targets
            .levels
            .iter()
            .scan(0, |offset, level| {
                let padded_row = align_to(
                    level.width as u64 * size_of::<f32>() as u64,
                    COPY_BYTES_PER_ROW_ALIGNMENT as u64,
                );
                let mip = ReadbackMip {
                    width: level.width,
                    height: level.height,
                    offset: *offset,
                    padded_row,
                };
                *offset += padded_row * level.height as u64;
                Some(mip)
            })
            .collect();

        for (level, mip) in readback.mips.iter().enumerate() {
            frame.0.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: &targets.texture,
                    mip_level: level as u32,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: &readback.buffer,
                    layout: TexelCopyBufferLayout {
                        offset: mip.offset,
                        bytes_per_row: Some(mip.padded_row as u32),
                        rows_per_image: Some(mip.height),
                    },
                },
                Extent3d {
                    width: mip.width,
                    height: mip.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        readback.sequence = sequence;
        readback.view_projection = view_projection;
        readback.geometry_revision = geometry_revision;
        readback.in_flight = true;
        readback.state.store(MAPPING_PENDING, Ordering::Release);

        let state = readback.state.clone();
        frame
            .0
            .map_buffer_on_submit(&readback.buffer, MapMode::Read, .., move |result| {
                let mapped_state = if result.is_ok() {
                    MAPPING_READY
                } else {
                    MAPPING_FAILED
                };
                state.store(mapped_state, Ordering::Release);
            });
    }
}

impl Readback {
    fn new(canvas: &Canvas, targets: &HiZTargets) -> Self {
        let size = targets
            .levels
            .iter()
            .map(|level| {
                align_to(
                    level.width as u64 * size_of::<f32>() as u64,
                    COPY_BYTES_PER_ROW_ALIGNMENT as u64,
                ) * level.height as u64
            })
            .sum();

        let buffer = Buffer::new(
            canvas,
            &BufferConfig::<u8> {
                name: "hi_z_readback",
                init: BufferInit::Size(size),
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            },
        );

        Self {
            buffer,
            state: Arc::new(AtomicU8::new(MAPPING_PENDING)),
            in_flight: false,
            sequence: 0,
            view_projection: Mat4::IDENTITY,
            geometry_revision: 0,
            mips: Vec::new(),
        }
    }
}

impl ReadbackMip {
    fn decode(&self, mapped: &[u8]) -> HiZMip {
        let mut depths = Vec::with_capacity((self.width * self.height) as usize);
        let row_len = self.width as u64 * size_of::<f32>() as u64;

        for y in 0..self.height {
            let start = (self.offset + y as u64 * self.padded_row) as usize;
            let row = &mapped[start..start + row_len as usize];
            depths.extend(
                row.chunks_exact(size_of::<f32>())
                    .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap())),
            );
        }

        HiZMip {
            width: self.width,
            height: self.height,
            depths,
        }
    }
}

impl HiZTargets {
    fn new(canvas: &Canvas, viewport: [u32; 2]) -> Self {
        let scale = (MAX_QUERY_EXTENT as f32 / viewport[0].max(viewport[1]) as f32).min(1.0);
        let width = ((viewport[0] as f32 * scale).ceil() as u32).max(1);
        let height = ((viewport[1] as f32 * scale).ceil() as u32).max(1);
        let mip_level_count = width.max(height).ilog2() + 1;

        let texture = Texture::new(
            canvas,
            &TextureConfig {
                name: "hi_z",
                texs: vec![None],
                width,
                height,
                mip_level_count,
                format: TextureFormat::R32Float,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_SRC,
            },
        );

        let depth_batch = ComputeBatch::new(
            canvas,
            &ComputeBatchConfig {
                name: "hi_z_depth",
                shader: "hi_z_depth",
            },
        );
        let mip_batch = ComputeBatch::new(
            canvas,
            &ComputeBatchConfig {
                name: "hi_z_mip",
                shader: "hi_z_mip",
            },
        );

        let depth_source = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "hi_z_depth_source",
                content: &canvas.depth_view,
            },
        );

        let mut levels = Vec::with_capacity(mip_level_count as usize);
        for level in 0..mip_level_count {
            let level_width = (width >> level).max(1);
            let level_height = (height >> level).max(1);

            let storage_view = TextureView::new(
                &texture,
                &TextureViewConfig {
                    name: "hi_z_storage",
                    dimension: TextureViewDimension::D2,
                    mip_level: Some(level),
                },
            );
            let storage = BindSet::new(
                canvas,
                &BindSetConfig {
                    name: "hi_z_storage",
                    content: &storage_view,
                },
            );

            let source = (level > 0).then(|| {
                let source_view = TextureView::new(
                    &texture,
                    &TextureViewConfig {
                        name: "hi_z_source",
                        dimension: TextureViewDimension::D2,
                        mip_level: Some(level - 1),
                    },
                );
                BindSet::new(
                    canvas,
                    &BindSetConfig {
                        name: "hi_z_source",
                        content: &source_view,
                    },
                )
            });

            levels.push(HiZLevel {
                width: level_width,
                height: level_height,
                source,
                storage,
            });
        }

        Self {
            viewport,
            texture,
            levels,
            depth_source,
            depth_batch,
            mip_batch,
        }
    }
}

/// One level of the conservative depth pyramid, in reverse depth order.
struct HiZMip {
    width: u32,
    height: u32,
    depths: Vec<f32>,
}

/// The depth pyramid and camera transform of one completed readback.
struct VisibilitySnapshot {
    sequence: u64,
    mips: Vec<HiZMip>,
    view_projection: Mat4,
    geometry_revision: u64,
}

/// The screen-space extent and nearest depth of a box in the captured view.
#[derive(Clone, Copy)]
struct ProjectedBounds {
    min: Vec2,
    max: Vec2,
    nearest_depth: f32,
}

impl VisibilitySnapshot {
    /// Tests whether every covered depth texel is closer to the camera than the
    /// nearest point of `bound`.
    ///
    /// The scene uses reverse depth, where a larger depth is closer. A pyramid
    /// texel therefore stores the smallest depth of its footprint, and the box
    /// is occluded only when the whole footprint is closer than its nearest
    /// point.
    fn is_occluded(&self, bound: AABB<Vec3>, geometry_revision: u64) -> bool {
        if self.geometry_revision != geometry_revision {
            return false;
        }

        let Some(rect) = self.project(bound) else {
            return false;
        };
        let depth = self.min_depth(rect);

        depth > rect.nearest_depth + 1.0e-5
    }

    /// Projects the corners into the viewport captured with the pyramid.
    ///
    /// A box that crosses the camera plane or leaves the captured viewport has
    /// unknown occlusion, so it is never reported as occluded.
    fn project(&self, bound: AABB<Vec3>) -> Option<ProjectedBounds> {
        let corners = [
            Vec3::new(bound.min.x, bound.min.y, bound.min.z),
            Vec3::new(bound.min.x, bound.min.y, bound.max.z),
            Vec3::new(bound.min.x, bound.max.y, bound.min.z),
            Vec3::new(bound.min.x, bound.max.y, bound.max.z),
            Vec3::new(bound.max.x, bound.min.y, bound.min.z),
            Vec3::new(bound.max.x, bound.min.y, bound.max.z),
            Vec3::new(bound.max.x, bound.max.y, bound.min.z),
            Vec3::new(bound.max.x, bound.max.y, bound.max.z),
        ];

        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut nearest_depth = f32::NEG_INFINITY;

        for corner in corners {
            let clip = self.view_projection * corner.extend(1.0);

            if !clip.is_finite() || clip.w <= 1.0e-4 {
                return None;
            }

            let ndc = clip.truncate() / clip.w;
            if !ndc.is_finite() || !(0.0..=1.0).contains(&ndc.z) {
                return None;
            }

            let screen = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

            min = min.min(screen);
            max = max.max(screen);
            nearest_depth = nearest_depth.max(ndc.z);
        }

        // A partially visible box could be occluded only outside the captured
        // viewport, where this frame has no depth information.
        if min.x < 0.0 || min.y < 0.0 || max.x > 1.0 || max.y > 1.0 {
            return None;
        }

        (min.x < max.x && min.y < max.y).then_some(ProjectedBounds {
            min,
            max,
            nearest_depth,
        })
    }

    /// Returns the smallest depth of the mip texels that cover `rect`.
    fn min_depth(&self, rect: ProjectedBounds) -> f32 {
        let Some(base) = self.mips.first() else {
            return 0.0;
        };

        let size = Vec2::new(base.width as f32, base.height as f32);
        let span = ((rect.max - rect.min) * size).max_element().max(1.0);
        let level = (span.log2().ceil() as usize).min(self.mips.len() - 1);
        let mip = &self.mips[level];

        let x0 = (rect.min.x * mip.width as f32)
            .floor()
            .clamp(0.0, (mip.width - 1) as f32) as u32;
        let y0 = (rect.min.y * mip.height as f32)
            .floor()
            .clamp(0.0, (mip.height - 1) as f32) as u32;
        let x1 = (rect.max.x * mip.width as f32)
            .ceil()
            .clamp((x0 + 1) as f32, mip.width as f32) as u32;
        let y1 = (rect.max.y * mip.height as f32)
            .ceil()
            .clamp((y0 + 1) as f32, mip.height as f32) as u32;

        let mut depth = 1.0f32;
        for y in y0..y1 {
            for x in x0..x1 {
                depth = depth.min(mip.depths[(y * mip.width + x) as usize]);
            }
        }
        depth
    }
}

/// Polls the depth pyramid after the camera transform and before the world is
/// drawn with the resulting visibility.
pub(crate) struct VisibilityPreparer;

impl System<2> for VisibilityPreparer {
    type CompQueries = ((), ());
    type ResQuery = (ResRead<Canvas>, ResWrite<Visibility>);

    fn update(
        &mut self,
        _: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _: [Commander<'_>; 2],
    ) {
        let (canvas, visibility) = res;
        visibility.prepare(canvas);
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}
