use crate::actor::{Eye, PlayerControlled, Position, Rotation};
use crate::ecs::*;
use crate::render::*;
use crate::util::transform::Trans4;
use glam::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::FRAC_PI_2;
use wgpu::*;

/// The camera of the world pass, with the buffer its transform is uploaded to.
pub struct Camera {
    /// Distances of the near and the far plane of the projection.
    pub(crate) near: f32,
    pub(crate) far: f32,
    /// Vertical field of view, in radians.
    pub(crate) fov: f32,

    /// The transform of the camera and the position it stands at.
    pub buffer: Buffer,
    pub transform: BindSet<CameraTransform>,
    /// The matrix that projects world positions into the scene.
    pub(crate) view_projection: Mat4,
    /// The position the camera stands at and the direction it looks in, which
    /// the passes that are not built from its frustum need, such as the ones
    /// that fit the cascades of the shadow map around the camera.
    pub eye: Vec3,
    pub forward: Vec3,
    pub(crate) right: Vec3,
    pub(crate) up: Vec3,
}

impl Resource for Camera {}

impl Camera {
    /// Creates the uniform buffer and the bind group of the camera transform.
    pub fn new(canvas: &Canvas) -> Self {
        let buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: "camera",
                init: BufferInit::Content(&[CameraTransform::default()]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        );
        let transform = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "camera",
                content: &buffer,
            },
        );
        Self {
            near: 0.125,
            far: 4096.0,
            fov: FRAC_PI_2,

            buffer,
            transform,
            view_projection: Mat4::IDENTITY,

            eye: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            right: Vec3::X,
            up: Vec3::Y,
        }
    }

    /// Points the camera at `eye` and records the transform of the frame.
    ///
    /// The position is interpolated between the last two ticks by the caller,
    /// so that the camera follows the player smoothly between ticks.
    pub fn transform(&mut self, canvas: &Canvas, eye: Vec3, rot: &Rotation) {
        let rot = rot.0;
        let aspect = canvas.surface_config.width as f32 / canvas.surface_config.height as f32;

        let trans = Mat4::translation(-eye[0], -eye[1], -eye[2]);
        let rot = Mat4::rotation(-rot[0], -rot[1], -rot[2]);
        let proj = Mat4::projection(self.near, self.far, self.fov, aspect);

        let mat = proj * rot * trans;
        self.view_projection = mat;
        let queue = &canvas.queue;
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[CameraTransform {
                matrix: mat,
                position: eye.extend(1.0),
            }]),
        );

        let rot = Mat3::from_mat4(rot).transpose();
        self.eye = eye;
        self.right = rot * Vec3::X;
        self.up = rot * Vec3::Y;
        self.forward = rot * Vec3::new(0.0, 0.0, -1.0);
    }
}

/// A square area of the window that UI space is mapped onto.
pub struct ViewPort {
    /// Size of the UI space of this viewport, in UI units.
    pub width: f32,
    pub height: f32,
    alignment: ViewPortAlignment,
    matrix: Mat4,
    /// Position of the cursor inside the UI space, or `None` when the cursor is
    /// grabbed or outside this viewport.
    pub cursor: Option<Vec2>,

    pub buffer: Buffer,
    pub transform: BindSet<Transformation>,
}

/// The edge of the window that a viewport is anchored to.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ViewPortAlignment {
    Middle,
    Left,
    Right,
    Bottom,
    Up,
}

impl ViewPortAlignment {
    /// Every alignment, ordered so that [`Self::idx`] indexes into this slice.
    pub const ALL: &'static [Self; 5] = &[
        Self::Middle,
        Self::Left,
        Self::Right,
        Self::Bottom,
        Self::Up,
    ];

    pub const fn idx(&self) -> usize {
        match self {
            Self::Middle => 0,
            Self::Left => 1,
            Self::Right => 2,
            Self::Bottom => 3,
            Self::Up => 4,
        }
    }
}

impl Resource for ViewPort {}

/// The viewport of every alignment, in the order of [`ViewPortAlignment::ALL`].
pub struct Viewports(Vec<ViewPort>);

impl Resource for Viewports {}

impl Viewports {
    /// Creates one viewport per alignment, all of them `width` by `height` UI
    /// units.
    pub fn new(canvas: &Canvas, width: f32, height: f32) -> Self {
        Self(
            ViewPortAlignment::ALL
                .iter()
                .map(|&alignment| ViewPort::new(canvas, width, height, alignment))
                .collect(),
        )
    }

    /// The viewport anchored to `alignment`.
    pub fn get(&self, alignment: ViewPortAlignment) -> &ViewPort {
        &self.0[alignment.idx()]
    }

    /// The cursor position inside the viewport of `alignment`.
    pub fn cursor(&self, alignment: ViewPortAlignment) -> Option<Vec2> {
        self.get(alignment).cursor
    }

    /// Rebuilds the transform of every viewport.
    pub fn transform(&mut self, canvas: &Canvas) {
        self.0
            .iter_mut()
            .for_each(|viewport| viewport.transform(canvas));
    }

    /// Maps the window cursor position into every viewport, or clears the cursor
    /// position of all of them while the cursor is grabbed.
    pub fn track_cursor(&mut self, canvas: &Canvas, grabbed: bool, pos: Vec2) {
        self.0.iter_mut().for_each(|viewport| {
            viewport.cursor = (!grabbed).then(|| viewport.cast_pos(canvas, pos)).flatten();
        });
    }
}

impl ViewPort {
    /// Creates the uniform buffer and the bind group of a viewport transform.
    pub fn new(canvas: &Canvas, width: f32, height: f32, alignment: ViewPortAlignment) -> Self {
        let buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: "viewport",
                init: BufferInit::Content(&[Mat4::default()]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        );
        let transform = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "viewport",
                content: &buffer,
            },
        );

        Self {
            width,
            height,
            alignment,
            matrix: Mat4::default(),
            cursor: None,

            buffer,
            transform,
        }
    }

    /// Rebuilds the matrix that maps UI space onto the window.
    pub fn transform(&mut self, canvas: &Canvas) {
        let aspect = canvas.surface_config.width as f32 / canvas.surface_config.height as f32;
        let mat = Mat4::viewport(self.width, self.height, self.alignment, aspect);
        self.matrix = mat;
        let queue = &canvas.queue;
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[mat]));
    }

    /// Converts window pixels, with a top-left origin, into this viewport's UI
    /// space, or returns `None` when the position is outside the viewport.
    pub fn cast_pos(&self, canvas: &Canvas, pos: Vec2) -> Option<Vec2> {
        let window = Vec2::new(
            canvas.surface_config.width as f32,
            canvas.surface_config.height as f32,
        );
        let ndc = pos / window * Vec2::new(2.0, -2.0) + Vec2::new(-1.0, 1.0);
        let view = (self.matrix.inverse() * ndc.extend(0.0).extend(1.0)).truncate();

        ((0.0..=self.width).contains(&view.x) && (0.0..=self.height).contains(&view.y))
            .then(|| Vec2::new(view.x, view.y))
    }
}

/// The bind group of a 4x4 transform matrix, which the viewports are mapped
/// onto the window with.
pub struct Transformation;

impl BindSignature for Transformation {
    const NAME: &'static str = "transform";
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

/// The bind group of the world camera: the matrix that projects the world and
/// the position it is seen from.
///
/// The sprites of the world turn towards that position, so that an actor is
/// drawn from the side the camera is on, and the passes that light the world
/// read it to tell how far a fragment is from the camera.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct CameraTransform {
    pub matrix: Mat4,
    pub position: Vec4,
}

impl BindSignature for CameraTransform {
    const NAME: &'static str = "camera";
    const LAYOUTS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    type Content<'a> = &'a Buffer;
}

/// Uploads the transform of the player camera every frame.
pub(super) struct CameraTransformer;

impl System<2> for CameraTransformer {
    type CompQueries = (
        (
            CompRead<PlayerControlled>,
            CompRead<Position>,
            CompRead<Rotation>,
            OptionalRead<Eye>,
        ),
        (),
    );
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<Camera>,
        ResWrite<Visibility>,
        ResRead<PartialTick>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, camera, visibility, partial) = res;

        for entry in comp.0.iter() {
            // The camera stands where the player looks from, which is its
            // position raised to its eyes.
            let eye = entry.2.interpolated(partial.0) + Vec3::Y * entry.4.map_or(0.0, |eye| eye.0);

            camera.transform(canvas, eye, entry.3);
            visibility.update(canvas, camera);
        }
    }
}
