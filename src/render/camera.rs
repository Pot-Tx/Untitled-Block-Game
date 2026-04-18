use crate::actor::{PlayerControlled, Position, Rotation, Velocity};
use crate::ecs::*;
use crate::render::*;
use crate::util::bounding::Plane;
use crate::util::transform::Trans4;
use glam::{f32, Mat3, Mat4, Vec3};
use std::f32::consts::FRAC_PI_2;
use wgpu::*;

pub struct Camera {
    near: f32,
    far: f32,
    fov: f32,

    pub buffer: Buffer,
    pub transform: BindSet<Transformation>,
    pub frustum: [Plane<Vec3>; 5],
}

impl Resource for Camera {}

impl Camera {
    pub fn new(canvas: &Canvas) -> Self {
        let buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: "camera",
                init: BufferInit::Content(&[Mat4::default()]),
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

            frustum: [Plane::default(); 5],
        }
    }

    pub fn transform(
        &mut self,
        pos: &Position,
        rot: &Rotation,
        vel: &Velocity,
        partial_tick: &PartialTick,
    ) {
        let pos = pos.0 - vel.0 * (1.0 - partial_tick.0);
        let rot = rot.0;
        let aspect;

        let trans = Mat4::translation(-pos[0], -pos[1], -pos[2]);
        let rot = Mat4::rotation(-rot[0], -rot[1], -rot[2]);

        {
            let canvas = CANVAS.read().unwrap();
            aspect = canvas.surface_config.width as f32 / canvas.surface_config.height as f32;
            let proj = Mat4::projection(self.near, self.far, self.fov, aspect);
            let mat = proj * rot * trans;

            let queue = &canvas.queue;
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[mat]));
        }

        let dy = self.far * (self.fov / 2.0).tan();
        let dx = dy * aspect;
        let rot = Mat3::from_mat4(rot).transpose();

        let bl = pos + rot * Vec3::new(dx, -dy, -self.far);
        let br = pos + rot * Vec3::new(-dx, -dy, -self.far);
        let tl = pos + rot * Vec3::new(dx, dy, -self.far);
        let tr = pos + rot * Vec3::new(-dx, dy, -self.far);
        let back = pos;
        let orient = pos + rot * Vec3::new(0.0, 0.0, -self.near);

        self.frustum = [
            Plane::from_points(back, bl, tl, orient),
            Plane::from_points(back, tr, br, orient),
            Plane::from_points(back, tl, tr, orient),
            Plane::from_points(back, br, bl, orient),
            Plane::from_points(bl, br, tl, orient),
        ];
    }
}

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

pub struct CameraTransformer;

impl System for CameraTransformer {
    type CompQuery = (
        CompRead<PlayerControlled>,
        CompRead<Position>,
        CompRead<Rotation>,
        CompRead<Velocity>,
    );
    type ResQuery = (ResWrite<Camera>, ResRead<PartialTick>);

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        res.0.transform(entry.2, entry.3, entry.4, res.1);

        None
    }
}
