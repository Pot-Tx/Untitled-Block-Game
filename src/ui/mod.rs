use crate::ecs::{Command, CompQuery, ResQuery, ResRead, ResWrite, System};
use crate::render::{
    BindSet, Canvas, Frame, FromConfig, Geometry, Inst, InstGroup, Instances, Mesh, Render,
    RenderBatch, RenderBatchConfig, RenderDescriptor, RenderItem, Tex, TextureSampler, Transformation,
    Vertex, ViewPort, QUAD_INDICES,
};
use crate::util::OnceInit;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::fs::File;
use wgpu::{
    BufferAddress, LoadOp, PrimitiveTopology, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexStepMode,
};

pub static SPRITE_GEOMETRY: OnceInit<Geometry<()>> = OnceInit::new();

pub fn create_sprite_geometry(canvas: &Canvas) -> Geometry<()> {
    Mesh {
        vertices: Vec::new(),
        indices: Vec::from(QUAD_INDICES),
    }
    .geometry(canvas, "sprite")
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Sprite {
    pub pos: Vec3,
    pub rot: f32,
    pub width: f32,
    pub height: f32,
    pub min_uv: Vec2,
    pub max_uv: Vec2,
}

impl Inst for Sprite {
    const EMPTY: bool = false;

    fn layout<'a, V: Vertex>() -> Option<VertexBufferLayout<'a>> {
        Some(VertexBufferLayout {
            array_stride: size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: V::ATTRIBUTE_COUNT,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: size_of::<Vec3>() as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 1,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: (size_of::<Vec3>() + size_of::<f32>()) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 2,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: (size_of::<Vec3>() + size_of::<f32>() * 2) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 3,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: (size_of::<Vec3>() + size_of::<f32>() * 3) as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 4,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: (size_of::<Vec3>() + size_of::<f32>() * 3 + size_of::<Vec2>())
                        as BufferAddress,
                    shader_location: V::ATTRIBUTE_COUNT + 5,
                    format: VertexFormat::Float32x2,
                },
            ],
        })
    }
}

pub struct CrosshairRenderer {
    desc: RenderDescriptor<'static>,
    batch: RenderBatch<(Transformation, TextureSampler), (), Sprite>,
    crosshair: BindSet<TextureSampler>,
    instance: Instances<Sprite>,
}

impl System for CrosshairRenderer {
    type CompQuery = ();
    type ResQuery = (ResRead<Canvas>, ResWrite<Option<Frame>>, ResRead<ViewPort>);

    fn operate(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if let Some(frame) = res.1 {
            frame.render(&self.desc, |mut pass| {
                self.batch.begin(&mut pass);
                self.batch
                    .push(&mut pass, (&res.2.transform, &self.crosshair));
                self.batch.draw(&mut pass, self);
            });
        }

        None
    }
}

impl Render<(), Sprite> for CrosshairRenderer {
    fn rendered(&self) -> Vec<RenderItem<'_, (), Sprite>> {
        vec![RenderItem {
            geometry: &SPRITE_GEOMETRY,
            instances: &self.instance,
        }]
    }
}

impl CrosshairRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        let crosshair = Tex::from_png(
            File::open("assets/textures/crosshair.png").expect("failed to load crosshair texture"),
        )
        .expect("failed to load crosshair texture")
        .create_texture_sampler(canvas, "crosshair");
        let sprite = Sprite {
            pos: Vec3::new(-8.0, -8.0, 0.5),
            rot: 0.0,
            width: 16.0,
            height: 16.0,
            min_uv: Vec2::ZERO,
            max_uv: Vec2::ONE,
        };

        Self {
            desc: RenderDescriptor {
                name: "crosshair",
                color_load: LoadOp::Load,
                depth_load: LoadOp::Clear(0.0),
            },
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "crosshair",
                    shader: "crosshair",
                    translucent: true,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            crosshair,
            instance: [sprite].instances(canvas, "crosshair"),
        }
    }
}
