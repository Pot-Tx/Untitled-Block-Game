//! The systems that draw the world: the opaque pass, the shadows and the
//! occlusion derived from them, and the translucent pass with its resolve.

use crate::ecs::*;
use crate::render::*;
use crate::world::*;
use wgpu::*;

/// Registers the passes of the world, in the order they have to run in.
pub(super) fn register_canvas(frame: &mut SystemManager<2>, canvas: &Canvas) {
    // Every light and shadow view is prepared before any pass that reads them.
    frame.register(3, "lighting", LightingUpdater);

    // All prepared shadow views are drawn before the passes that read them.
    frame.register(4, "shadow", ShadowRenderer::new(canvas));

    frame.register(5, "world_renderer", WorldRenderer::new(canvas));

    // The occlusion shades the faces of the blocks the pass above drew.
    frame.register(6, "occlusion", OcclusionRenderer::new(canvas));

    frame.register(7, "hi_z_builder", HiZBuilder);

    frame.register(9, "translucent", TranslucentRenderer::new(canvas));
    frame.register(10, "wboit_resolve", WboitResolver);
}

/// Captures the opaque terrain after the world passes and before actors, so
/// that dynamic objects do not become stale occluders.
struct HiZBuilder;

impl System<2> for HiZBuilder {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (
        ResRead<Canvas>,
        ResRead<Camera>,
        ResRead<World>,
        ResWrite<Visibility>,
        ResWrite<FrameSlot>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _: [Commander<'_>; 2],
    ) {
        if comp.0.iter().next().is_none() {
            return;
        }

        let (canvas, camera, world, visibility, frame) = res;
        if let Some(frame) = frame.as_mut() {
            visibility.capture(canvas, frame, camera, world.geometry_revision);
        }
    }
}

struct WorldRenderer {
    opaque_batch: RenderBatch<
        (
            BlockTextureSet,
            CameraTransform,
            WorldLightsLayout,
            BlockTemplates,
        ),
        (),
        BlockRectInst,
        ColorDraw<1>,
        DepthDraw,
    >,
}

impl System<2> for WorldRenderer {
    type CompQueries = (
        (
            CompRead<PlayerControlled>,
            CompRead<Position>,
            CompRead<Rotation>,
        ),
        (),
    );
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<FrameSlot>,
        ResRead<Camera>,
        ResWrite<World>,
        ResRead<Lighting>,
        ResRead<Visibility>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, frame, camera, world, lights, visibility) = res;

        for _ in comp.0.iter() {
            world.pre_render(canvas);

            if let Some(frame) = frame.as_mut() {
                world.cull(visibility);

                let colors = [ColorTarget {
                    view: &canvas.color_view,
                    load: LoadOp::Clear(Color {
                        r: 0.375,
                        g: 0.625,
                        b: 1.0,
                        a: 1.0,
                    }),
                }];
                let depth = DepthTarget {
                    view: &canvas.depth_view,
                    load: LoadOp::Clear(0.0),
                };

                frame.render::<ColorDraw<1>, DepthDraw>("block", colors, depth, |mut pass| {
                    self.opaque_batch.begin(&mut pass);
                    self.opaque_batch.push(
                        &mut pass,
                        (
                            &BLOCK_SURFACES,
                            &camera.transform,
                            &lights.bind_group,
                            &BLOCK_TEMPLATES,
                        ),
                    );

                    for region in world.visible_regions() {
                        if let Some(rects) = region.model().blocks() {
                            self.opaque_batch.draw(&mut pass, &rects);
                        }
                    }
                });
            }
        }
    }
}

impl WorldRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            opaque_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "block",
                    // `block` draws the same geometry without any lighting.
                    shader: "block_lighting",
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: None,
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
        }
    }
}

/// Shades the corners of the blocks that the light of the sky does not reach.
///
/// The geometry is the faces of the blocks a second time, so the pass reads the
/// pattern of each face and darkens the texels of the block with the occlusion
/// of the corners around them.
struct OcclusionRenderer {
    batch: RenderBatch<
        (TextureArraySampler, CameraTransform, BlockTemplates),
        (),
        BlockRectInst,
        ColorDraw<1>,
        DepthDraw,
    >,
}

impl System<2> for OcclusionRenderer {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<FrameSlot>,
        ResRead<Camera>,
        ResRead<World>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, frame, camera, world) = res;

        for _ in comp.0.iter() {
            if let Some(frame) = frame.as_mut() {
                let colors = [ColorTarget {
                    view: &canvas.color_view,
                    load: LoadOp::Load,
                }];
                let depth = DepthTarget {
                    view: &canvas.depth_view,
                    load: LoadOp::Load,
                };

                frame.render::<ColorDraw<1>, DepthDraw>("occlusion", colors, depth, |mut pass| {
                    self.batch.begin(&mut pass);
                    self.batch.push(
                        &mut pass,
                        (&OCCLUSION_TEXTURES, &camera.transform, &BLOCK_TEMPLATES),
                    );

                    for region in world.visible_regions() {
                        if let Some(rects) = region.model().occlusion() {
                            self.batch.draw(&mut pass, &rects);
                        }
                    }
                });
            }
        }
    }
}

impl OcclusionRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "occlusion",
                    shader: "occlusion",
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: false,
                },
            ),
        }
    }
}

/// Draws camera-visible translucent blocks into the WBOIT targets.
struct TranslucentRenderer {
    batch: RenderBatch<
        (
            BlockTextureSet,
            CameraTransform,
            WorldLightsLayout,
            BlockTemplates,
        ),
        (),
        BlockRectInst,
        ColorDraw<2>,
        DepthDraw,
    >,
}

impl System<2> for TranslucentRenderer {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<FrameSlot>,
        ResRead<Camera>,
        ResRead<World>,
        ResRead<Lighting>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, frame, camera, world, lighting) = res;
        for _ in comp.0.iter() {
            if let Some(frame) = frame.as_mut() {
                let colors = [
                    ColorTarget {
                        view: &canvas.wboit_accum_view,
                        load: LoadOp::Clear(Color::TRANSPARENT),
                    },
                    ColorTarget {
                        view: &canvas.wboit_reveal_view,
                        load: LoadOp::Clear(Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                    },
                ];
                let depth = DepthTarget {
                    view: &canvas.depth_view,
                    load: LoadOp::Load,
                };

                frame.render::<ColorDraw<2>, DepthDraw>(
                    "translucent",
                    colors,
                    depth,
                    |mut pass| {
                        self.batch.begin(&mut pass);
                        self.batch.push(
                            &mut pass,
                            (
                                &BLOCK_SURFACES,
                                &camera.transform,
                                &lighting.bind_group,
                                &BLOCK_TEMPLATES,
                            ),
                        );
                        for region in world.visible_regions() {
                            if let Some(rects) = region.model().translucent() {
                                self.batch.draw(&mut pass, &rects);
                            }
                        }
                    },
                );
            }
        }
    }
}

impl TranslucentRenderer {
    fn new(canvas: &Canvas) -> Self {
        Self {
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "translucent",
                    shader: "block_wboit",
                    color_formats: [TextureFormat::Rgba16Float, TextureFormat::R16Float],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: Some(BlendState::ADDITIVE),
                    secondary_blend: Some(WBOIT_REVEAL_BLEND),
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: false,
                },
            ),
        }
    }
}

/// Composites the WBOIT targets over the opaque scene.
struct WboitResolver;

impl System<2> for WboitResolver {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (ResRead<Canvas>, ResWrite<FrameSlot>);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        if comp.0.iter().next().is_none() {
            return;
        }
        let (canvas, frame) = res;
        if let Some(frame) = frame.as_mut() {
            canvas.resolve_wboit(frame);
        }
    }
}
