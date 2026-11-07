use crate::ecs::*;
use crate::game::ModRegistry;
use crate::id_of;
use crate::render::*;
use crate::ui::*;
use crate::util::OnceInit;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::world::{BLOCK_TEXTURES, Block, BlockMeta};
use glam::camera::rh::view::look_at_mat4;
use glam::{Mat4, Vec2, Vec3, Vec4};
use glyph_brush::ab_glyph::FontArc;
use glyph_brush::*;
use log::error;
use std::collections::HashMap;
use std::path::Path;
use wgpu::*;

/// The textures of the interface, drawn as the layers of one array, and the names of the
/// layers, which are the files of `assets/textures/ui`. A layer is named by its file and
/// numbered by its position among the names, which is what the screens of
/// `assets/screens` name.
static UI_TEXTURES: OnceInit<BindSet<TextureArraySampler>> = OnceInit::new();
static UI_TEXTURE_HANDLES: OnceInit<Registry<()>> = OnceInit::new();

/// Loads the textures of the interface.
pub(super) fn init_ui_textures(canvas: &Canvas, mods: &ModRegistry) -> anyhow::Result<()> {
    let (reg, _, view) = load_texture_array(
        canvas,
        mods,
        "assets/textures/ui",
        "ui",
        TextureFormat::Rgba8UnormSrgb,
        false,
        |_| (),
    )?;

    let sampler = Sampler::new(
        canvas,
        &SamplerConfig {
            name: "ui",
            filter: FilterMode::Nearest,
            address_mode: AddressMode::Repeat,
            mipmap_filter: MipmapFilterMode::Linear,
            mip_level_count: 1,
            compare: None,
        },
    );

    let binding = BindSet::new(
        canvas,
        &BindSetConfig {
            name: "ui",
            content: (&view, &sampler),
        },
    );

    UI_TEXTURE_HANDLES.init(reg);
    UI_TEXTURES.init(binding);
    Ok(())
}

/// Side length of the glyph atlas texture.
const ATLAS_SIZE: u32 = 1024;

/// How much of the depth of an element the faces of its block use.
///
/// The faces have to differ in depth, because two fragments of the same depth
/// fail the `Greater` comparison the batches use; the band stays well below the
/// step between the depths of two elements, so a block never reaches into the
/// layer of the element in front of it.
const BLOCK_DEPTH: f32 = 1.0e-4;

/// The glyph atlas of the text batch, together with the brush that fills it.
struct TextAtlas {
    texture: Texture,
    bind: BindSet<TextureSampler>,
    brush: GlyphBrush<UiInst>,
}

impl TextAtlas {
    /// Creates the font, the atlas texture and the glyph brush.
    fn new(canvas: &Canvas, mods: &ModRegistry) -> Self {
        let font_asset = mods
            .asset(Path::new("assets/fonts/Pixelspace.ttf"))
            .expect("failed to find the ui font");
        let font = FontArc::try_from_vec(
            mods.read(&font_asset).expect("failed to read the ui font"),
        )
        .expect("failed to load the ui font");
        let texture = Texture::new(
            canvas,
            &TextureConfig {
                name: "text",
                texs: vec![None],
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                mip_level_count: 1,
                format: TextureFormat::R8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            },
        );
        let view = TextureView::new(
            &texture,
            &TextureViewConfig {
                name: "text",
                dimension: TextureViewDimension::D2,
                mip_level: None,
            },
        );
        let sampler = Sampler::new(
            canvas,
            &SamplerConfig {
                name: "text",
                filter: FilterMode::Nearest,
                address_mode: AddressMode::Repeat,
                mipmap_filter: MipmapFilterMode::Nearest,
                mip_level_count: 1,
                compare: None,
            },
        );
        let bind = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "text",
                content: (&view, &sampler),
            },
        );
        let brush = GlyphBrushBuilder::using_font(font)
            .draw_cache_position_tolerance(0.0)
            .draw_cache_scale_tolerance(0.0)
            // Text is laid out per viewport, so cached redraws cannot be reused.
            .cache_redraws(false)
            .initial_cache_size((ATLAS_SIZE, ATLAS_SIZE))
            .build();

        Self {
            texture,
            bind,
            brush,
        }
    }

    fn queue(&mut self, rect: AABB<Vec2>, z: f32, text: &UiText) {
        // The glyph brush anchors text by its alignment, so the position and the
        // bounds are taken from the rectangle the text is centred in.
        let (x, h_align) = match text.h_align {
            UiHAlign::Left => (rect.min.x, HorizontalAlign::Left),
            UiHAlign::Center => (rect.center().x, HorizontalAlign::Center),
            UiHAlign::Right => (rect.max.x, HorizontalAlign::Right),
        };
        let (y, v_align) = match text.v_align {
            UiVAlign::Top => (rect.min.y, VerticalAlign::Top),
            UiVAlign::Center => (rect.center().y, VerticalAlign::Center),
            UiVAlign::Bottom => (rect.max.y, VerticalAlign::Bottom),
        };

        let pos = Vec2::new(x, y);
        let bounds = rect.size();

        self.brush.queue(
            Section::default()
                .with_screen_position((pos.x, pos.y))
                .with_bounds((bounds.x, bounds.y))
                .with_layout(Layout::default().h_align(h_align).v_align(v_align))
                .add_text(
                    Text::new(&text.text)
                        .with_scale(text.scale)
                        .with_color(text.color)
                        .with_z(z),
                ),
        );
    }

    fn process(&mut self, canvas: &Canvas) -> anyhow::Result<Option<Vec<UiInst>>> {
        let action = self.brush.process_queued(
            |rect, data| {
                canvas.queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: Origin3d {
                            x: rect.min[0],
                            y: rect.min[1],
                            z: 0,
                        },
                        aspect: TextureAspect::All,
                    },
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(rect.width()),
                        rows_per_image: Some(rect.height()),
                    },
                    Extent3d {
                        width: rect.width(),
                        height: rect.height(),
                        depth_or_array_layers: 1,
                    },
                );
            },
            |v| UiInst {
                tex: id_of!(UI_TEXTURE_HANDLES, "white"),
                min: Vec2::new(v.pixel_coords.min.x, v.pixel_coords.min.y),
                max: Vec2::new(v.pixel_coords.max.x, v.pixel_coords.max.y),
                z: v.extra.z,
                min_uv: Vec2::new(v.tex_coords.min.x, v.tex_coords.min.y),
                max_uv: Vec2::new(v.tex_coords.max.x, v.tex_coords.max.y),
                color: v.extra.color,
            },
        )?;

        match action {
            // The vertices are only handed out when something changed; a
            // redraw means the buffer is still up to date.
            BrushAction::Draw(vertices) => Ok(Some(vertices)),

            BrushAction::ReDraw => Ok(None),
        }
    }
}

/// Draws every element of every open screen.
///
/// The quads and the glyphs are collected per viewport, because the z of an
/// element only has to be comparable with the elements of its own viewport.
pub(super) struct UiRenderer {
    quad_batch:
        RenderBatch<(Transformation, TextureArraySampler), (), UiInst, ColorDraw<1>, DepthDraw>,
    text_batch: RenderBatch<(Transformation, TextureSampler), (), UiInst, ColorDraw<1>, DepthDraw>,
    block_batch: RenderBatch<
        (TextureArraySampler, CameraTransform, Transformation),
        NormTexVertex,
        UiBlockInst,
        ColorDraw<1>,
        DepthDraw,
    >,
    /// The camera every block of the interface is seen through. It never
    /// changes, so it is uploaded once and shared by every block.
    block_camera: BindSet<CameraTransform>,
    /// One mesh per block the interface shows, with the rectangles of the
    /// elements that show it.
    blocks: HashMap<BlockMeta, UiBlockModel>,
    /// The identity transform of the fullscreen overlay, whose instances are
    /// already in normalised device coordinates.
    full: BindSet<Transformation>,
    quads: Vec<BufferVec<UiInst>>,
    glyphs: Vec<BufferVec<UiInst>>,
    dim: BufferVec<UiInst>,
    text: TextAtlas,
}

impl System<2> for UiRenderer {
    type CompQueries = (
        (),
        (
            CompRead<UiRect>,
            CompRead<UiSprite>,
            CompRead<UiText>,
            OptionalRead<UiBlock>,
        ),
    );
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<FrameSlot>,
        ResRead<ActiveScreens>,
        ResRead<Viewports>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (canvas, frame, screens, viewports) = res;

        if let Some(frame) = frame.as_mut() {
            let (rects, sprites, texts, blocks) = comp.1;
            // The number of layers of the texture array, which bounds the layer a sprite
            // names.
            let layers = UI_TEXTURE_HANDLES.entries.len() as u32;

            // The depth of an item is normalised by the number of items, so the
            // items are counted before anything is laid out.
            let count = screens
                .order
                .iter()
                .filter_map(|screen| screens.get(*screen))
                .flat_map(|screen| &screen.entities)
                .filter(|entity| rects.get(**entity).is_some())
                .map(|entity| {
                    sprites.get(*entity).is_some() as usize
                        + texts.get(*entity).is_some() as usize
                        + blocks.get(*entity).flatten().is_some() as usize
                })
                .sum::<usize>();
            let total = count + screens.pauses() as usize;

            // Later items end up higher, and z stays within (0, 1).
            let mut layer = 0usize;
            let next_z = |next: &mut usize| {
                *next += 1;
                *next as f32 / (total as f32 + 1.0)
            };

            let mut quads = vec![Vec::new(); ViewPortAlignment::ALL.len()];
            let mut queued = vec![Vec::new(); ViewPortAlignment::ALL.len()];
            let mut shown = vec![
                HashMap::<BlockMeta, (Block, Vec<UiBlockInst>)>::new();
                ViewPortAlignment::ALL.len()
            ];
            let mut dim = Vec::new();

            for screen in &screens.order {
                let Some(instance) = screens.get(*screen) else {
                    continue;
                };
                let screen_type = SCREEN_TYPES.get(*screen);
                let view = screen_type.viewport.idx();

                if screen_type.pauses && dim.is_empty() {
                    dim.push(UiInst {
                        tex: id_of!(UI_TEXTURE_HANDLES, "white"),
                        min: Vec2::ZERO,
                        max: Vec2::ONE,
                        z: next_z(&mut layer),
                        min_uv: Vec2::ZERO,
                        max_uv: Vec2::ONE,
                        color: [0.0, 0.0, 0.0, 0.5],
                    });
                }

                for entity in &instance.entities {
                    let Some(rect) = rects.get(*entity) else {
                        continue;
                    };

                    if let Some(sprite) = sprites.get(*entity) {
                        // A sprite that names a layer the atlas does not have
                        // would sample the wrong texture, so the id is clamped;
                        // debug builds point at the mistake instead.
                        debug_assert!(sprite.tex < layers);

                        quads[view].push(UiInst {
                            tex: sprite.tex.min(layers - 1),
                            min: rect.0.min,
                            max: rect.0.max,
                            z: next_z(&mut layer),
                            min_uv: sprite.uv.min,
                            max_uv: sprite.uv.max,
                            color: sprite.color.into(),
                        });
                    }

                    if let Some(text) = texts.get(*entity) {
                        queued[view].push((rect.0, next_z(&mut layer), text));
                    }

                    if let Some(block) = blocks.get(*entity).flatten() {
                        let z = next_z(&mut layer);

                        shown[view]
                            .entry(block.0.to_meta())
                            .or_insert_with(|| (block.0, Vec::new()))
                            .1
                            .push(UiBlockInst {
                                min: rect.0.min,
                                max: rect.0.max,
                                min_z: z,
                                max_z: z + BLOCK_DEPTH,
                            });
                    }
                }
            }

            for (view, buffer) in self.quads.iter_mut().enumerate() {
                buffer.set_content(canvas, &quads[view]);
            }
            self.dim.set_content(canvas, &dim);

            self.blocks
                .retain(|meta, _| shown.iter().any(|view| view.contains_key(meta)));

            for (view, blocks) in shown.iter().enumerate() {
                for (meta, (block, instances)) in blocks {
                    self.blocks
                        .entry(*meta)
                        .or_insert_with(|| UiBlockModel::new(canvas, *block))
                        .instances[view]
                        .set_content(canvas, instances);
                }
            }

            for (view, buffer) in self.glyphs.iter_mut().enumerate() {
                queued[view]
                    .iter()
                    .for_each(|(rect, z, text)| self.text.queue(*rect, *z, text));

                match self.text.process(canvas) {
                    Ok(Some(vertices)) => buffer.set_content(canvas, &vertices),

                    Ok(None) => (),

                    Err(e) => error!("failed to rasterize text: {}", e),
                }
            }

            // All viewports are drawn in the same pass, so their depth values are
            // comparable and the dim overlay can cover the ones below it.
            let colors = [ColorTarget {
                view: &canvas.color_view,
                load: LoadOp::Load,
            }];
            let depth = DepthTarget {
                view: &canvas.depth_view,
                load: LoadOp::Clear(0.0),
            };

            frame.render::<ColorDraw<1>, DepthDraw>("ui", colors, depth, |mut pass| {
                for alignment in ViewPortAlignment::ALL {
                    let view = alignment.idx();
                    let viewport = viewports.get(*alignment);

                    self.quad_batch.begin(&mut pass);
                    self.quad_batch
                        .push(&mut pass, (&viewport.transform, &UI_TEXTURES));
                    self.quad_batch.draw(&mut pass, &UiList(&self.quads[view]));

                    self.block_batch.begin(&mut pass);
                    self.block_batch.push(
                        &mut pass,
                        (&BLOCK_TEXTURES, &self.block_camera, &viewport.transform),
                    );
                    self.block_batch.draw(
                        &mut pass,
                        &UiBlocks {
                            models: &self.blocks,
                            alignment: *alignment,
                        },
                    );

                    self.text_batch.begin(&mut pass);
                    self.text_batch
                        .push(&mut pass, (&viewport.transform, &self.text.bind));
                    self.text_batch.draw(&mut pass, &UiList(&self.glyphs[view]));
                }

                self.quad_batch.begin(&mut pass);
                self.quad_batch.push(&mut pass, (&self.full, &UI_TEXTURES));
                self.quad_batch.draw(&mut pass, &UiList(&self.dim));
            });
        }
    }
}

impl UiRenderer {
    /// Creates the batches, the buffers and the atlases of the UI.
    pub fn new(canvas: &Canvas, mods: &ModRegistry) -> Self {
        // Maps the unit square onto the whole window, in normalised device
        // coordinates.
        let full_buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: "ui_fullscreen",
                init: BufferInit::Content(&[Mat4::from_cols_array(&[
                    2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                    1.0,
                ])]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        );
        let full = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "ui_fullscreen",
                content: &full_buffer,
            },
        );

        // The camera of the blocks never changes, so its matrix is uploaded
        // once, unlike the one of the world camera.
        let block_camera_buffer = Buffer::new(
            canvas,
            &BufferConfig {
                name: "ui_block_camera",
                init: BufferInit::Content(&[CameraTransform {
                    matrix: Self::block_camera(),
                    // The blocks of the interface are not sprites, so nothing
                    // turns towards the camera that draws them.
                    position: Vec4::ZERO,
                }]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        );
        let block_camera = BindSet::new(
            canvas,
            &BindSetConfig {
                name: "ui_block_camera",
                content: &block_camera_buffer,
            },
        );

        Self {
            quad_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "ui",
                    shader: "ui",
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            text_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "ui_text",
                    shader: "text",
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            block_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "ui_block",
                    shader: "ui_block",
                    // Blocks are drawn opaque; translucent ones would need a
                    // batch of their own, drawn after the opaque geometry.
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: None,
                    secondary_blend: None,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            block_camera,
            blocks: HashMap::new(),
            full,
            quads: ViewPortAlignment::ALL
                .iter()
                .map(|_| BufferVec::instance(canvas, "ui_quad", BufferInit::Size(0)))
                .collect(),
            glyphs: ViewPortAlignment::ALL
                .iter()
                .map(|_| BufferVec::instance(canvas, "ui_glyph", BufferInit::Size(0)))
                .collect(),
            dim: BufferVec::instance(canvas, "ui_dim", BufferInit::Size(0)),
            text: TextAtlas::new(canvas, mods),
        }
    }

    /// The transform of the camera every block of the interface is seen
    /// through: an orthographic view from outside the `(1, 1, 1)` corner of the
    /// block, looking back down the diagonal at its center.
    fn block_camera() -> Mat4 {
        // The eye sits one block diagonal beyond the `(1, 1, 1)` corner, along
        // the direction it looks back in.
        let distance = 1.0;
        let eye = Vec3::ONE + Vec3::ONE * distance;
        let view = look_at_mat4(eye, Vec3::splat(0.5), Vec3::Y);

        // The `(1, 1, 1)` corner is the closest point of the block to the eye
        // and the `(0, 0, 0)` corner the farthest, so the frustum has to cover
        // that range, with a margin in front of the closer one.
        let near = distance * 3.0f32.sqrt() - 0.125;
        let far = distance * 3.0f32.sqrt() + 2.0 * 3.0f32.sqrt() + 0.125;

        // A unit block seen along its space diagonal is a regular hexagon of
        // radius sqrt(2/3), so this half extent leaves a small margin around it.
        let radius = (2.0f32 / 3.0).sqrt() * 1.05;

        // The projection flips y, because the space of a viewport grows
        // downwards. Its depth is reversed, like the projection of the world
        // camera: the near plane maps to 1 and the far plane to 0, which
        // matches the `Greater` comparison the batches use.
        let proj = Mat4::from_cols_array(&[
            1.0 / radius,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0 / radius,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / (far - near),
            0.0,
            0.0,
            0.0,
            far / (far - near),
            1.0,
        ]);

        proj * view
    }
}

/// The instances of one viewport, as geometry that a UI batch can draw.
struct UiList<'a>(&'a BufferVec<UiInst>);

impl Render<(), UiInst> for UiList<'_> {
    fn rendered(&self) -> Vec<RenderItem<'_, (), UiInst>> {
        let mut items = Vec::new();

        if let Ok(item) = RenderItem::new(&EMPTY_BUFFER_VEC, &QUAD_INDEX_BUFFER_VEC, self.0) {
            items.push(item);
        }

        items
    }
}

/// The geometry of one block, together with the rectangles of the elements that
/// show it.
struct UiBlockModel {
    vertices: BufferVec<NormTexVertex>,
    indices: BufferVec<u16>,
    /// The rectangles of the elements that show this block, one buffer per
    /// viewport, because the depths of an element are only comparable with the
    /// elements of its own viewport.
    instances: Vec<BufferVec<UiBlockInst>>,
}

impl UiBlockModel {
    /// Uploads the mesh of `block`, leaving room for the rectangles of the
    /// elements that show it.
    fn new(canvas: &Canvas, block: Block) -> Self {
        let mesh = block.mesh();

        Self {
            vertices: mesh.vertex_buffer(canvas, "ui_block"),
            indices: mesh.index_buffer_vec(canvas, "ui_block"),
            instances: ViewPortAlignment::ALL
                .iter()
                .map(|_| BufferVec::instance(canvas, "ui_block", BufferInit::Size(0)))
                .collect(),
        }
    }
}

/// The blocks of one viewport, as geometry that the block batch draws.
struct UiBlocks<'a> {
    models: &'a HashMap<BlockMeta, UiBlockModel>,
    alignment: ViewPortAlignment,
}

impl Render<NormTexVertex, UiBlockInst> for UiBlocks<'_> {
    fn rendered(&self) -> Vec<RenderItem<'_, NormTexVertex, UiBlockInst>> {
        self.models
            .values()
            .filter_map(|model| {
                RenderItem::new(
                    &model.vertices,
                    &model.indices,
                    &model.instances[self.alignment.idx()],
                )
                .ok()
            })
            .collect()
    }
}
