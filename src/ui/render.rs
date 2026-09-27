use crate::ecs::*;
use crate::render::*;
use crate::ui::*;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::Id;
use glam::{Mat4, Vec2};
use glyph_brush::ab_glyph::FontArc;
use glyph_brush::*;
use log::error;
use std::fs::File;
use wgpu::*;

/// Layer of the UI atlas holding a white pixel, used to draw plain quads.
pub const UI_WHITE: Id = 0;
/// Layer of the UI atlas holding the crosshair.
pub const UI_CROSSHAIR: Id = 1;

/// Number of layers in the UI texture array.
const UI_TEXTURES: u32 = 2;
/// Side length of the glyph atlas texture, and of the UI atlas.
const ATLAS_SIZE: u32 = 1024;

/// The glyph atlas of the text batch, together with the brush that fills it.
struct TextAtlas {
    texture: Texture,
    bind: BindSet<TextureSampler>,
    brush: GlyphBrush<UiInst>,
}

impl TextAtlas {
    /// Creates the font, the atlas texture and the glyph brush.
    fn new(canvas: &Canvas) -> Self {
        let font = FontArc::try_from_slice(include_bytes!("../../assets/fonts/Pixelspace.ttf"))
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
                storage: false,
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
                address_mode: AddressMode::Repeat,
                mipmap_filter: MipmapFilterMode::Nearest,
                mip_level_count: 1,
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
        let pos = Vec2::new(
            match text.h_align {
                HorizontalAlign::Left => rect.min.x,
                HorizontalAlign::Center => rect.center().x,
                HorizontalAlign::Right => rect.max.x,
            },
            match text.v_align {
                VerticalAlign::Top => rect.min.y,
                VerticalAlign::Center => rect.center().y,
                VerticalAlign::Bottom => rect.max.y,
            },
        );
        let bounds = rect.size();

        self.brush.queue(
            Section::default()
                .with_screen_position((pos.x, pos.y))
                .with_bounds((bounds.x, bounds.y))
                .with_layout(
                    Layout::default()
                        .h_align(text.h_align)
                        .v_align(text.v_align),
                )
                .add_text(
                    Text::new(&text.text)
                        .with_scale(text.scale)
                        .with_color(text.color)
                        .with_z(z),
                ),
        );
    }

    fn process(&mut self, canvas: &Canvas) -> anyhow::Result<Option<Vec<UiInst>>> {
        let Self { texture, brush, .. } = self;

        let action = brush.process_queued(
            |rect, data| {
                canvas.queue.write_texture(
                    TexelCopyTextureInfo {
                        texture,
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
                tex: UI_WHITE,
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
pub struct UiRenderer {
    desc: RenderDescriptor<'static>,
    quad_batch: RenderBatch<(Transformation, TextureArraySampler), (), UiInst>,
    text_batch: RenderBatch<(Transformation, TextureSampler), (), UiInst>,
    atlas: BindSet<TextureArraySampler>,
    /// The identity transform of the fullscreen overlay, whose instances are
    /// already in normalised device coordinates.
    full: BindSet<Transformation>,
    quads: Vec<BufferVec<UiInst>>,
    glyphs: Vec<BufferVec<UiInst>>,
    dim: BufferVec<UiInst>,
    text: TextAtlas,
}

impl System for UiRenderer {
    type CompQuery = (CompRead<UiRect>, CompRead<UiSprite>, CompRead<UiText>);
    type ResQuery = (
        ResRead<Canvas>,
        ResWrite<Option<Frame>>,
        ResRead<ActiveScreens>,
        ResRead<Viewports>,
    );

    fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        if let Some(frame) = resources.get_mut::<Option<Frame>>() {
            let canvas = resources.get::<Canvas>();
            let screens = resources.try_get::<ActiveScreens>()?;
            let viewports = resources.try_get::<Viewports>()?;
            let rects = entities.components.try_get::<UiRect>()?;
            let sprites = entities.components.try_get::<UiSprite>()?;
            let texts = entities.components.try_get::<UiText>()?;

            // The depth of an item is normalised by the number of items, so the
            // items are counted before anything is laid out.
            let count = screens
                .order
                .iter()
                .filter_map(|screen| screens.get(*screen))
                .flat_map(|screen| &screen.entities)
                .filter(|entity| rects.contains(**entity))
                .map(|entity| sprites.contains(*entity) as usize + texts.contains(*entity) as usize)
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
            let mut dim = Vec::new();

            for screen in &screens.order {
                let Some(instance) = screens.get(*screen) else {
                    continue;
                };
                let screen_type = SCREEN_TYPES.get(*screen);
                let view = screen_type.viewport.idx();

                if screen_type.pauses && dim.is_empty() {
                    dim.push(UiInst {
                        tex: UI_WHITE,
                        min: Vec2::ZERO,
                        max: Vec2::ONE,
                        z: next_z(&mut layer),
                        min_uv: Vec2::ZERO,
                        max_uv: Vec2::ONE,
                        color: [0.0, 0.0, 0.0, 0.5],
                    });
                }

                for entity in &instance.entities {
                    let Some(rect) = rects.get::<UiRect>(*entity) else {
                        continue;
                    };

                    if let Some(sprite) = sprites.get::<UiSprite>(*entity) {
                        quads[view].push(UiInst {
                            tex: sprite.tex.min(UI_TEXTURES - 1),
                            min: rect.rect.min,
                            max: rect.rect.max,
                            z: next_z(&mut layer),
                            min_uv: sprite.uv.min,
                            max_uv: sprite.uv.max,
                            color: sprite.color,
                        });
                    }

                    if let Some(text) = texts.get::<UiText>(*entity) {
                        queued[view].push((rect.rect, next_z(&mut layer), text));
                    }
                }
            }

            for (view, buffer) in self.quads.iter_mut().enumerate() {
                buffer.set_content(canvas, &quads[view]);
            }
            self.dim.set_content(canvas, &dim);

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
            frame.render(&self.desc, |mut pass| {
                for alignment in ViewPortAlignment::ALL {
                    let view = alignment.idx();
                    let viewport = viewports.get(*alignment);

                    self.quad_batch.begin(&mut pass);
                    self.quad_batch
                        .push(&mut pass, (&viewport.transform, &self.atlas));
                    self.quad_batch.draw(&mut pass, &UiList(&self.quads[view]));

                    self.text_batch.begin(&mut pass);
                    self.text_batch
                        .push(&mut pass, (&viewport.transform, &self.text.bind));
                    self.text_batch.draw(&mut pass, &UiList(&self.glyphs[view]));
                }

                self.quad_batch.begin(&mut pass);
                self.quad_batch.push(&mut pass, (&self.full, &self.atlas));
                self.quad_batch.draw(&mut pass, &UiList(&self.dim));
            });
        }

        Ok(Vec::new())
    }
}

impl UiRenderer {
    /// Creates the batches, the buffers and the atlases of the UI.
    pub fn new(canvas: &Canvas) -> Self {
        let crosshair = Tex::from_png(
            File::open("assets/textures/crosshair.png").expect("failed to open the crosshair"),
        )
        .expect("failed to load the crosshair");
        let mut textures = Registry::<Tex>::new();
        // The layers of an array texture must all have the same size, so the
        // white layer is created with the size of the crosshair.
        textures.register(
            UI_WHITE,
            "white",
            Tex {
                width: crosshair.width,
                height: crosshair.height,
                data: vec![255; (crosshair.width * crosshair.height * 4) as usize],
            },
        );
        textures.register(UI_CROSSHAIR, "crosshair", crosshair);

        let atlas = textures.create_texture_sampler(canvas, "ui", false);
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

        Self {
            desc: RenderDescriptor {
                name: "ui",
                color_load: LoadOp::Load,
                depth_load: LoadOp::Clear(0.0),
            },
            quad_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "ui",
                    shader: "ui",
                    translucent: true,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            text_batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "ui_text",
                    shader: "text",
                    translucent: true,
                    topology: PrimitiveTopology::TriangleList,
                    depth_write: true,
                },
            ),
            atlas,
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
            text: TextAtlas::new(canvas),
        }
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
