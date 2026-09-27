use crate::ecs::*;
use crate::game::{InputState, Paused};
use crate::render::ViewPortAlignment;
use crate::resources;
use crate::ui::*;
use crate::util::bounding::AABB;
use crate::util::collection::Registry;
use crate::util::Id;
use glam::Vec2;
use glyph_brush::{HorizontalAlign, VerticalAlign};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use winit::keyboard::KeyCode;

/// The id of the crosshair and welcome text screen.
pub const HUD: Id = 0;
/// The id of the hotbar screen.
pub const HOTBAR: Id = 1;
/// The id of the pause menu.
pub const PAUSE: Id = 2;

/// Number of slots on the hotbar.
const SLOT_COUNT: usize = 9;

/// The screen types of the game, in the order their ids are registered in.
pub static SCREEN_TYPES: LazyLock<Registry<ScreenType>> = LazyLock::new(|| {
    let mut screens = Registry::new();

    screens.register(
        HUD,
        "hud",
        ScreenType {
            viewport: ViewPortAlignment::Middle,
            pauses: false,
            dismissible: false,
            entities: ScreenType::hud(),
        },
    );
    screens.register(
        HOTBAR,
        "hotbar",
        ScreenType {
            viewport: ViewPortAlignment::Bottom,
            pauses: false,
            dismissible: false,
            entities: ScreenType::hotbar(),
        },
    );
    screens.register(
        PAUSE,
        "pause",
        ScreenType {
            viewport: ViewPortAlignment::Middle,
            pauses: true,
            dismissible: true,
            entities: ScreenType::pause(),
        },
    );

    screens
});

/// The definition of a screen: where it is drawn, how it behaves and which
/// elements it contains.
pub struct ScreenType {
    /// The viewport the screen is laid out in and drawn into.
    pub viewport: ViewPortAlignment,
    /// Whether the game pauses while this screen is open.
    pub pauses: bool,
    /// Whether escape closes this screen again.
    pub dismissible: bool,
    /// The elements of the screen, in drawing order.
    pub entities: Vec<EntityDescriptor>,
}

/// An instance of a [`ScreenType`]: the elements it spawned, which are drawn
/// back to front.
pub struct Screen {
    pub entities: Vec<Id>,
}

impl ScreenType {
    /// The crosshair and the welcome text of the HUD.
    fn hud() -> Vec<EntityDescriptor> {
        vec![
            EntityDescriptor::new()
                .with(UiRect {
                    rect: AABB {
                        min: Vec2::splat(120.0),
                        max: Vec2::splat(136.0),
                    },
                })
                .with(UiSprite {
                    tex: UI_CROSSHAIR,
                    uv: AABB {
                        min: Vec2::ZERO,
                        max: Vec2::ONE,
                    },
                    color: [1.0; 4],
                }),
            EntityDescriptor::new()
                .with(UiRect {
                    rect: AABB {
                        min: Vec2::new(0.0, 0.0),
                        max: Vec2::new(256.0, 16.0),
                    },
                })
                .with(UiText {
                    text: "Oh, hi! Welcome to my block game!".into(),
                    h_align: HorizontalAlign::Center,
                    v_align: VerticalAlign::Top,
                    color: [1.0; 4],
                    scale: 10.0,
                }),
        ]
    }

    /// The slots of the hotbar, with the first one selected.
    fn hotbar() -> Vec<EntityDescriptor> {
        let slot = Vec2::splat(18.0);
        let gap = 4.0;
        let total = slot.x * SLOT_COUNT as f32 + gap * (SLOT_COUNT - 1) as f32;
        let origin = Vec2::new((256.0 - total) / 2.0, 256.0 - slot.y - 6.0);

        (0..SLOT_COUNT)
            .map(|i| {
                let pos = origin + Vec2::new((slot.x + gap) * i as f32, 0.0);
                let color = match i {
                    0 => [0.375, 0.625, 0.375, 1.0],
                    _ => [0.125, 0.125, 0.125, 0.875],
                };

                EntityDescriptor::new()
                    .with(UiRect {
                        rect: AABB {
                            min: pos,
                            max: pos + slot,
                        },
                    })
                    .with(UiSprite {
                        tex: UI_WHITE,
                        uv: AABB {
                            min: Vec2::ZERO,
                            max: Vec2::ONE,
                        },
                        color,
                    })
            })
            .collect()
    }

    /// The pause menu: a dimmed panel with a title and a resume button.
    fn pause() -> Vec<EntityDescriptor> {
        vec![
            EntityDescriptor::new()
                .with(UiRect {
                    rect: AABB {
                        min: Vec2::new(64.0, 96.0),
                        max: Vec2::new(192.0, 160.0),
                    },
                })
                .with(UiSprite {
                    tex: UI_WHITE,
                    uv: AABB {
                        min: Vec2::ZERO,
                        max: Vec2::ONE,
                    },
                    color: [0.125, 0.125, 0.125, 0.875],
                })
                .with(OnClick(Arc::new(|components, _, id| {
                    let Ok(sprites) = components.try_get::<UiSprite>() else {
                        return Vec::new();
                    };

                    if let Some(sprite) = sprites.get_mut::<UiSprite>(id) {
                        sprite.color.swap(0, 2);
                    }

                    Vec::new()
                }))),
            EntityDescriptor::new()
                .with(UiRect {
                    rect: AABB {
                        min: Vec2::new(64.0, 104.0),
                        max: Vec2::new(192.0, 120.0),
                    },
                })
                .with(UiText {
                    text: "Game Paused".into(),
                    h_align: HorizontalAlign::Center,
                    v_align: VerticalAlign::Center,
                    color: [1.0; 4],
                    scale: 10.0,
                }),
            EntityDescriptor::new()
                .with(UiRect {
                    rect: AABB {
                        min: Vec2::new(96.0, 128.0),
                        max: Vec2::new(160.0, 152.0),
                    },
                })
                .with(UiSprite {
                    tex: UI_WHITE,
                    uv: AABB {
                        min: Vec2::ZERO,
                        max: Vec2::ONE,
                    },
                    color: [0.25, 0.5, 0.25, 1.0],
                })
                .with(UiText {
                    text: "Resume".into(),
                    h_align: HorizontalAlign::Center,
                    v_align: VerticalAlign::Center,
                    color: [1.0; 4],
                    scale: 10.0,
                })
                .with(OnClick(Arc::new(|_, resources, _| {
                    let (Ok(screens), Ok(paused)) = (
                        resources.try_get_mut::<ActiveScreens>(),
                        resources.try_get_mut::<Paused>(),
                    ) else {
                        return Vec::new();
                    };

                    let commands = screens.close_top();
                    paused.0 = screens.pauses();

                    commands
                }))),
        ]
    }
}

resources! {
    /// The screens that are currently open, and the elements they spawned.
    pub struct ActiveScreens {
        pub screens: HashMap<Id, Screen>,
        // The open order of the screens, where the last entry is the topmost
        // screen: it receives the clicks and is drawn last.
        pub order: Vec<Id>,
    };
}

impl ActiveScreens {
    /// The topmost screen.
    pub fn top(&self) -> Option<Id> {
        self.order.last().copied()
    }

    /// Returns whether `screen` is open.
    pub fn contains(&self, screen: Id) -> bool {
        self.screens.contains_key(&screen)
    }

    /// The open instance of `screen`.
    pub fn get(&self, screen: Id) -> Option<&Screen> {
        self.screens.get(&screen)
    }

    /// Returns whether any open screen pauses the game.
    pub fn pauses(&self) -> bool {
        self.order
            .iter()
            .any(|screen| SCREEN_TYPES.get(*screen).pauses)
    }

    /// Returns whether any open screen is a menu that the player can close.
    pub fn menu(&self) -> bool {
        self.order
            .iter()
            .any(|screen| SCREEN_TYPES.get(*screen).dismissible)
    }

    /// Opens `screen` and returns the commands that spawn its elements.
    pub fn open(&mut self, screen: Id) -> Vec<Command> {
        if self.contains(screen) {
            return Vec::new();
        }

        self.screens.insert(
            screen,
            Screen {
                entities: Vec::new(),
            },
        );
        self.order.push(screen);

        SCREEN_TYPES
            .get(screen)
            .entities
            .iter()
            .enumerate()
            .map(|(priority, entity)| {
                Command::Spawn(entity.clone().with(ScreenTag {
                    screen,
                    priority: priority as u32,
                }))
            })
            .collect()
    }

    /// Closes the topmost screen when it is dismissible, returning the commands
    /// that remove its elements.
    pub fn close_top(&mut self) -> Vec<Command> {
        let Some(screen) = self.top() else {
            return Vec::new();
        };

        if !SCREEN_TYPES.get(screen).dismissible {
            return Vec::new();
        }

        self.order.pop();

        self.screens
            .remove(&screen)
            .map(|screen| screen.entities)
            .unwrap_or_default()
            .into_iter()
            .map(Command::Despawn)
            .collect()
    }
}

/// Collects the elements of the screens that were just opened.
pub struct ScreenCollector;

impl System for ScreenCollector {
    type CompQuery = CompRead<ScreenTag>;
    type ResQuery = ResWrite<ActiveScreens>;

    fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        let tags = entities.components.try_get::<ScreenTag>()?;
        let screens = resources.try_get_mut::<ActiveScreens>()?;

        let mut collected: HashMap<Id, Vec<(u32, Id)>> = HashMap::new();
        let mut commands = Vec::new();

        // The elements of a screen arrive as tagged entities, which are sorted
        // by their priority so that the screen knows what to draw on top.
        for (entity, tag) in tags.iter::<ScreenTag>() {
            match screens.contains(tag.screen) {
                true => collected
                    .entry(tag.screen)
                    .or_default()
                    .push((tag.priority, entity)),

                // The screen was closed before its elements were collected.
                false => commands.push(Command::Despawn(entity)),
            }

            // The tag is only needed while the screen is being built.
            commands.push(Command::Remove((entity, TypeId::of::<ScreenTag>())));
        }

        for (screen, mut entities) in collected {
            entities.sort();

            if let Some(instance) = screens.screens.get_mut(&screen) {
                instance
                    .entities
                    .extend(entities.into_iter().map(|(_, entity)| entity));
            }
        }

        Ok(commands)
    }
}

/// Opens the pause screen and closes it again when escape is pressed.
pub struct ScreenController;

impl System for ScreenController {
    type CompQuery = ();
    type ResQuery = (
        ResRead<InputState>,
        ResWrite<ActiveScreens>,
        ResWrite<Paused>,
    );

    fn update(
        &mut self,
        _: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        if !resources
            .try_get::<InputState>()?
            .is_key_just_pressed(KeyCode::Escape)
        {
            return Ok(Vec::new());
        }

        let screens = resources.try_get_mut::<ActiveScreens>()?;

        let commands = match screens.top() {
            Some(screen) if SCREEN_TYPES.get(screen).dismissible => screens.close_top(),

            _ => screens.open(PAUSE),
        };

        resources.try_get_mut::<Paused>()?.0 = screens.pauses();

        Ok(commands)
    }
}
