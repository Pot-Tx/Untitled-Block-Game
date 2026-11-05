use crate::actor::{Inventory, PlayerControlled};
use crate::ecs::*;
use crate::game::{INPUT_MAP, InputState, Paused};
use crate::render::ViewPortAlignment;
use crate::resources;
use crate::ui::*;
use crate::util::Id;
use crate::util::OnceInit;
use crate::util::collection::Registry;
use crate::{by_name, id_of};
use anyhow::anyhow;
use ron::value::RawValue;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use winit::event::MouseButton;

/// Every file of `assets/screens` names a screen, how it behaves and which
/// elements it contains; [`init_screen_types`] turns them into the registry the
/// game looks screens up in.
pub(crate) static SCREEN_TYPES: OnceInit<Registry<ScreenType>> = OnceInit::new();

/// One element of a screen as it is written in a data file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct RawElement {
    /// The components of the element, named and written like those of an actor.
    #[serde(default)]
    pub components: Vec<(String, Box<RawValue>)>,
    /// The names of the systems that handle the triggers of this element.
    #[serde(default)]
    pub triggers: Vec<String>,
}

/// One screen as it is written in a data file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct RawScreen {
    /// The viewport the screen is laid out in and drawn into.
    pub viewport: ViewPortAlignment,
    /// Whether the game pauses while this screen is open.
    #[serde(default)]
    pub pauses: bool,
    /// Whether escape closes this screen again.
    #[serde(default)]
    pub dismissible: bool,
    /// The elements of the screen, in drawing order.
    pub elements: Vec<RawElement>,
}

/// Builds the screens of `assets/screens`.
///
/// Has to be called after the systems of the user interface have been
/// registered, because an element names the systems that handle its triggers.
pub(crate) fn init_screen_types(
    components: &ComponentManager,
    systems: &SystemManager<2>,
) -> anyhow::Result<()> {
    let reg = Registry::load_rons_from("assets/screens")?;
    SCREEN_TYPES.init(reg.map(|raw| {
        ScreenType::from_raw(components, systems, raw).expect("failed to create a screen from data")
    }));
    Ok(())
}

/// The definition of a screen: where it is drawn, how it behaves and which
/// elements it contains.
pub(crate) struct ScreenType {
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
#[derive(Default)]
pub struct Screen {
    pub entities: Vec<Id>,
    pub ready: bool,
}

impl Screen {
    pub fn new(entities: Vec<Id>) -> Self {
        Self {
            entities,
            ready: false,
        }
    }
}

impl ScreenType {
    /// Builds a screen from its data, turning every element into the descriptor
    /// it is spawned with.
    fn from_raw(
        components: &ComponentManager,
        systems: &SystemManager<2>,
        raw: &RawScreen,
    ) -> anyhow::Result<Self> {
        let mut entities = Vec::new();

        for element in raw.elements.iter() {
            // The element data is copied once while the screens are built,
            // because `EntityDescriptor::from_raw` reads the shape a whole
            // entity has in a data file.
            let entity = RawEntity {
                components: element.components.clone(),
            };
            let mut descriptor = EntityDescriptor::from_raw(components, &entity)?;

            let triggers = resolve_triggers(systems, &element.triggers)?;
            if !triggers.is_empty() {
                descriptor = descriptor.with(Triggers(triggers));
            }

            entities.push(descriptor);
        }

        Ok(Self {
            viewport: raw.viewport,
            pauses: raw.pauses,
            dismissible: raw.dismissible,
            entities,
        })
    }
}

/// Turns the system names of a data file into the type ids a [`Triggers`]
/// component holds.
fn resolve_triggers(systems: &SystemManager<2>, names: &[String]) -> anyhow::Result<Vec<TypeId>> {
    names
        .iter()
        .map(|name| {
            systems
                .type_id_of(name)
                .ok_or_else(|| anyhow!("system {} is not registered", name))
        })
        .collect()
}

resources! {
    /// The screens that are currently open, and the elements they spawned.
    #[derive(Default)]
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

    pub fn get_mut(&mut self, screen: Id) -> Option<&mut Screen> {
        self.screens.get_mut(&screen)
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

    /// Opens `screen` and spawns its elements with `spawn`, recording the ids
    /// in drawing order.
    pub fn open(&mut self, type_id: Id, mut spawn: impl FnMut(EntityDescriptor) -> Id) {
        if self.contains(type_id) {
            return;
        }

        let entities = SCREEN_TYPES
            .get(type_id)
            .entities
            .iter()
            .map(|entity| spawn(entity.clone()))
            .collect();

        self.screens.insert(type_id, Screen::new(entities));
        self.order.push(type_id);
    }

    /// Closes the topmost screen when it is dismissible, returning the elements
    /// it removes.
    pub fn close_top(&mut self) -> Vec<Id> {
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
    }
}

/// Opens the pause screen and closes it again when escape is pressed.
pub(super) struct ScreenController;

pub(super) struct HotbarController;

impl System<2> for ScreenController {
    type CompQueries = ((), (CompRead<Triggers>,));
    type ResQuery = (
        ResRead<ActiveElement>,
        ResRead<InputState>,
        ResWrite<ActiveScreens>,
        ResWrite<Paused>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        com: [Commander<'_>; 2],
    ) {
        if (res.0.triggers_for::<Self>(&comp.1.0)
            && res.1.is_button_just_released(MouseButton::Left))
            || res.1.is_input_present(by_name!(INPUT_MAP, "return"))
        {
            let mut ui = com[1].lock();

            match res.2.top() {
                Some(screen) if SCREEN_TYPES.get(screen).dismissible => {
                    for entity in res.2.close_top() {
                        ui.despawn(entity);
                    }
                }

                _ => {
                    res.2
                        .open(id_of!(SCREEN_TYPES, "pause"), |element| ui.spawn(element));
                }
            }

            res.3.0 = res.2.pauses();
        }
    }
}

impl System<2> for HotbarController {
    type CompQueries = (
        (CompRead<PlayerControlled>, CompWrite<Inventory>),
        (CompWrite<UiSprite>, CompWrite<UiBlock>),
    );
    type ResQuery = (
        ResRead<Paused>,
        ResRead<InputState>,
        ResWrite<ActiveScreens>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        com: [Commander<'_>; 2],
    ) {
        for (_, _, inv) in comp.0.iter() {
            if !res.0.0 {
                let delta = -(res.1.mouse_scroll.y.round() as isize);
                inv.scroll(delta);
            }

            if let Some(hotbar) = res.2.get_mut(id_of!(SCREEN_TYPES, "hotbar")) {
                let (prev, changed) = if hotbar.ready {
                    (inv.prev(), inv.changed())
                } else {
                    hotbar.ready = true;
                    (usize::MAX, (0..inv.cap()).collect())
                };

                let cur = inv.cur();
                if prev != cur {
                    if let Some(sprite) = hotbar
                        .entities
                        .get(prev)
                        .and_then(|&entity| comp.1.0.get(entity))
                    {
                        sprite.color = Vec4::new(0.125, 0.125, 0.125, 0.875);
                    }

                    if let Some(sprite) = hotbar
                        .entities
                        .get(cur)
                        .and_then(|&entity| comp.1.0.get(entity))
                    {
                        sprite.color = Vec4::new(0.375, 0.625, 0.375, 1.0);
                    }
                }

                let mut guard = com[1].lock();
                for index in changed {
                    if let Some(&entity) = hotbar.entities.get(index) {
                        match (comp.1.1.get(entity), inv.get(index)) {
                            (None, Some(block)) => {
                                guard.insert(entity, UiBlock(block));
                            }

                            (Some(_), None) => {
                                guard.remove::<UiBlock>(entity);
                            }

                            (Some(ui_block), Some(block)) => {
                                ui_block.0 = block;
                            }

                            _ => (),
                        }
                    }
                }
            }

            inv.flush();
        }
    }
}

pub(super) struct FrameRater {
    duration: Duration,
    timer: Instant,
    recent: u32,
}

impl System<2> for FrameRater {
    type CompQueries = ((), (CompWrite<UiText>,));
    type ResQuery = (ResRead<ActiveScreens>,);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _: [Commander<'_>; 2],
    ) {
        if self.timer.elapsed() > self.duration {
            self.timer = Instant::now();
            let fps = (self.recent as f32 / self.duration.as_secs_f32()).round() as u32;

            if let Some(hud) = res.0.get(id_of!(SCREEN_TYPES, "hud"))
                && let Some(text) = comp.1.0.get(hud.entities[2])
            {
                text.text = format!("{:.2} fps", fps);
            }

            self.recent = 0;
        }

        self.recent += 1;
    }
}

impl FrameRater {
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            timer: Instant::now(),
            recent: 0,
        }
    }
}
