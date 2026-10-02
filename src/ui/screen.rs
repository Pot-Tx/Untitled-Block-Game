use crate::ecs::*;
use crate::game::{InputState, Paused, INPUT_MAP};
use crate::render::ViewPortAlignment;
use crate::resources;
use crate::ui::*;
use crate::util::collection::Registry;
use crate::util::Id;
use crate::util::OnceInit;
use crate::{by_name, id_of};
use anyhow::anyhow;
use ron::value::RawValue;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::HashMap;
use winit::event::MouseButton;

/// Every file of `assets/screens` names a screen, how it behaves and which
/// elements it contains; [`build_screens`] turns them into the registry the
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
pub(crate) fn build_screens(
    components: &ComponentManager,
    systems: &SystemManager,
) -> Registry<ScreenType> {
    Registry::load_rons_from("assets/screens")
        .expect("failed to load screens")
        .map(|raw| {
            ScreenType::from_raw(components, systems, raw)
                .expect("failed to create a screen from data")
        })
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
}

impl ScreenType {
    /// Builds a screen from its data, turning every element into the descriptor
    /// it is spawned with.
    fn from_raw(
        components: &ComponentManager,
        systems: &SystemManager,
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
fn resolve_triggers(systems: &SystemManager, names: &[String]) -> anyhow::Result<Vec<TypeId>> {
    names
        .iter()
        .map(|name| {
            systems
                .type_id_of(name)
                .ok_or_else(|| anyhow!("system {} is not registered", name))
        })
        .collect()
}

impl Screen {
    pub fn new() -> Self {
        Self::default()
    }
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
    pub fn open(&mut self, type_id: Id) -> Vec<Command> {
        if self.contains(type_id) {
            return Vec::new();
        }

        self.screens.insert(
            type_id,
            Screen {
                entities: Vec::new(),
            },
        );
        self.order.push(type_id);

        SCREEN_TYPES
            .get(type_id)
            .entities
            .iter()
            .enumerate()
            .map(|(priority, entity)| {
                Command::Spawn(entity.clone().with(ScreenTag {
                    screen: type_id,
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
pub(super) struct ScreenCollector;

impl System for ScreenCollector {
    type CompQuery = (CompRead<ScreenTag>,);
    type ResQuery = (ResWrite<ActiveScreens>,);

    fn update(&mut self, comp: Self::CompQuery, res: Self::ResQuery) -> Vec<Command> {
        let (screens,) = res.get();

        let mut collected: HashMap<Id, Vec<(u32, Id)>> = HashMap::new();
        let mut commands = Vec::new();

        // The elements of a screen arrive as tagged entities, which are sorted
        // by their priority so that the screen knows what to draw on top.
        for (entity, tag) in comp.iter() {
            match screens.contains(tag.screen) {
                true => collected
                    .entry(tag.screen)
                    .or_default()
                    .push((tag.priority, entity)),

                // The screen was closed before its elements were collected.
                false => commands.push(Command::Despawn(entity)),
            }

            // The tag is only needed while the screen is being built.
            commands.push(Command::Remove(entity, TypeId::of::<ScreenTag>()));
        }

        for (screen, mut entities) in collected {
            entities.sort();

            if let Some(instance) = screens.screens.get_mut(&screen) {
                instance
                    .entities
                    .extend(entities.into_iter().map(|(_, entity)| entity));
            }
        }

        commands
    }
}

/// Opens the pause screen and closes it again when escape is pressed.
pub(super) struct ScreenController;

impl System for ScreenController {
    type CompQuery = CompRead<Triggers>;
    type ResQuery = (
        ResRead<ActiveElement>,
        ResRead<InputState>,
        ResWrite<ActiveScreens>,
        ResWrite<Paused>,
    );

    fn update(&mut self, comp: Self::CompQuery, res: Self::ResQuery) -> Vec<Command> {
        let (active, input, screens, paused) = res.get();
        let triggered = active.0.is_some_and(|id| {
            comp.get(id).is_some_and(|tags| tags.is_for::<Self>())
        }) && input.is_button_just_released(MouseButton::Left);

        if triggered || input.is_input_present(by_name!(INPUT_MAP, "return")) {
            let commands = match screens.top() {
                Some(screen) if SCREEN_TYPES.get(screen).dismissible => screens.close_top(),

                _ => screens.open(id_of!(SCREEN_TYPES, "pause")),
            };

            paused.0 = screens.pauses();
            commands
        } else {
            Vec::new()
        }
    }
}
