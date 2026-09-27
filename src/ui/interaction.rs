use crate::components;
use crate::ecs::*;
use crate::game::InputState;
use crate::render::{Canvas, Viewports};
use crate::ui::{ActiveScreens, UiRect, SCREEN_TYPES};
use crate::util::Id;
use std::sync::Arc;
use winit::event::MouseButton;

/// A click handler, which receives the world it belongs to and the entity that
/// was clicked and returns the commands the click produced.
pub type Handler =
    Arc<dyn Fn(&ComponentManager, &ResourceManager, Id) -> Vec<Command> + Send + Sync>;

components! {
    /// Makes an element clickable, running the handler when it is clicked.
    #[derive(Clone)]
    pub struct OnClick(Handler): Cold;
}

/// Maps the window cursor into the viewports every frame.
pub struct CursorTracker;

impl System for CursorTracker {
    type CompQuery = ();
    type ResQuery = (ResRead<InputState>, ResRead<Canvas>, ResWrite<Viewports>);

    fn operate(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let input = res.0;
        res.2
            .track_cursor(res.1, input.cursor_grabbed, input.cursor_pos);

        None
    }
}

/// The element the pointer is currently pressing, together with the systems
/// that hit test the UI.
#[derive(Default)]
pub struct UiPointer {
    /// The element that was pressed down on, which only counts as a click when
    /// the button is released over the same element.
    active: Option<Id>,
}

impl UiPointer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl System for UiPointer {
    type CompQuery = (CompRead<UiRect>, OptionalRead<OnClick>);
    type ResQuery = (
        ResRead<InputState>,
        ResRead<ActiveScreens>,
        ResRead<Viewports>,
    );

    fn update(
        &mut self,
        entities: &EntityManager,
        resources: &ResourceManager,
    ) -> Result<Vec<Command>, QueryError> {
        let input = resources.try_get::<InputState>()?;
        let screens = resources.try_get::<ActiveScreens>()?;
        let viewports = resources.try_get::<Viewports>()?;
        let rects = entities.components.try_get::<UiRect>()?;
        let mut hit = None;

        // Only the topmost screen receives clicks, and inside it the last
        // element of the screen is the one in front.
        if let Some(top) = screens.top()
            && let Some(cursor) = viewports.cursor(SCREEN_TYPES.get(top).viewport)
            && let Some(screen) = screens.get(top)
        {
            for entity in screen.entities.iter().rev() {
                if let Some(rect) = rects.get::<UiRect>(*entity)
                    && rect.rect.is_point_inside(cursor)
                {
                    hit = Some(*entity);
                    break;
                }
            }
        }

        let hovered = hit;

        if let Some(target) = hovered
            && input.is_button_just_pressed(MouseButton::Left)
        {
            self.active = Some(target);
        }

        if input.is_button_just_released(MouseButton::Left)
            && let Some(target) = self.active.take()
            && hovered == Some(target)
            && let Some(click) = entities
                .components
                .try_get::<OnClick>()?
                .get::<OnClick>(target)
        {
            return Ok(click.0(&entities.components, resources, target));
        }

        Ok(Vec::new())
    }
}
