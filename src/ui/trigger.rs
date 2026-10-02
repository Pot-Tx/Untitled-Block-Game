use crate::components;
use crate::ecs::*;
use crate::game::InputState;
use crate::render::{Canvas, Viewports};
use crate::resources;
use crate::ui::{ActiveScreens, UiRect, UiSprite, SCREEN_TYPES};
use crate::util::Id;
use std::any::TypeId;
use winit::event::MouseButton;

components! {
    /// The systems that handle the triggers of a UI element.
    ///
    /// A trigger system compares this against its own type id, which keeps the
    /// behavior of an element in the system that executes it instead of in a
    /// callback that bypasses the system's declared access.
    #[derive(Clone)]
    pub struct Triggers(Vec<TypeId>): Cold;
}

impl Triggers {
    /// Returns whether any of the tagged systems is `S`.
    pub fn is_for<S: 'static>(&self) -> bool {
        self.0.contains(&TypeId::of::<S>())
    }
}

resources! {
    /// The element the pointer currently lies on, or `None` while it lies on
    /// nothing.
    ///
    /// It is written by [`UiPointer`] and read by the systems that handle the
    /// triggers of an element, so that a trigger does not have to hit test the
    /// screens itself.
    pub struct ActiveElement(Option<Id>);
}

/// Maps the window cursor into the viewports every frame.
pub(super) struct CursorTracker;

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

/// The topmost UI element under the cursor, which the trigger systems read to
/// decide whether the pointer lies on one of their elements.
pub(super) struct UiPointer;

impl System for UiPointer {
    type CompQuery = CompRead<UiRect>;
    type ResQuery = (ResRead<ActiveScreens>, ResRead<Viewports>, ResWrite<ActiveElement>);

    fn update(
        &mut self,
        comp: Self::CompQuery,
        res: Self::ResQuery,
    ) -> Vec<Command> {
        let (screens, viewports, active) = res.get();

        // Only the topmost screen receives clicks and hovers, and inside it the
        // last element of the screen is the one in front.
        let mut hit = None;
        if let Some(top) = screens.top()
            && let Some(cursor) = viewports.cursor(SCREEN_TYPES.get(top).viewport)
            && let Some(screen) = screens.get(top)
        {
            hit = screen.entities.iter().rev().copied().find(|entity| {
                comp.get(*entity)
                    .is_some_and(|rect| rect.0.is_point_inside(cursor))
            });
        }

        active.0 = hit;

        Vec::new()
    }
}

/// Tints the color of the UI elements whose [`Triggers`] name it, while the
/// pointer is on one of them.
pub(super) struct Highlighter {
    /// The factors the color is scaled with while a highlighted element is
    /// hovered, and while a mouse button is held down on it.
    hover: f32,
    press: f32,
    /// The element whose color is currently scaled, with the color it had
    /// before and the factor that was applied to it, so that the base color can
    /// be restored when the pointer leaves.
    active: Option<(Id, [f32; 4], f32)>,
}

impl Highlighter {
    pub fn new(hover: f32, press: f32) -> Self {
        Self {
            hover,
            press,
            active: None,
        }
    }

    /// Scales the color channels of `color` by `factor`, keeping its alpha so
    /// that highlighting does not change the transparency of the element.
    fn tint(color: [f32; 4], factor: f32) -> [f32; 4] {
        [
            color[0] * factor,
            color[1] * factor,
            color[2] * factor,
            color[3],
        ]
    }
}

impl System for Highlighter {
    type CompQuery = (CompRead<Triggers>, CompWrite<UiSprite>);
    type ResQuery = (ResRead<ActiveElement>, ResRead<InputState>);

    fn update(&mut self, comp: Self::CompQuery, res: Self::ResQuery) -> Vec<Command> {
        let (pointer, input) = res.get();

        // The element under the pointer, the factor its color is scaled with
        // and the color the factor applies to. The base color is only read from
        // the sprite when the element changes, so that a change of the factor
        // does not compound on the color that is already tinted.
        let target = pointer.0.and_then(|entity| {
            if !comp.0.get(entity).is_some_and(Triggers::is_for::<Self>) {
                return None;
            }

            let factor = if input.is_button_pressed(MouseButton::Left) {
                self.press
            } else {
                self.hover
            };
            let base = match self.active {
                Some((prev, base, _)) if prev == entity => base,
                _ => comp.1.get(entity)?.color,
            };

            Some((entity, base, factor))
        });

        if target == self.active {
            return Vec::new();
        }

        // The element that is no longer highlighted shows its own color again.
        if let Some((prev, base, _)) = self.active
            && let Some(sprite) = comp.1.get(prev)
        {
            sprite.color = base;
        }

        // The element the pointer is on now is scaled with its factor.
        if let Some((entity, base, factor)) = target
            && let Some(sprite) = comp.1.get(entity)
        {
            sprite.color = Self::tint(base, factor);
        }

        self.active = target;

        Vec::new()
    }
}
