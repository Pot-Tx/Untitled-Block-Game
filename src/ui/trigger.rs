use crate::components;
use crate::ecs::*;
use crate::game::InputState;
use crate::render::{Canvas, Viewports};
use crate::resources;
use crate::ui::{ActiveScreens, SCREEN_TYPES, UiRect, UiSprite};
use crate::util::Id;
use glam::Vec4;
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
    /// Returns whether `S` is one of the trigger systems of this element.
    #[inline]
    pub fn contains<S: 'static>(&self) -> bool {
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

impl ActiveElement {
    pub(super) fn triggers_for<S: 'static>(&self, triggers: &CompRead<Triggers>) -> bool {
        self.0
            .and_then(|id| triggers.get(id))
            .is_some_and(Triggers::contains::<S>)
    }
}

/// Maps the window cursor into the viewports every frame.
pub(super) struct CursorTracker;

impl System<2> for CursorTracker {
    type CompQueries = ((), ());
    type ResQuery = (ResRead<InputState>, ResRead<Canvas>, ResWrite<Viewports>);

    fn update(
        &mut self,
        _: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let input = res.0;
        res.2
            .track_cursor(res.1, input.cursor_grabbed, input.cursor_pos);
    }
}

/// The topmost UI element under the cursor, which the trigger systems read to
/// decide whether the pointer lies on one of their elements.
pub(super) struct UiPointer;

impl System<2> for UiPointer {
    type CompQueries = ((), (CompRead<UiRect>,));
    type ResQuery = (
        ResRead<ActiveScreens>,
        ResRead<Viewports>,
        ResWrite<ActiveElement>,
    );

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        // Only the topmost screen receives clicks and hovers, and inside it the
        // last element of the screen is the one in front.
        let mut hit = None;
        if let Some(top) = res.0.top()
            && let Some(cursor) = res.1.cursor(SCREEN_TYPES.get(top).viewport)
            && let Some(screen) = res.0.get(top)
        {
            hit = screen.entities.iter().rev().copied().find(|entity| {
                comp.1
                    .0
                    .get(*entity)
                    .is_some_and(|rect| rect.0.is_point_inside(cursor))
            });
        }

        res.2.0 = hit;
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
    active: Option<(Id, Vec4, f32)>,
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
    fn tint(color: Vec4, factor: f32) -> Vec4 {
        Vec4::new(
            color[0] * factor,
            color[1] * factor,
            color[2] * factor,
            color[3],
        )
    }
}

impl System<2> for Highlighter {
    type CompQueries = ((), (CompRead<Triggers>, CompWrite<UiSprite>));
    type ResQuery = (ResRead<ActiveElement>, ResRead<InputState>);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let (triggers, sprites) = comp.1;
        let (pointer, input) = res;

        // The element under the pointer, the factor its color is scaled with
        // and the color the factor applies to. The base color is only read from
        // the sprite when the element changes, so that a change of the factor
        // does not compound on the color that is already tinted.
        let target = pointer.0.and_then(|entity| {
            if !triggers.get(entity).is_some_and(Triggers::contains::<Self>) {
                return None;
            }

            let factor = if input.is_button_pressed(MouseButton::Left) {
                self.press
            } else {
                self.hover
            };
            let base = match self.active {
                Some((prev, base, _)) if prev == entity => base,
                _ => sprites.get(entity)?.color,
            };

            Some((entity, base, factor))
        });

        if target == self.active {
            return;
        }

        // The element that is no longer highlighted shows its own color again.
        if let Some((prev, base, _)) = self.active
            && let Some(sprite) = sprites.get(prev)
        {
            sprite.color = base;
        }

        // The element the pointer is on now is scaled with its factor.
        if let Some((entity, base, factor)) = target
            && let Some(sprite) = sprites.get(entity)
        {
            sprite.color = Self::tint(base, factor);
        }

        self.active = target;
    }
}
