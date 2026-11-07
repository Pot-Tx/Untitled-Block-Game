use crate::ecs::*;
use crate::game::ASSET_OPTIONS;
use crate::game::client::WINDOW;
use crate::ui::ActiveScreens;
use crate::util::Id;
use crate::util::OnceInit;
use crate::util::collection::Registry;
use glam::Vec2;
use log::error;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufReader;
use std::sync::LazyLock;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::CursorGrabMode;

/// The actions of the game, in the order of their ids.
pub(crate) static INPUT_MAP: OnceInit<Registry<Input>> = OnceInit::new();
/// Rotation per pixel of mouse motion, in radians.
pub(crate) static MOUSE_SENSITIVITY: LazyLock<f32> = LazyLock::new(|| PI / 256.0);

/// The bindings are the ones of `settings/controls.ron`, which belongs to the
/// game rather than to a mod, so that a mod cannot take the controls away.
pub(super) fn init_input_map() -> anyhow::Result<()> {
    let file = File::open("settings/controls.ron")?;
    let map: HashMap<String, Input> = ASSET_OPTIONS.from_reader(BufReader::new(file))?;
    let mut reg = Registry::new();

    for (id, (name, input)) in map.into_iter().enumerate() {
        reg.register(id as Id, name, input);
    }

    INPUT_MAP.init(reg);
    Ok(())
}

/// One action of the game and the input it reacts to.
#[derive(Serialize, Deserialize)]
pub struct Input {
    pub button: InputButton,
    pub action: InputAction,
}

/// The key or mouse button an action is bound to.
#[derive(Serialize, Deserialize)]
pub enum InputButton {
    Key(KeyCode),
    Mouse(MouseButton),
}

/// The kind of press an action reacts to.
#[derive(Serialize, Deserialize)]
pub enum InputAction {
    /// The action is present while the button is held down.
    Pressed,
    /// The action is present on the frame the button goes down.
    JustPressed,
    /// The action is present on the frame the button comes up.
    JustReleased,
}

/// The state of the keyboard, the mouse and the cursor.
#[derive(Default)]
pub struct InputState {
    pressed_keys: HashSet<KeyCode>,
    just_pressed_keys: HashSet<KeyCode>,
    just_released_keys: HashSet<KeyCode>,
    pub cursor_grabbed: bool,
    pub cursor_pos: Vec2,
    pub mouse_motion: Vec2,
    pub mouse_scroll: Vec2,
    pressed_buttons: HashSet<MouseButton>,
    just_pressed_buttons: HashSet<MouseButton>,
    just_released_buttons: HashSet<MouseButton>,
}

impl Resource for InputState {}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a key going down or coming up.
    pub fn push_key_event(&mut self, event: KeyEvent) {
        if let PhysicalKey::Code(key) = event.physical_key {
            match event.state {
                ElementState::Pressed => {
                    self.pressed_keys.insert(key);
                    self.just_pressed_keys.insert(key);
                }

                ElementState::Released => {
                    self.pressed_keys.remove(&key);
                    self.just_released_keys.insert(key);
                }
            }
        }
    }

    /// Records the position of the cursor, in window coordinates.
    pub fn push_cursor_pos(&mut self, pos: PhysicalPosition<f64>) {
        self.cursor_pos = Vec2::new(pos.x as f32, pos.y as f32);
    }

    /// Adds a mouse motion, whose `y` is inverted because the screen grows
    /// downwards.
    pub fn push_mouse_motion(&mut self, delta: (f64, f64)) {
        self.mouse_motion[0] += delta.0 as f32;
        self.mouse_motion[1] -= delta.1 as f32;
    }

    pub fn push_mouse_scroll(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(dx, dy) => {
                self.mouse_scroll.x += dx;
                self.mouse_scroll.y += dy;
            }

            MouseScrollDelta::PixelDelta(delta) => {
                self.mouse_scroll.x += delta.x as f32;
                self.mouse_scroll.y += delta.y as f32;
            }
        }
    }

    /// Records a mouse button going down or coming up.
    pub fn push_button_event(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_buttons.insert(button);
                self.just_pressed_buttons.insert(button);
            }

            ElementState::Released => {
                self.pressed_buttons.remove(&button);
                self.just_released_buttons.insert(button);
            }
        }
    }

    /// Clears the state that only lasts one frame.
    pub fn clear(&mut self) {
        self.mouse_motion = Vec2::ZERO;
        self.mouse_scroll = Vec2::ZERO;
        self.just_pressed_keys.clear();
        self.just_released_keys.clear();
        self.just_pressed_buttons.clear();
        self.just_released_buttons.clear();
    }

    /// Returns whether `input` is currently present.
    #[inline]
    pub fn is_input_present(&self, input: &Input) -> bool {
        match input.button {
            InputButton::Key(key) => match input.action {
                InputAction::Pressed => self.pressed_keys.contains(&key),
                InputAction::JustPressed => self.just_pressed_keys.contains(&key),
                InputAction::JustReleased => self.just_released_keys.contains(&key),
            },

            InputButton::Mouse(button) => match input.action {
                InputAction::Pressed => self.pressed_buttons.contains(&button),
                InputAction::JustPressed => self.just_pressed_buttons.contains(&button),
                InputAction::JustReleased => self.just_released_buttons.contains(&button),
            },
        }
    }

    #[inline]
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    #[inline]
    pub fn is_key_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed_keys.contains(&key)
    }

    #[inline]
    pub fn is_key_just_released(&self, key: KeyCode) -> bool {
        self.just_released_keys.contains(&key)
    }

    #[inline]
    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.pressed_buttons.contains(&button)
    }

    #[inline]
    pub fn is_button_just_pressed(&self, button: MouseButton) -> bool {
        self.just_pressed_buttons.contains(&button)
    }

    #[inline]
    pub fn is_button_just_released(&self, button: MouseButton) -> bool {
        self.just_released_buttons.contains(&button)
    }
}

/// Grabs the cursor while the game is playing and releases it while a menu is
/// open.
#[derive(Default)]
pub(super) struct CursorApplier {
    /// The grab state that was last applied to the window.
    grabbed: bool,
}

impl CursorApplier {
    /// Creates the applier with the cursor ungrabbed.
    pub fn new() -> Self {
        Self::default()
    }
}

impl System<2> for CursorApplier {
    type CompQueries = ((), ());
    type ResQuery = (ResRead<ActiveScreens>, ResWrite<InputState>);

    fn update(
        &mut self,
        _: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        let wanted = !res.0.menu();

        // A menu releases the cursor so that the player can click on it.
        if res.1.cursor_grabbed != wanted {
            res.1.cursor_grabbed = wanted;
        }

        if res.1.cursor_grabbed == self.grabbed {
            return;
        }

        self.grabbed = res.1.cursor_grabbed;

        if self.grabbed {
            // Locking the cursor keeps it inside the window and hides it, which
            // is what makes the mouse motion the only source of rotation.
            let size = WINDOW.inner_size();
            if let Err(e) =
                WINDOW.set_cursor_position(PhysicalPosition::new(size.width / 2, size.height / 2))
            {
                error!("failed to centre cursor: {}", e);
            }

            if let Err(e) = WINDOW.set_cursor_grab(CursorGrabMode::Locked) {
                error!("failed to grab cursor: {}", e);
            }

            WINDOW.set_cursor_visible(false);
        } else {
            if let Err(e) = WINDOW.set_cursor_grab(CursorGrabMode::None) {
                error!("failed to release cursor: {}", e);
            }

            WINDOW.set_cursor_visible(true);
        }
    }
}
