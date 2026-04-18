use crate::ecs::*;
use crate::util::collection::Registry;
use crate::util::Id;
use glam::Vec2;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::sync::LazyLock;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};

pub static INPUT_MAP: LazyLock<Registry<Input>> = LazyLock::new(|| build_input_map());
pub static MOUSE_SENSITIVITY: LazyLock<f32> = LazyLock::new(|| PI / 64.0);

fn build_input_map() -> Registry<Input> {
    let mut input_map = Registry::new();

    let forward = Input {
        button: InputButton::Key(KeyCode::KeyW),
        input_type: InputType::Pressed,
    };
    let left = Input {
        button: InputButton::Key(KeyCode::KeyA),
        input_type: InputType::Pressed,
    };
    let backward = Input {
        button: InputButton::Key(KeyCode::KeyS),
        input_type: InputType::Pressed,
    };
    let right = Input {
        button: InputButton::Key(KeyCode::KeyD),
        input_type: InputType::Pressed,
    };
    let ascend = Input {
        button: InputButton::Key(KeyCode::Space),
        input_type: InputType::Pressed,
    };
    let descend = Input {
        button: InputButton::Key(KeyCode::ShiftLeft),
        input_type: InputType::Pressed,
    };

    input_map.register(0, forward);

    input_map.register(1, left);

    input_map.register(2, backward);

    input_map.register(3, right);

    input_map.register(4, ascend);

    input_map.register(5, descend);

    input_map
}

pub struct Input {
    pub button: InputButton,
    pub input_type: InputType,
}

pub enum InputButton {
    Key(KeyCode),
    Mouse(MouseButton),
}

pub enum InputType {
    Pressed,
    JustPressed,
    JustReleased,
}

#[derive(Default)]
pub struct InputState {
    pressed_keys: HashSet<KeyCode>,
    just_pressed_keys: HashSet<KeyCode>,
    just_released_keys: HashSet<KeyCode>,
    pub cursor_pos: Vec2,
    pub mouse_motion: Vec2,
    pressed_buttons: HashSet<MouseButton>,
    just_pressed_buttons: HashSet<MouseButton>,
    just_released_buttons: HashSet<MouseButton>,
}

impl Resource for InputState {}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn push_cursor_pos(&mut self, pos: PhysicalPosition<f64>) {
        self.cursor_pos = Vec2::new(pos.x as f32, pos.y as f32);
    }

    pub fn push_mouse_motion(&mut self, delta: (f64, f64)) {
        self.mouse_motion[0] += delta.0 as f32;
        self.mouse_motion[1] -= delta.1 as f32;
    }

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

    pub fn clear(&mut self) {
        self.just_pressed_keys.clear();
        self.just_released_keys.clear();
        self.just_pressed_buttons.clear();
        self.just_released_buttons.clear();
    }

    pub fn clear_mouse_motion(&mut self) {
        self.mouse_motion = Vec2::ZERO;
    }

    #[inline]
    pub fn is_action_present(&self, action_id: Id) -> bool {
        let input = INPUT_MAP.get(action_id);
        self.is_input_present(input)
    }

    #[inline]
    pub fn is_input_present(&self, input: &Input) -> bool {
        match input.button {
            InputButton::Key(key) => match input.input_type {
                InputType::Pressed => self.pressed_keys.contains(&key),
                InputType::JustPressed => self.just_pressed_keys.contains(&key),
                InputType::JustReleased => self.just_released_keys.contains(&key),
            },

            InputButton::Mouse(button) => match input.input_type {
                InputType::Pressed => self.pressed_buttons.contains(&button),
                InputType::JustPressed => self.just_pressed_buttons.contains(&button),
                InputType::JustReleased => self.just_released_buttons.contains(&button),
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

pub struct InputFlusher;

pub struct MouseMotionFlusher;

impl System for InputFlusher {
    type CompQuery = ();
    type ResQuery = ResWrite<InputState>;

    fn operate<'a>(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'a>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'a>,
    ) -> Option<Vec<Command>> {
        res.clear();

        None
    }
}

impl System for MouseMotionFlusher {
    type CompQuery = ();
    type ResQuery = ResWrite<InputState>;

    fn operate<'a>(
        &mut self,
        _: <Self::CompQuery as CompQuery>::Item<'a>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'a>,
    ) -> Option<Vec<Command>> {
        res.clear_mouse_motion();

        None
    }
}
