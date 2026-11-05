use crate::actor::*;
use crate::by_name;
use crate::components;
use crate::ecs::*;
use crate::game::*;
use crate::render::*;
use crate::util::Id;
use crate::util::bounding::{AABBGroup, Ray};
use crate::util::coord::{Direction, ICoord3};
use crate::world::{Block, BlockPos, World};
use glam::Vec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use wgpu::{Face, LoadOp, PrimitiveTopology};

components! {
    /// Marks the actor the player steers, together with what it is looking at
    /// and what an interaction would affect.
    #[derive(Clone, Copy)]
    pub struct PlayerControlled { pub selection: Option<SelectedItem> }: Cold, asset, save;
}

/// The selection is derived from the world every tick, so it is not written; an
/// actor read from data starts out looking at nothing.
impl Serialize for PlayerControlled {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit()
    }
}

impl<'de> Deserialize<'de> for PlayerControlled {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let () = Deserialize::deserialize(deserializer)?;

        Ok(Self { selection: None })
    }
}

/// The thing the player points at.
#[derive(Clone, Copy, Debug)]
pub enum SelectedItem {
    /// A block, with the face the ray entered it through.
    Block {
        pos: BlockPos,
        block: Block,
        face: Direction,
    },
    /// An actor, with its position and its collision box.
    Actor {
        entity: Id,
        pos: Vec3,
        bound: AABB<Vec3>,
    },
}

impl SelectedItem {
    /// The wireframe that outlines the selection.
    fn mesh(&self) -> Mesh<BasicVertex> {
        match self {
            Self::Block { block, .. } => {
                let bound = block
                    .bounds(BlockPos::ZERO)
                    .merge()
                    .expect("selected block should have bounds");
                Mesh::<BasicVertex>::frame(bound.min, bound.max)
            }
            _ => Mesh::new(),
        }
    }

    /// The position the selection is drawn at.
    fn inst(&self) -> TransInst {
        TransInst {
            pos: match self {
                Self::Block { pos, .. } => pos.as_vec3(),
                Self::Actor { pos, .. } => *pos,
            },
        }
    }
}

/// Turns the movement and jump actions into velocity.
pub(super) struct PlayerController;

/// Turns the mouse motion into the rotation of the player.
pub(super) struct PlayerRotator;

/// Shoots a ray through the world and stores what the player looks at.
pub(super) struct Selector;

/// Breaks and places blocks when the attack and interact actions are pressed.
pub(super) struct Interactor;

/// Draws the outline of the current selection.
pub(super) struct SelectionRenderer {
    batch: RenderBatch<CameraTransform, BasicVertex, TransInst, ColorDraw<1>, DepthDraw>,
    vertices: BufferVec<BasicVertex>,
    indices: BufferVec<u16>,
    instances: BufferVec<TransInst>,
}

impl System<1> for PlayerController {
    type CompQueries = ((
        CompRead<PlayerControlled>,
        CompWrite<Velocity>,
        CompRead<Rotation>,
        CompRead<Speed>,
        OptionalRead<Flight>,
        OptionalRead<Bound>,
    ),);
    type ResQuery = (ResRead<InputState>,);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        let (inputs,) = res;
        for entry in comp.0.iter() {
            if inputs.cursor_grabbed {
                let mut dir = Vec3::ZERO;

                if inputs.is_input_present(by_name!(INPUT_MAP, "forward")) {
                    dir.z -= 1.0;
                }

                if inputs.is_input_present(by_name!(INPUT_MAP, "left")) {
                    dir.x -= 1.0;
                }

                if inputs.is_input_present(by_name!(INPUT_MAP, "backward")) {
                    dir.z += 1.0;
                }

                if inputs.is_input_present(by_name!(INPUT_MAP, "right")) {
                    dir.x += 1.0;
                }

                let (_, _, vel, rot, Speed { walk, jump }, flight, bound) = entry;

                let mut walk = *walk;
                match flight {
                    // Without flight, the player may only jump from the ground;
                    // the speed is reduced in every other case, which keeps the
                    // player from drifting sideways in the air.
                    None => {
                        if let Some(bound) = bound {
                            if let Some(p) = bound.contact[Axis::Y.idx()]
                                && !p
                            {
                                if inputs.is_input_present(by_name!(INPUT_MAP, "ascend")) {
                                    vel.0.y = *jump;
                                }
                            } else {
                                walk *= 0.0625;
                            }
                        }
                    }

                    Some(_) => {
                        if inputs.is_input_present(by_name!(INPUT_MAP, "ascend")) {
                            dir.y += 1.0;
                        }

                        if inputs.is_input_present(by_name!(INPUT_MAP, "descend")) {
                            dir.y -= 1.0;
                        }
                    }
                }

                vel.accelerate(walk, dir.rotate_y(-rot.0[0]));
            }
        }
    }
}

impl System<2> for PlayerRotator {
    type CompQueries = ((CompRead<PlayerControlled>, CompWrite<Rotation>), ());
    type ResQuery = (ResRead<InputState>,);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        for entry in comp.0.iter() {
            if res.0.cursor_grabbed {
                entry.2.rotate(res.0.mouse_motion * *MOUSE_SENSITIVITY);
            }
        }
    }
}

impl System<1> for Selector {
    type CompQueries = ((
        CompWrite<PlayerControlled>,
        CompRead<Position>,
        CompRead<Rotation>,
        OptionalRead<Eye>,
    ),);
    type ResQuery = (ResRead<World>,);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 1],
    ) {
        for entry in comp.0.iter() {
            // The player looks from its eyes, which are above its feet.
            let ray = Ray {
                origin: entry.2.cur + Vec3::Y * entry.4.map_or(0.0, |eye| eye.0),
                direction: entry.3.direction(),
            };

            // The player can only reach a few blocks.
            entry.1.selection = ray.traverse(res.0, 8.0);
        }
    }
}

impl System<2> for Interactor {
    type CompQueries = (
        (
            CompRead<PlayerControlled>,
            CompRead<Position>,
            OptionalRead<Bound>,
            CompRead<Inventory>,
        ),
        (),
    );
    type ResQuery = (ResRead<InputState>, ResWrite<World>);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        if !res.0.cursor_grabbed {
            return;
        }

        for (_, controlled, position, bound, inventory) in comp.0.iter() {
            if let Some(selected) = controlled.selection {
                if res.0.is_input_present(by_name!(INPUT_MAP, "attack")) {
                    match selected {
                        // Breaking a block replaces it with air.
                        SelectedItem::Block { pos, .. } => {
                            res.1.set_block(pos, Block::air());
                        }

                        SelectedItem::Actor { .. } => (),
                    }
                }

                if res.0.is_input_present(by_name!(INPUT_MAP, "interact")) {
                    match selected {
                        // Placing a block needs the face the ray entered
                        // through, and it has to be free of the player's own
                        // collision box.
                        SelectedItem::Block { pos, face, .. } => {
                            let current = inventory.get(inventory.cur);
                            let bound = bound.map(|b| b.translate(position));

                            if let Some(block) = current
                                && bound.is_none_or(|b| {
                                    block
                                        .bounds(pos.step(face))
                                        .into_iter()
                                        .all(|b1| !b.intersects_with(b1))
                                })
                            {
                                res.1.set_block(pos.step(face), block);
                            }
                        }

                        SelectedItem::Actor { .. } => (),
                    }
                }
            }
        }
    }
}

impl System<2> for SelectionRenderer {
    type CompQueries = ((CompRead<PlayerControlled>,), ());
    type ResQuery = (ResWrite<FrameSlot>, ResRead<Camera>, ResRead<Canvas>);

    fn update(
        &mut self,
        comp: Self::CompQueries,
        res: <Self::ResQuery as ResQuery>::Item<'_>,
        _com: [Commander<'_>; 2],
    ) {
        for entry in comp.0.iter() {
            let canvas = res.2;

            match entry.1.selection {
                Some(item) => {
                    let mesh = item.mesh();
                    self.vertices.set_content(canvas, &mesh.vertices);
                    self.indices.set_content(canvas, &mesh.indices);
                    self.instances.set_content(canvas, &[item.inst()]);
                }

                None => {
                    self.vertices.set_content(canvas, &[]);
                    self.indices.set_content(canvas, &[]);
                    self.instances.set_content(canvas, &[]);
                }
            }

            if let Some(frame) = res.0.as_mut() {
                let colors = [ColorTarget {
                    view: &canvas.color_view,
                    load: LoadOp::Load,
                }];
                let depth = DepthTarget {
                    view: &canvas.depth_view,
                    load: LoadOp::Load,
                };

                frame.render::<ColorDraw<1>, DepthDraw>("selection", colors, depth, |mut pass| {
                    self.batch.begin(&mut pass);
                    self.batch.push(&mut pass, &res.1.transform);
                    self.batch.draw(&mut pass, self);
                });
            }
        }
    }
}

impl Render<BasicVertex, TransInst> for SelectionRenderer {
    fn rendered(&self) -> Vec<RenderItem<'_, BasicVertex, TransInst>> {
        let mut items = Vec::new();

        if let Ok(item) = RenderItem::new(&self.vertices, &self.indices, &self.instances) {
            items.push(item);
        }

        items
    }
}

impl SelectionRenderer {
    pub fn new(canvas: &Canvas) -> Self {
        Self {
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "selection",
                    shader: "selection",
                    color_formats: [Canvas::SCENE_COLOR_FORMAT],
                    depth_format: Canvas::SCENE_DEPTH_FORMAT,
                    cull: Some(Face::Back),
                    blend: None,
                    secondary_blend: None,
                    topology: PrimitiveTopology::LineList,
                    depth_write: false,
                },
            ),
            vertices: BufferVec::vertex(canvas, "selection_vertex", BufferInit::Size(0)),
            indices: BufferVec::index(canvas, "selection_index", BufferInit::Size(0)),
            instances: BufferVec::vertex(canvas, "selection_instance", BufferInit::Size(0)),
        }
    }
}
