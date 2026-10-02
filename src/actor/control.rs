use crate::actor::*;
use crate::{by_name, id_of};
use crate::components;
use crate::ecs::*;
use crate::game::*;
use crate::render::*;
use crate::util::bounding::{AABBGroup, Ray};
use crate::util::coord::{Direction, ICoord3};
use crate::util::Id;
use crate::world::{BLOCK_TYPES, Block, BlockPos, World};
use glam::Vec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use wgpu::{LoadOp, PrimitiveTopology};

components! {
    /// Marks the actor the player steers, together with what it is looking at
    /// and what an interaction would affect.
    #[derive(Clone, Copy)]
    pub struct PlayerControlled { pub selection: Option<SelectedItem> }: Cold, data;
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
    desc: RenderDescriptor<'static>,
    batch: RenderBatch<Transformation, BasicVertex, TransInst>,
    vertices: BufferVec<BasicVertex>,
    indices: BufferVec<u16>,
    instances: BufferVec<TransInst>,
}

impl System for PlayerController {
    type CompQuery = (
        CompRead<PlayerControlled>,
        CompWrite<Velocity>,
        CompRead<Rotation>,
        CompRead<Speed>,
        OptionalRead<Flight>,
        OptionalRead<Bound>,
    );
    type ResQuery = ResRead<InputState>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if res.cursor_grabbed {
            let mut dir = Vec3::ZERO;

            if res.is_input_present(by_name!(INPUT_MAP, "forward")) {
                dir.z += 1.0;
            }

            if res.is_input_present(by_name!(INPUT_MAP, "left")) {
                dir.x -= 1.0;
            }

            if res.is_input_present(by_name!(INPUT_MAP, "backward")) {
                dir.z -= 1.0;
            }

            if res.is_input_present(by_name!(INPUT_MAP, "right")) {
                dir.x += 1.0;
            }

            let mut speed = entry.4.0;
            match entry.5 {
                // Without flight, the player may only jump from the ground; the
                // speed is reduced in every other case, which keeps the player
                // from drifting sideways in the air.
                None => {
                    if let Some(bound) = entry.6 {
                        if let Some(p) = bound.contact[Axis::Y.idx()]
                            && !p
                        {
                            if res.is_input_present(by_name!(INPUT_MAP, "ascend")) {
                                entry.2.0.y = 0.75;
                            }
                        } else {
                            speed *= 0.0625;
                        }
                    }
                }

                Some(_) => {
                    if res.is_input_present(by_name!(INPUT_MAP, "ascend")) {
                        dir.y += 1.0;
                    }

                    if res.is_input_present(by_name!(INPUT_MAP, "descend")) {
                        dir.y -= 1.0;
                    }
                }
            }

            entry.2.accelerate(entry.3, &Speed(speed), dir);
        }

        None
    }
}

impl System for PlayerRotator {
    type CompQuery = (CompRead<PlayerControlled>, CompWrite<Rotation>);
    type ResQuery = ResRead<InputState>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if res.cursor_grabbed {
            entry.2.rotate(res.mouse_motion * *MOUSE_SENSITIVITY);
        }

        None
    }
}

impl System for Selector {
    type CompQuery = (
        CompWrite<PlayerControlled>,
        CompRead<Position>,
        CompRead<Rotation>,
    );
    type ResQuery = ResRead<World>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let ray = Ray {
            origin: entry.2.cur,
            direction: entry.3.direction(),
        };

        // The player can only reach a few blocks.
        entry.1.selection = ray.traverse(res, 8.0);

        None
    }
}

impl System for Interactor {
    type CompQuery = (CompRead<PlayerControlled>, CompRead<Position>, CompRead<Bound>);
    type ResQuery = (ResRead<InputState>, ResWrite<World>);

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if !res.0.cursor_grabbed {
            return None;
        }

        if let Some(selected) = entry.1.selection {
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
                    // Placing a block needs the face the ray entered through,
                    // and it has to be free of the player's own collision box.
                    SelectedItem::Block { pos, face, .. } => {
                        let block = Block::default_of(id_of!(BLOCK_TYPES, "bricks"));
                        let bound = entry.3.translate(entry.2);

                        if block
                            .bounds(pos.step(face))
                            .into_iter()
                            .all(|b| !bound.intersects_with(b))
                        {
                            res.1.set_block(pos.step(face), block);
                        }
                    }

                    SelectedItem::Actor { .. } => (),
                }
            }
        }

        None
    }
}

impl System for SelectionRenderer {
    type CompQuery = CompRead<PlayerControlled>;
    type ResQuery = (ResWrite<Option<Frame>>, ResRead<Camera>, ResRead<Canvas>);

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
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

        if let Some(frame) = res.0 {
            frame.render(&self.desc, |mut pass| {
                self.batch.begin(&mut pass);
                self.batch.push(&mut pass, &res.1.transform);
                self.batch.draw(&mut pass, self);
            });
        }

        None
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
            desc: RenderDescriptor {
                name: "selection",
                color_load: LoadOp::Load,
                depth_load: LoadOp::Load,
            },
            batch: RenderBatch::new(
                canvas,
                &RenderBatchConfig {
                    name: "selection",
                    shader: "selection",
                    translucent: false,
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
