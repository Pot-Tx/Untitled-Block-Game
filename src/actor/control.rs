use crate::actor::*;
use crate::components;
use crate::ecs::*;
use crate::game::*;
use crate::render::*;
use crate::util::bounding::{AABBGroup, Ray};
use crate::util::coord::{Direction, ICoord3};
use crate::util::Id;
use crate::world::{Block, BlockPos, World};
use glam::Vec3;
use wgpu::{LoadOp, PrimitiveTopology};

components! {
    /// Marks the actor the player steers.
    #[derive(Clone, Copy)]
    pub struct PlayerControlled: Cold;

    /// What the player is looking at, and what an interaction would affect.
    #[derive(Clone, Copy)]
    pub struct Selection(Option<SelectedItem>): Cold;
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
pub struct PlayerController;

/// Turns the mouse motion into the rotation of the player.
pub struct PlayerRotator;

/// Shoots a ray through the world and stores what the player looks at.
pub struct Selector;

/// Breaks and places blocks when the attack and interact actions are pressed.
pub struct Interactor;

/// Draws the outline of the current selection.
pub struct SelectionRenderer {
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
        OptionalRead<Contact>,
    );
    type ResQuery = ResRead<InputState>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if res.cursor_grabbed {
            let mut dir = Vec3::ZERO;

            // The ids are the entries of `INPUT_MAP`: 1 forward, 2 left,
            // 3 backward, 4 right, 5 ascend, 6 descend, 7 attack and
            // 8 interact.
            if res.is_action_present(1) {
                dir.z += 1.0;
            }

            if res.is_action_present(2) {
                dir.x -= 1.0;
            }

            if res.is_action_present(3) {
                dir.z -= 1.0;
            }

            if res.is_action_present(4) {
                dir.x += 1.0;
            }

            let mut speed = entry.4.0;
            match entry.5 {
                // Without flight, the player may only jump from the ground; the
                // speed is reduced in every other case, which keeps the player
                // from drifting sideways in the air.
                None => {
                    if let Some(contact) = entry.6 {
                        if let Some(p) = contact.0[Axis::Y.idx()]
                            && !p
                        {
                            if res.is_action_present(5) {
                                entry.2.0.y = 0.75;
                            }
                        } else {
                            speed *= 0.0625;
                        }
                    }
                }

                Some(_) => {
                    if res.is_action_present(5) {
                        dir.y += 1.0;
                    }

                    if res.is_action_present(6) {
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
    type CompQuery = (CompWrite<Selection>, CompRead<Position>, CompRead<Rotation>);
    type ResQuery = ResRead<World>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let ray = Ray {
            origin: entry.2.0,
            direction: entry.3.direction(),
        };

        // The player can only reach a few blocks.
        entry.1.0 = ray.traverse(res, 8.0);

        None
    }
}

impl System for Interactor {
    type CompQuery = (
        CompRead<PlayerControlled>,
        CompRead<Selection>,
        CompRead<Position>,
        CompRead<Bound>,
    );
    type ResQuery = (ResRead<InputState>, ResWrite<World>);

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if !res.0.cursor_grabbed {
            return None;
        }

        if let Some(selected) = entry.2.0 {
            if res.0.is_action_present(7) {
                match selected {
                    // Breaking a block replaces it with air.
                    SelectedItem::Block { pos, .. } => {
                        res.1.set_block(pos, Block::air());
                    }

                    SelectedItem::Actor { .. } => (),
                }
            }

            if res.0.is_action_present(8) {
                match selected {
                    // Placing a block needs the face the ray entered through,
                    // and it has to be free of the player's own collision box.
                    SelectedItem::Block { pos, face, .. } => {
                        let block = Block::default_of(1);
                        let bound = entry.4.translate(entry.3);

                        if block
                            .bounds(pos.step(face))
                            .into_iter()
                            .all(|b| !bound.intersects_with(b))
                        {
                            res.1.set_block(pos.step(face), Block::default_of(1));
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
    type CompQuery = CompRead<Selection>;
    type ResQuery = (ResWrite<Option<Frame>>, ResRead<Camera>, ResRead<Canvas>);

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let canvas = res.2;

        match entry.1.0 {
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
