use crate::actor::*;
use crate::components;
use crate::ecs::*;
use crate::game::*;
use crate::render::*;
use crate::util::bounding::{AABBGroup, Ray};
use crate::util::coord::{Direction, ICoord3};
use crate::util::Id;
use crate::world::{Block, BlockPos, Generate, World};
use glam::Vec3;
use std::marker::PhantomData;
use wgpu::{LoadOp, PrimitiveTopology};

components! {
    pub struct PlayerControlled: Cold;
}

pub struct Selection {
    item: SelectedItem,
    geometry: Geometry<BasicVertex>,
    instance: Instances<TransInst>,
}

#[derive(Debug)]
pub enum SelectedItem {
    Block {
        pos: BlockPos,
        block: Block,
        face: Direction,
    },
    Actor {
        entity: Id,
        pos: Vec3,
        bound: AABB<Vec3>,
    },
}

impl Component for Option<Selection> {
    const STORAGE_TYPE: StorageType = StorageType::Cold;
}

impl SelectedItem {
    fn update(&self, other: &Self) -> (bool, bool) {
        match (self, other) {
            (
                Self::Block { pos, block, .. },
                Self::Block { pos: pos1, block: block1, .. },
            )
                => (block != block1, pos != pos1),
            
            (
                Self::Actor { entity, .. },
                Self::Actor { entity: entity1, .. },
            )
                => (entity != entity1, true),
            
            _ => (true, true),
        }
    }
    
    fn mesh(&self) -> Mesh<BasicVertex> {
        match self {
            Self::Block { block, .. } => {
                let bound = block
                    .bounds()
                    .merge()
                    .expect("Selected Block doesn't have bounds");
                Mesh::<BasicVertex>::frame(bound.min, bound.max)
            }
            _ => Mesh::new(),
        }
    }

    fn inst(&self) -> TransInst {
        TransInst {
            pos: match self {
                Self::Block { pos, .. } => pos.as_vec3(),
                Self::Actor { pos, .. } => *pos,
            },
        }
    }
}

impl Render<BasicVertex, TransInst> for Selection {
    fn rendered(&self) -> Vec<RenderItem<'_, BasicVertex, TransInst>> {
        vec![RenderItem {
            geometry: &self.geometry,
            instances: &self.instance,
        }]
    }
}

pub struct PlayerController;

pub struct PlayerRotator;

pub struct Selector<G: Generate>(pub PhantomData<G>);

pub struct Interactor<G: Generate>(pub PhantomData<G>);

pub struct SelectionRenderer {
    desc: RenderDescriptor<'static>,
    batch: RenderBatch<Transformation, BasicVertex, TransInst>,
}

impl System for PlayerController {
    type CompQuery = (
        CompRead<PlayerControlled>,
        CompWrite<Velocity>,
        CompRead<Rotation>,
        CompRead<Speed>,
    );
    type ResQuery = ResRead<InputState>;

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if res.cursor_grabbed {
            let mut dir = Vec3::ZERO;
            
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
            
            if res.is_action_present(5) {
                dir.y += 1.0;
            }
            
            if res.is_action_present(6) {
                dir.y -= 1.0;
            }
            
            entry.2.accelerate(entry.3, entry.4, dir);
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

impl<G: Generate> System for Selector<G> {
    type CompQuery = (
        CompWrite<Option<Selection>>,
        CompRead<Position>,
        CompRead<Rotation>,
    );
    type ResQuery = (ResRead<Canvas>, ResRead<World<G>>);

    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        let ray = Ray {
            origin: entry.2.0,
            direction: entry.3.direction(),
        };
        
        if let Some(item) = ray.traverse(res.1, 8.0) {
            if let Some(selection) = entry.1 {
                let (g, i) = selection.item.update(&item);
                if g {
                    selection.geometry = item.mesh().geometry(res.0, "selection");
                }
                if i {
                    selection.instance = [item.inst()].instances(res.0, "selection");
                }
                
                selection.item = item;
            } else {
                let geometry = item.mesh().geometry(res.0, "selection");
                let instance = [item.inst()].instances(res.0, "selection");
                
                entry.1.replace(Selection { item, geometry, instance });
            }
        } else {
            entry.1.take();
        }

        None
    }
}

impl<G: Generate> System for Interactor<G> {
    type CompQuery = (CompRead<PlayerControlled>, CompRead<Option<Selection>>);
    type ResQuery = (ResRead<InputState>, ResWrite<World<G>>);
    
    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if let Some(selection) = entry.2 {
            if res.0.is_action_present(7) {
                match selection.item {
                    SelectedItem::Block { pos, .. } => {
                        res.1.set_block(pos, Block::air());
                    }
                    
                    SelectedItem::Actor { .. } => (),
                }
            }
            
            if res.0.is_action_present(8) {
                match selection.item {
                    SelectedItem::Block { pos, face, .. } => {
                        res.1.set_block(pos.step(face), Block::default_of(1));
                    }
                    
                    SelectedItem::Actor { .. } => (),
                }
            }
        }
        
        None
    }
}

impl System for SelectionRenderer {
    type CompQuery = CompRead<Option<Selection>>;
    type ResQuery = (ResWrite<Option<Frame>>, ResRead<Camera>);
    
    fn operate(
        &mut self,
        entry: <Self::CompQuery as CompQuery>::Item<'_>,
        res: &mut <Self::ResQuery as ResQuery>::Item<'_>,
    ) -> Option<Vec<Command>> {
        if let Some(frame) = res.0 && let Some(selection) = entry.1 {
            frame.render(&self.desc, |mut pass| {
                self.batch.begin(&mut pass);
                self.batch.push(&mut pass, &res.1.transform);
                self.batch.draw(&mut pass, selection);
            });
        }
        
        None
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
            batch: RenderBatch::new(&canvas, &RenderBatchConfig {
                name: "selection",
                shader: "selection",
                translucent: false,
                topology: PrimitiveTopology::LineList,
                depth_write: false,
            }),
        }
    }
}
