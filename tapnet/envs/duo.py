
import pyphysx as phy
from pyphysx_utils.rate import Rate
from pyphysx_render.pyrender import PyPhysxViewer
import numpy as np
import time
import copy

def timer(func):
    def wrapper(*args, **kw):
        start = time.time()
        ret = func(*args, **kw)
        end = time.time()
        print(end-start)
        
        return ret
    return wrapper


def new_box(size, pos, mass=None):
    # pos is the left botton corner of box
    
    size = np.array(size)
    pos = np.array(pos)
    
    actor = phy.RigidDynamic()
    actor.attach_shape(phy.Shape.create_box(size, phy.Material( 0.4, 0.4, restitution=0.5)))

    real_pos = pos + size / 2.0
    
    actor.set_global_pose(real_pos)

    if mass is None:
        mass = size[0] * size[1] * size[2]
    
    actor.set_mass(mass)
    return actor


class Duo():
    def __init__(self, base_size) -> None:

        self.scale = 0.01

        base_size = np.array(base_size)

        self.base_mass = base_size[0] * base_size[1] * base_size[2]
        self.base_size = base_size
        self.z_offset = np.array([0, 0, base_size[2] * self.scale])
        
        self.scene = None
        self.base_box = None
        self.boxes = []
        self.poses = []

    def add_box(self, size, pos, mass=None):
        if mass is None:
            mass = size[0] * size[1] * size[2]

        real_size = np.array(size) * self.scale
        real_pos = np.array(pos) * self.scale + self.z_offset

        box = new_box(real_size, real_pos, mass)
        self.scene.add_actor(box)
        self.boxes.append(box)
    
    def get_poses(self):
        poses = []
        for box in self.boxes:
            pose = box.get_global_pose()
            poses.append(pose[0][2])
        return poses
    
    def check_stable(self, boxes, poses, velocity = 0.3, all_step=500, change_gap=50, min_height_diff=5):
        rate = Rate(240)

        self.init(boxes, poses)

        before_poses = self.get_poses()

        test_vs = [ [velocity, 0], [-velocity, 0],[0, velocity], [0, -velocity] ]
        vs = copy.deepcopy(test_vs)
        vx, vy = vs.pop()
        
        for c in range(all_step):
            self.scene.simulate(rate.period())

            if c % change_gap == 1:
                if len(vs) == 0:
                    vs = copy.deepcopy(test_vs)
                vx, vy = vs.pop()

            self.base_box.set_linear_velocity([vx, vy, 0])

        after_poses = self.get_poses()
        
        box_num = len(after_poses)
        not_same_count = 0
        for i in range(box_num):
            bp = before_poses[i]
            ap = after_poses[i]

            if abs( ap - bp ) > min_height_diff * self.scale:
                not_same_count += 1
                break
        
        if not_same_count > 0:
            return False
        else:
            return True
    
    def init(self, boxes, poses):
        self.boxes.clear()
        self.poses.clear()
        self.scene = phy.Scene()
        self.scene.add_actor(phy.RigidStatic.create_plane(material=phy.Material(static_friction=0.1, dynamic_friction=0.1, restitution=0.5)))

        self.base_box = new_box( self.base_size * self.scale, [0,0,0], self.base_mass )
        self.scene.add_actor(self.base_box)

        box_num = len(boxes)
        for i in range(box_num):
            self.add_box(boxes[i], poses[i])

if __name__ == '__main__':
    duo = Duo([120, 100, 10])

    v = 0.5
    all_step = 600
    gap = 30

    boxes = [
        [20, 20, 10],
        [20, 20, 10],
        [20, 20, 10],
    ]

    poses = [
        [0,0,0],
        [20,20,0],
        [10,3,12]
    ]

    for i in range(1):
        # ret = duo.check_stable([ [30, 20, 10] ], [ [0, 0, 0] ], v)
        ret = duo.check_stable(boxes, poses, v, all_step, gap)
        print(ret)

    duo.init( boxes, poses )


    scene = duo.scene

    rate = Rate(240)

    # for _ in range(10):
    #     scene.simulate(rate.period())


    render = PyPhysxViewer()
    render.add_physx_scene(scene)

    c = 0

    vx = v*1
    vy = v*1

    test_vs = [ [v, 0], [-v, 0],[0, v], [0, -v] ]
    vs = copy.deepcopy(test_vs)

    print(duo.boxes[2].get_global_pose()[0])

    while render.is_active:
        scene.simulate(rate.period())
        render.update()
        rate.sleep()
        c += 1

        if c % gap == 1:
            # if np.random.random() > 0.5: vx *= -1
            # if np.random.random() > 0.5: vy *= -1
            vx, vy = vs.pop()

            if len(vs) == 0:
                vs = copy.deepcopy(test_vs)
        
        duo.base_box.set_linear_velocity([vx, vy, 0])

        if c == all_step: break

    print(duo.boxes[2].get_global_pose()[0])
