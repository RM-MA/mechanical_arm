from envs import Factory
from tqdm import tqdm
import numpy as np
from envs import Container
from envs import ems_tools

if __name__ == '__main__':
    data_num = 5000

    box_num = 20
    box_range = [10, 60]

    # box_num = 10
    # box_range = [10, 80]
    target_container_size = [120, 120, 5000]
    gripper_width = 10

    # box_num = 10
    scale = 10
    box_range = [1, 6]
    target_container_size = np.array([5, 5, 500]) * scale
    # gripper_width = 1

    container = Container(target_container_size, 'C', 1, 'new', 'none')

    boxes = np.array([
        [2,3,3],
        [2,2,1],
        [1,2,1],
        [1,1,1],
    ]) * scale
    poses = np.array([
        [0,0,0],
        [3,0,0],
        [2,1,0],
        [2,0,0]
    ]) * scale

    gripper_size = np.array([15, 15])

    for i in range(len(boxes)):
        box = boxes[i]
        pos = poses[i]

        if i == 3:
            origin_ems, ems, ems_mask = container.get_ems()
            ems_size_mask, ems_to_box_mask, ems_box_grasp_mask = ems_tools.compute_box_ems_mask(box[None,:], origin_ems, 1, heightmap=container.heightmap, check_box_stable=False, gripper_size=gripper_size)
            print(origin_ems)
            print(ems_to_box_mask)
            print(ems_box_grasp_mask)
        container.add_new_box( box, real_pos=pos[:2] )
        container.update_ems()
        container.save_states()
    
    
    # gripper_width= int(np.ceil(100 * 0.1))
    # factory = Factory('tap_fake', 'ppsg', [100, 100, 200], gripper_width )
    # factory.new_order([10, 60], 20)
    # factory.source_container.save_states()

    
    pass