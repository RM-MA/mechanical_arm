from tapnet.envs.container import Container
import numpy as np
import matplotlib.pyplot as plt

data_path = "./web_service/logs/RL/231024-190434/logs_0_packed_seq.npy"

data_path = "./web_service/logs/RL/231030-210807/logs_0_packed_seq.npy"


# 231019-162716
from web_service.vis_tools import render_all

wy = 16
wy = 24
wx = 30
l = 7
data = np.array([
    [wx, wy, 10, 0, 0 , 0],
    # [2,  l, 10, 0, wy, 0],
    [wx, wx, 10, 0, 0 , 10],
    [wx, wx, 10, 0, 0 , 20],
])

# data = np.array([
#     [20, 20, 30, 0, 0, 0],
#     [40, 40, 30, 0, 0, 30],
#     [50, 50, 30, 0, 0, 60],
# ])


# data = np.array([
#     [20, 20, 30, 0, 0, 0],
#     [20, 20, 29, 20, 0, 0],
#     [20, 20, 23, 40, 0, 0],
#     [20, 20, 25, 60, 0, 0],
#     [20, 20, 25, 80, 0, 0],
#     [20, 20, 25, 100, 0, 0],
#     [20, 20, 25, 100, 0, 0],
# ])

data = np.load(data_path).astype('int')[:13]

# print(data)

render_all(data, [120, 100, 100])

boxes = data[:, :3]
poses = data[:, 3:]


container = Container([125, 105, 150], "H", 1, 'hard_before_pack', 0.5, container_type='multi', same_height_threshold=1, init_ctn_num=1, pack_type='all', ems_type='ems-id-stair')

box_num = len(boxes)
for i in range(box_num):
    box = boxes[i]
    pos = poses[i]

    # if i == 7:
    #     container.same_height_threshold = 2
    
    container.add_new_box(box, real_pos=[pos[0], pos[1], pos[2], 0])

    hm = container.each_container_heightmap[0]
    
    # if i == 7:
    #     plt.imshow(hm)
    #     plt.show()



# hm = container.update_heightmap(0, 6)


plt.imshow(hm)
plt.show()

print( len(container.each_container_boxes[0]) )

# 231009-150353
# 231009-183003
# 231010-145325
# 231011-162618