import os
import datetime
import numpy as np
import torch
import random
import copy
import itertools
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import gymnasium as gym
from tqdm import tqdm
import tapnet.envs.ems_tools as ET
import tapnet


def reorder(boxes, box_ids, box_areas, order_type='rand'):
    
    if order_type == 'rand':
        new_order = [ i for i in range(len(boxes))]
        np.random.shuffle(new_order)
    elif order_type == 'area':
        new_order = np.argsort(box_areas)
                    
    boxes = np.array(boxes)
    box_ids = np.array(box_ids)
    box_areas = np.array(box_areas)
    

    boxes = boxes[new_order]
    box_ids = box_ids[new_order]
    box_areas = box_areas[new_order]
    
    boxes = list(boxes)
    box_ids = list(box_ids)
    box_areas = list(box_areas)

    return boxes, box_ids, box_areas

class OBS():
    def __init__(self, obs) -> None:
        self.box_num =  np.array([obs['box_num']])
        self.ems_num =  np.array([obs['ems_num']])
        self.state_num =  np.array([obs['state_num']])
        self.box_states =  obs['box_states'][None, :]
        self.valid_mask =  obs['valid_mask'][None, :]
        self.access_mask =  obs['access_mask'][None, :]
        self.ems =  obs['ems'][None, :]
        self.ems_mask =  obs['ems_mask'][None, :]
        self.ems_size_mask =  obs['ems_size_mask'][None, :]
        self.ems_to_box_mask =  obs['ems_to_box_mask'][None, :]
        
        self.precedence =  obs['precedence'][None, :]
        self.pre_box =  obs['pre_box'][None, :]
        self.heightmap =  obs['heightmap'][None, :]
        self.container_width = np.array([obs['container_width']])
        self.container_length =  np.array([obs['container_length']])


class TAPer:
    def __init__(self, container_width=100, container_length=100, container_height=200, preview_num=5, \
                hidden_dim = 128, checkpoint_path=None, device='cuda', \
                    unit_scale=1, stable_rule='hard_before_pack', stable_scale_factor=0.4, \
                        use_bridge=False, same_height_threshold=0, min_ems_width=0, min_height_diff=0, \
                            scale_to_large=False, gripper_size=None, init_ctn_num=None, \
                                stable_predict = False, prec_type = 'none', rotate_axes = ['z'], data_type = 'rand', fact_type = 'box', world_type = 'real'
                                ):

        assert checkpoint_path is not None

        max_box_num = 100

        container_width = container_width * unit_scale
        container_length = container_length * unit_scale
        container_height = container_height * unit_scale

        # self.container_width = container_width
        # self.container_length = container_length
        # self.container_height = container_height

        self.preview_num = preview_num
        
        container_size = [container_width, container_length, container_height]
        self.container_size = container_size

        if 'hard' in stable_rule:
            allow_unstable = False
        else:
            allow_unstable = True


        # if use_bridge:
        #     ems_dim = 7
        # else:
        #     ems_dim = 6


        # prec_type = 'none'
        # rotate_axes = [ 'z']
        # data_type = 'rand'
        # fact_type = 'box'

        # stable_predict = True
        require_box_num = None

        if init_ctn_num is None:
            container_type = 'single'
        else:
            container_type = 'multi'
        
        ems_dim = 6 + (container_type == 'multi')
        box_dim = 3

        env = gym.make('tapnet/TAP-v0', 
            box_num=max_box_num,  
            ems_dim=ems_dim,
            container_size=container_size, 
            box_range=[10, 80],
            stable_rule=stable_rule,
            allow_unstable=allow_unstable,
            stable_scale_factor=stable_scale_factor,
            use_bridge=use_bridge,
            for_test=True,
            same_height_threshold=same_height_threshold,
            min_ems_width=min_ems_width,
            min_height_diff=min_height_diff,
            fact_type=fact_type,
            data_type=data_type,
            rotate_axes=rotate_axes,
            action_type='box-ems',
            ems_type='ems-id-stair',
            scale_to_large=scale_to_large,
            gripper_size=gripper_size,
            require_box_num=require_box_num,
            init_ctn_num=init_ctn_num,
            world_type=world_type,
            container_type=container_type,
            pack_type="all",
            reward_type='H' )
        # env.seed(666)
        
        prec_dim = 2
        if prec_type == 'cnn':
            prec_dim = max_box_num * prec_dim

        from tapnet.models.network import Actor

        actor_list = []
        act_num = len(checkpoint_path)

        for ai in range(act_num):
            if checkpoint_path[ai] is None: continue
            
            actor = Actor(box_dim, ems_dim, hidden_dim, prec_dim, prec_type, stable_predict, device)
            actor = actor.to(device)

            # from tapnet.models.greedy import Greedy
            # actor = Greedy( pack_type='last', container_height=container_size[2], device=device).to(device)

            if checkpoint_path is not None:
                print('Loading ', checkpoint_path[ai])
                state_dict = torch.load(checkpoint_path[ai], map_location=device)

                if 'model' in state_dict:
                    state_dict = state_dict['model']
                model_dict = actor.state_dict()
                actor_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                # actor_dict = { k[6:] :v for k,v in state_dict.items() if ( '_actor_critic' not in k and 'critic' not in k ) }
                
                actor.load_state_dict(actor_dict)
            actor.eval()

            actor_list.append(actor)

        self.device = device
        self.env = env
        self.actor = actor_list[0]

        self.actor_list = actor_list

        # import IPython
        # IPython.embed()
        

        self.unit_scale = unit_scale

        self.env.reset()

    def init_packing_state(self, pre_boxes=None, heightmap=None, packing_mask=None, reset=False):
        
        pack_info_box = []
        pack_info_pos = []
        
        if reset:
            self.env.reset()

        if heightmap is not None:
            self.env.set_init_height(heightmap)
        
        if pre_boxes is not None:
            # print("pre num: ", len(pre_boxes))
            for pre_b in pre_boxes:
                w,l,h,x,y,z = pre_b[:6]
                box = np.array([w,l,h]) * self.unit_scale
                pos = np.array([x,y,z,0]) * self.unit_scale
                self.env.container.add_new_box( box, real_pos=pos)
                
                pack_info_box += [[box]]
                pack_info_pos += [[pos]]

            self.env.container.update_ems()
            
            # packing_mask = self.env.container.each_container_heightmap[0] != 0
            # # convex hull of current packing
            # packing_points = np.where(packing_mask)
            # packing_points = np.column_stack(packing_points)

            # if len(packing_points ) < 3:
            #     packing_mask = None
            # else:
            #     packing_convex = ConvexHull(packing_points)
            #     hull_path = Path(packing_points[packing_convex.vertices])

            #     max_x = packing_points[:,0].max()
            #     max_y = packing_points[:,1].max()

            #     xs, ys = np.meshgrid(range(0, max_x+1), range(0, max_y+1))
            #     xs = np.reshape(xs, (-1))
            #     ys = np.reshape(ys, (-1))
            #     grid_points = np.column_stack([xs, ys])
            #     insides = hull_path.contains_points(grid_points)

            #     inside_ids = np.where(insides == 1)[0]
            #     for idx in inside_ids:
            #         inside_point = grid_points[idx]
            #         packing_mask[ inside_point[0], inside_point[1] ] = True
            
        if packing_mask is not None:
            self.env.container.set_packing_mask(packing_mask)
        
        return pack_info_box, pack_info_pos

    def pick_one_to_pack(self, boxes, check_box_stable=True):

        # obs
        self.env.set_new_task(boxes)

        obs = self.env.get_obs(check_box_stable=check_box_stable)

        if len(obs['ems_to_box_mask']) == 0 or obs['ems_to_box_mask'].max() == False:
            return True, None, None, None, None, None, [0,0]

        # get best action
        min_probs = None
        std = 0
        min_std = 999999999999
        max_std = -1
        min_p = 6999
        max_p = -1

        min_loc = None

        use_loc = True

        for actor in self.actor_list:
            probs, _state = actor( OBS(obs) )
            
            if use_loc:
                tmp_action = int(torch.argmax(probs))
                _, _, tmp_ems_id, tmp_corner_id, _, _, tmp_box, tmp_box_state_id, _ = self.env.decode_action(tmp_action)
                ems_xy, _, tmp_container_id = ET.compute_packing_pos(tmp_box, tmp_ems_id, tmp_corner_id, self.env.container.empty_max_spaces)
                tmp_hm = self.env.container.each_container_heightmap[tmp_container_id]
                ex, ey = np.array(ems_xy).astype('int')
                bx, by, bz = np.array(tmp_box).astype('int')
                ez = tmp_hm[ ex:ex+bx, ey:ey+by ].max()

                loc = [ex, ey, ez]

                axis_order = [0,1,2]
                axis_order = [2,0,1]

                if min_loc is None:
                    min_loc = loc
                    min_probs = probs
                if loc[ axis_order[0] ] == min_loc[ axis_order[0] ]:
                    if loc[ axis_order[1] ] == min_loc[ axis_order[1] ]:
                        if loc[ axis_order[2] ] < min_loc[ axis_order[2] ]:
                            min_loc = loc
                            min_probs = probs
                    elif loc[ axis_order[1] ] < min_loc[ axis_order[1] ]:
                        min_loc = loc
                        min_probs = probs
                elif loc[ axis_order[0] ] < min_loc[ axis_order[0] ]:
                    min_loc = loc
                    min_probs = probs

            else:
                all_valid_probs = probs[probs != 0]
                std = all_valid_probs.std()
                if min_probs is None:
                    min_probs = probs

                if min_std > std:
                    min_std = std
                    min_probs = probs
            
        
        ems_box_grasp_mask = obs['ems_box_grasp_mask']
        
        action = int(torch.argmax(min_probs))

        obs, reward, terminated, truncated, info = self.env.step( action )

        # print(terminated, truncated, info)
        if truncated or terminated:
            return True, None, None, None, None, None, [0,0]
        
        container_id = self.env.container.last_pack_container

        hm = self.env.container.each_container_heightmap[self.env.container.last_pack_container]
        idx = info['box_id']
        box = info['box']
        pos = info['pos']

        if self.env.gripper_size is not None:
            ems_id = info['ems_id']
            box_state_id = info['box_state_id']
            ems_box_grasp_mask = ems_box_grasp_mask.reshape(-1, info['box_state_num'])
            grasp_mode = ems_box_grasp_mask[ems_id, box_state_id]
            # TODO maybe adjust grasp position

        cw = self.container_size[0]
        cl = self.container_size[1]

        # pos[0] = cw - pos[0]
        # pos[1] = cl - pos[1] - box[1]

        return False, box, pos, idx, hm, container_id, [std, min_probs.max()]

    def pack(self, boxes, pre_boxes=None, heightmap=None, preview_num=None, packing_mask=None, check_box_stable=False, invert=False, box_ids=None, box_areas=None):
        '''
        params:
            boxes: list [ [ x,y,z ], [x,y,z], ... ]
            pre_boxes: list [ [ w,l,h,x,y,z,t ], ... ]
            heightmap: np.arr [width x length]
            preview_num: int
        
        return:
            boxes: list [ [ w,l,h,x,y,z,t ], ... ]

        '''

        self.init_packing_state(pre_boxes, heightmap, packing_mask, reset=True)

        if preview_num is None:
            preview_num = self.preview_num

        ret_list = []

        retry_time = 0

        # loop_num = len(boxes)
        while(len(boxes) > 0):
        # for _ in tqdm(range(loop_num)):

            view_boxes = boxes[:preview_num]
            done, box, pos, idx, hm, container_id = self.pick_one_to_pack(view_boxes, check_box_stable)

            if not done:
                boxes.pop(idx)
                
                bid = 1
                if box_ids is not None:
                    bid = box_ids[idx]

                    box_ids.pop(idx)
                    box_areas.pop(idx)

                if invert:
                    ret_list.append( [ box[1], box[0], box[2], pos[1], pos[0], pos[2], bid] )
                else:
                    ret_list.append( list( box ) + list(pos) + [bid] )

            else:
                if box_ids is not None:
                    if retry_time > 0:
                        retry_time = 0
                        print(f"  no place to pack, still have {len(boxes)}")

                        boxes, box_ids, box_areas = reorder(boxes, box_ids, box_areas, order_type='rand')
                        break
                    retry_time = 1
                    boxes, box_ids, box_areas = reorder(boxes, box_ids, box_areas, order_type='area')
                    
                else:
                    # print(f"  no place to pack, still have {len(boxes)}")
                    break

        reward, delta_float, delta_int = self.env.get_reward(terminated=True)        
        # box_num = len(self.env.container.boxes)

        # height = np.max(self.env.container.heightmap)
        # print("  c     : ", reward )
        # print("  height: ", height )

        return ret_list

    def pack_box(self, boxes, check_box_stable=True, allow_preview=True, invert=False):
        '''
        params:
            boxes: list [[ x,y,z ], [ x,y,z ], ...]
            check_box_stable: bool
            allow_preview: bool
        
        return:
            ret_box: list [ w,l,h,x,y,z,t ]
            ret_id: int

        '''

        min_height = 10000
        ret_id = None
        ret_box = None
        container_id = None

        if len(boxes) == 0:
            return False, ret_box, ret_id, container_id, [0, 0]

        if allow_preview: 
            done, box, pos, idx, hm, container_id, extra = self.pick_one_to_pack(boxes, check_box_stable)
            
            if done:
                return True, None, None, None, [0, 0]
            if invert == True:
                ret_box = [ box[1], box[0], box[2], pos[1], pos[0], pos[2], 2]
            else:
                ret_box = list( box ) + list(pos) + [2]

            ret_id = idx
            return False, ret_box, ret_id, container_id, extra

        for i, box in enumerate(boxes):
            if len(boxes) > 1:
                copy_env = copy.deepcopy(self.env)

            pack_box = np.array(box)[None, :]
            done, box, pos, _, hm, container_id = self.pick_one_to_pack(pack_box, check_box_stable)
            
            height = hm.max()
            if box is not None and min_height > height:
                ret_id = i
                ret_box = list( box ) + list(pos) + [1]
            
            # TODO, box none
            if len(boxes) > 1:
                self.env = copy_env
        
        if len(boxes) > 1:
            box = boxes[ret_id]
            pack_box = np.array(box)[None, :]
            done, box, pos, _, hm, container_id = self.pick_one_to_pack(pack_box, check_box_stable)

        return False, ret_box, ret_id, container_id, [0, 0]

    def reset(self):
        self.env.reset()
        
    def close(self):
        reward, _, _ = self.env.get_reward(terminated=True, reward_type="H")
        heightmap = self.env.container.each_container_heightmap[0]
        max_height = np.max(heightmap)
        var = np.var(heightmap)
        # print("cntner: ", reward )
        # print("height: ", max_height )
        return reward, max_height, var


def gen_simu_seq(simu_len, item_size_set, boxes_proportion, shuffle=True):
    simu_seq = []
    for i, box in enumerate(item_size_set):
        for j in range(int(boxes_proportion[i] * simu_len)):
            simu_seq.append(box)
    if shuffle:
        random.shuffle(simu_seq)
    return simu_seq

def generate_boxes(total_num, rate, shuffle, seed=666, init_boxes=None):

    if seed is not None:
        random.seed(seed)
        print('  seed: ', seed)

    shuffle = True
    # shuffle = False

    rate = np.array(rate) * 1.0

    if shuffle == False:
        # rate = np.array([5,3,4]) * 1.0
        # rate = np.array([6,4,4]) * 1.0
        total_num = int(np.sum(rate))
    # else:
    #     rate = np.array([1,1,1]) * 1.0
    #     total_num = 60

    if init_boxes is None:
        # init_boxes = [
            # [22,31,17], [29,43,20], [29,33,21], #[35,58,24]
            # [32, 29, 20], [30, 23, 17], [26, 19, 13],
            # [43, 34, 24], [33, 29, 20], [31, 23, 16],
            # [32, 29, 20.2], [30, 23, 17.6], [26, 19, 13.4],
            # [32.4, 29.6, 20], [30.3, 23.2, 17], [26.8, 19.2, 13],
            # [32, 29, 15], [30, 23, 15], [26, 19, 15],
        # ]
        init_boxes = [[43, 34, 24], [43, 34, 18], [41, 29, 19], [33, 29, 20], [31, 23, 16], [26, 19, 13]]

    boxes = gen_simu_seq(total_num, init_boxes,  rate / np.sum(rate), shuffle )
    # print(boxes)

    # np.save("./test/box_644.npy", boxes)
    return boxes
