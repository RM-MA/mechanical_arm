import numpy as np
import torch
import random
import copy
import gymnasium as gym
import os

import sys
sys.path.append( os.path.join(os.path.dirname(__file__), "..") )


from .container import Container, left_bottom
from .factory import Factory
from . import ems_tools as ET

def get_current_mask(mask, precedences, box_num):
    current_mask = mask.clone()
    move_mask = precedences[:, :box_num, :].sum(1)
    rotate_small_mask = precedences[:, box_num:box_num*2, :].sum(1)
    rotate_large_mask = precedences[:, box_num*2:box_num*3, :].sum(1)
    rotate_mask = rotate_small_mask * rotate_large_mask
    dynamic_mask = rotate_mask + move_mask
    
    current_mask[ dynamic_mask.ne(0) ] = 0.
    return current_mask.detach()

class TAP(gym.Env):
    def __init__(self, box_num, ems_dim, container_size, box_range, \
                 reward_type='C', rotate_axes=['z'], \
                    unit_scale=1, stable_rule='hard_before_pack', stable_scale_factor=0.1, allow_unstable=False, \
                        same_height_threshold=0, min_ems_width=0, min_height_diff=0, ems_per_num = 8, \
                            use_bridge=False, render_mode=None, for_test=False, \
                            fact_type='box', data_type='rand', action_type='box-ems', ems_type='ems', \
                            world_type='real', container_type='multi', pack_type='last', stable_predict=False, \
                            fact_data_folder=None, scale_to_large=False, gripper_size=None, require_box_num=None,
                            simulation_scene=None, init_ctn_num = None
                        ):

        self.unit_scale = unit_scale
        self.allow_unstable = allow_unstable
        self.use_bridge = use_bridge
        self.stable_rule = stable_rule
        self.action_type = action_type

        self.scale_to_large = scale_to_large
        self.gripper_size = gripper_size
        
        self.same_height_threshold = same_height_threshold
        self.min_ems_width = min_ems_width
        self.min_height_diff = min_height_diff

        self.container = Container(container_size, reward_type, \
                               unit_scale=unit_scale, \
                               same_height_threshold=same_height_threshold, min_ems_width=min_ems_width, \
                                stable_rule=stable_rule, stable_scale_factor=stable_scale_factor, use_bridge=use_bridge, \
                                    ems_type=ems_type, world_type=world_type, container_type=container_type, pack_type=pack_type, init_ctn_num=init_ctn_num, stable_predict=stable_predict)
        
        self.container_num = 1

        self.rotate_axes = rotate_axes
        self.box_range = box_range
        self.reward_type = reward_type

        self.render_mode = render_mode
        self.save_dir = None
        self.note = None

        # generate boxes
        gripper_width= int(np.ceil(container_size[0] * 0.1))
        self.factory = Factory(fact_type, data_type, container_size, gripper_width, require_box_num, fact_data_folder, simulation_scene )
    
        self.fact_data_folder = fact_data_folder
        if self.fact_data_folder is not None:
            file_names = os.listdir(self.fact_data_folder)
            file_num = len(file_names)
            self.fact_max_num = int(file_num / 2)
        # data_folder = f"./data/{fact_type}/{data_type}/{box_num}/[{target_container_size[0]}_{target_container_size[1]}]_[{box_range[0]}_{box_range[1]}]_{gripper_width}"

        self.box_num = box_num
        self.container_size = container_size

        self.container_id = 0

        state_num = len(rotate_axes) * 2
        box_state_num = box_num * state_num
        self.state_num = state_num
        self.box_state_num = box_state_num

        
        max_ems_num = box_num * ems_per_num
        self.ems_dim = ems_dim
        self.ems_per_num = ems_per_num
        self.max_ems_num = max_ems_num

        self.for_test = for_test

        self.observation_space = gym.spaces.Dict(
            {
                "box_num": gym.spaces.Discrete(box_num + 1, start=0),
                "ems_num": gym.spaces.Discrete(max_ems_num, start=1),
                "state_num": gym.spaces.Discrete(state_num, start=1),
                "box_states": gym.spaces.Box(box_range[0], box_range[1], shape=( box_state_num * 3, ), dtype=np.float32),
                "valid_mask": gym.spaces.Box(0, 1, shape=( box_state_num, ), dtype=bool),
                "access_mask": gym.spaces.Box(0, 1, shape=( box_state_num, ), dtype=bool),
                "ems": gym.spaces.Box(0, max(container_size)+1, shape=( max_ems_num * ems_dim, ), dtype=np.float32),
                "ems_mask": gym.spaces.Box(0, 1, shape=(max_ems_num * max_ems_num, ), dtype=bool),
                "ems_size_mask": gym.spaces.Box(0, 1, shape=(max_ems_num * box_state_num, ), dtype=int),
                "ems_to_box_mask": gym.spaces.Box(0, 1, shape=(max_ems_num * box_state_num, ), dtype=int),
                "precedence": gym.spaces.Box(0, 1, shape=(2,), dtype=int),
                
                "ems_box_grasp_mask": gym.spaces.Box(0, 1, shape=(max_ems_num * box_state_num, ), dtype=int),
                
                # for tapnet
                "pre_box": gym.spaces.Box(box_range[0], box_range[1], shape=( 3, ), dtype=int),
                "heightmap": gym.spaces.Box(0, 1000, shape=( container_size[0] * container_size[1] * 2, ), dtype=int),
                "container_width": gym.spaces.Discrete(container_size[0], start=0),
                "container_length": gym.spaces.Discrete(container_size[1], start=0),
            }
        )
        self.action_space = gym.spaces.Discrete( box_state_num * ems_per_num )
        
    def generate_new_task(self):
        if self.fact_data_folder is not None:
            index = np.random.randint(self.fact_max_num)
            self.factory.load_order(index)
        else:
            self.factory.new_order(self.box_range, self.box_num)

    def set_new_task(self, boxes, prec_graph=None):
        self.factory.new_order(None, None, boxes, prec_graph)

    # def set_init_heightmap(self, heightmap):
    #     self.container.set_init_height(heightmap)

    def get_obs(self, check_box_stable=None, none_obs=False):
        if none_obs:
            obs = {
                "box_num": None,
                "ems_num": None,
                "state_num": None,
                "box_states": None,
                "valid_mask": None,
                "access_mask": None,
                "ems": None,
                "ems_mask": None,
                "ems_size_mask": None,
                "ems_to_box_mask": None,
                "precedence": None,

                "ems_box_grasp_mask": None,

                'pre_box': None,
                'heightmap': None,
                'container_width': None,
                'container_length': None
            }
            return obs

        origin_ems, ems, ems_mask = self.container.get_ems( self.ems_dim, self.for_test, self.max_ems_num )
        ems_num = len(ems)

        box_states, prec_states, valid_mask, access_mask = self.factory.get_box_state(self.rotate_axes)
        box_num = len(self.factory.select_list)
        state_num = self.state_num

        # for tapnet
        if len(self.container.boxes) > 0:
            pre_box = self.container.boxes[-1]
        else:
            pre_box = [0,0,0]
        pre_box = np.array(pre_box)
        
        heightmap = self.container.get_heightmap('diff')

        if check_box_stable is None:
            if self.stable_rule == 'none' or 'after' in self.stable_rule:
                check_box_stable = False
            else:
                check_box_stable = True

        if box_num == 0:
            ems_to_box_mask = np.ones((ems_num, 1))
            ems_size_mask = np.ones((ems_num, 1))
        else:
            ems_size_mask, ems_to_box_mask, ems_box_grasp_mask = ET.compute_box_ems_mask(box_states, origin_ems, box_num,
                    self.container.each_space, self.container.each_container_heightmap, self.container.packing_mask, self.factory.remove_list, check_box_stable, self.gripper_size,
                    same_height_threshold=self.same_height_threshold)

            if self.allow_unstable:
                ems_to_box_mask = ems_size_mask

            if self.container.container_type == 'multi':
                box_mask = access_mask * valid_mask
                filter_mask = np.einsum('j,aj->aj', box_mask, ems_to_box_mask)

                if len(filter_mask) == 0:
                    filter_max = False
                else:
                    filter_max = filter_mask.max()
                
                # check valid action
                if box_mask.any() and filter_max == False and self.container.init_ctn_num is None:
                    # new containeryou
                        self.container.each_valid_mask = list(np.array(self.container.each_valid_mask) * 0)
                        self.container.add_new_container()
                        self.container.update_ems()
                        origin_ems, ems, ems_mask = self.container.get_ems( self.ems_dim, self.for_test, self.max_ems_num )
                        ems_num = len(ems)

                        ems_size_mask, ems_to_box_mask, ems_box_grasp_mask = ET.compute_box_ems_mask(box_states, origin_ems, box_num,
                                self.container.each_space, self.container.each_container_heightmap, self.container.packing_mask, self.factory.remove_list, check_box_stable, self.gripper_size,
                                same_height_threshold=self.same_height_threshold
                                )


        box_states, ems = ET.normal_size(box_states, ems, self.container_size, self.unit_scale, self.scale_to_large)

        obs = {
            "box_num": box_num,
            "ems_num": ems_num,
            "state_num": state_num,
            "box_states": box_states.reshape(-1),
            "valid_mask": valid_mask,
            "access_mask": access_mask,
            "ems": ems.reshape(-1),
            "ems_mask": ems_mask.reshape(-1),
            "ems_size_mask": ems_size_mask.reshape(-1),
            "ems_to_box_mask": ems_to_box_mask.reshape(-1),
            "precedence": prec_states.reshape(-1),
            
            "ems_box_grasp_mask": ems_box_grasp_mask.reshape(-1),

            'pre_box': pre_box,
            'heightmap': heightmap.reshape(-1),
            'container_width': self.container_size[0],
            'container_length': self.container_size[1],
        }
        return obs

    def get_info(self):
        info = { 'ctn': len(self.container.Cs), 'box_num': len(self.factory.box_list)}
        return info

    def height_diff_action(self, action):
        max_h = -1
        min_h = 100000
        # [ [x,y,z], [x,y,z], [0,0,0] ]
        for ems in self.container.empty_max_spaces:
            if ems[1][2] > 0:
                ez = ems[0][2]
                if ez < min_h:
                    min_h = ez
                if ez > max_h:
                    max_h = ez
        
        if max_h - min_h >= self.min_height_diff:
            # find the lowest ems as action
            obs = self.get_obs()
            valid_mask = obs['valid_mask']
            access_mask = obs['access_mask']
            ems_to_box_mask = obs['ems_to_box_mask']
            ems_to_box_mask = ems_to_box_mask.reshape(-1, len(valid_mask))
            ems_num = len(self.container.empty_max_spaces)
            
            # only select lowest ems
            for ei in range(ems_num):
                ems = self.container.empty_max_spaces[ei]
                if ems[0][2] > min_h:
                    ems_to_box_mask[ei, :] = False
            
            total_mask = access_mask * valid_mask
            filter_mask = np.einsum('j,aj->aj', total_mask, ems_to_box_mask)            
            filter_mask = filter_mask.reshape(-1)

            valid_actions = np.where(filter_mask > 0)[0]
            if len(valid_actions) > 0 and action not in valid_actions:
                action = np.random.choice(valid_actions)
        return action

    def change_container_size(self, new_size):
        self.container_size = new_size
        self.container.change_heightmap_size(new_size)


    def greedy(self):
        # some thing wrong?
        obs = self.get_obs()
        box_num = obs['box_num']
        valid_mask = obs['valid_mask']
        access_mask = obs['access_mask']
        ems_to_box_mask = obs['ems_to_box_mask']
        ems_to_box_mask = ems_to_box_mask.reshape(-1, len(valid_mask))
        box_states = obs['box_states'].reshape(-1, 3)
        ems_num = len(self.container.empty_max_spaces)
        
        total_mask = access_mask * valid_mask
        filter_mask = np.einsum('j,aj->aj', total_mask, ems_to_box_mask)            
        
        valid_pair = np.where(filter_mask > 0)
        valid_pair = np.column_stack(valid_pair)

        max_ems_id = None
        max_box_id = None
        max_box = None
        max_reward = -1000
        for pair in valid_pair:
            container = copy.deepcopy(self.container)
            ems_idx = pair[0]
            box_idx = pair[1]

            box = (box_states[box_idx] * self.container_size[0]).astype('int')

            container.add_new_box(box, ems_idx, 1, False)
            r = container.calc_ratio()

            if r > max_reward:
                max_reward = r
                max_ems_id = ems_idx
                max_box_id = box_idx
                max_box = box

        box_id = max_box_id % box_num

        return max_box, max_ems_id, box_id

    def decode_action(self, action):

        box_num = len(self.factory.select_list)

        # ems_num = len(self.container.empty_max_spaces)
        if self.for_test:
            ems_num = len(self.container.empty_max_spaces)
        else:
            ems_num = self.max_ems_num

        axes_num = len(self.rotate_axes)
        
        box_state_num = 2 * axes_num * box_num # rot x axes x box_num

        action_num = ems_num * box_state_num

        corner_ptr = action % (action_num)
        ems_id = corner_ptr // box_state_num
        box_state_id = corner_ptr % box_state_num

        rot_id = box_state_id // box_num // axes_num
        axis_id = box_state_id // box_num % axes_num
        box_id = box_state_id % box_num

        corner_id = action // (action_num)
        corner_id = corner_id % 4
        
        # is_rotate = box_state_id >= box_num
        is_rotate = rot_id == 1
        
        select_id = self.factory.select_list[box_id]
        box = copy.deepcopy(self.factory.box_list[ select_id ])

        if self.rotate_axes[axis_id] == 'x':
            box = box[ [1,2,0] ]
        elif self.rotate_axes[axis_id] == 'y':
            box = box[ [0,2,1] ]
        
        if is_rotate: box[0], box[1] = box[1], box[0]
        
        real_pos = None
        if self.action_type == 'box':
            corner_id = 0
            ems_id = None
            ems_id = left_bottom(self.container, box, True, ret_ems=True)
            # real_pos = left_bottom(self.container, box, True, ret_ems=False)

        return box_id, axis_id, ems_id, corner_id, real_pos, is_rotate, box, box_state_id, box_state_num

    def step(self, action):
        if action is None:
            reward = 0
            terminated = False
            truncated = True
            obs = self.get_obs()
            info = self.get_info()
            return obs, reward, terminated, truncated, info
    
        # check height_diff
        # if self.for_test:
        #     if self.min_height_diff > 0 and self.for_test:
        #         action = self.height_diff_action(action)
        box_id, axis_id, ems_id, corner_id, real_pos, is_rotate, box, box_state_id, box_state_num = self.decode_action(action)

        if ems_id is None:
            real_pos = [0,0,0,0]
        # else:
        #     pack_pos = None

        try:
            box_id = box_id.item()
        except:
            box_id = int(box_id)
        
        pack_pos = self.container.add_new_box(box, ems_id, corner_id, is_rotate=is_rotate, allow_unstable=self.allow_unstable, real_pos=real_pos, box_id=box_id )
            
        if pack_pos is None and self.for_test:
            return None, None, True, False, None

        if pack_pos is not None:
            self.factory.remove_box(box_id, self.for_test)

        min_ems_width = None
        # if self.min_ems_width > 0 and len(self.box_list) > 0:
        #     current_boxes = np.array(self.box_list)
        #     min_ems_width = np.min(current_boxes[:,:2])
        #     print("   >> min ems width ", min_ems_width)
        self.container.update_ems(min_ems_width)

        truncated = False

        if ( pack_pos is None and np.sum(self.container.each_valid_mask) == 0 )   or ( not self.for_test and self.factory.is_order_finish()):
            terminated = True
        else:
            terminated = False

        if self.for_test:
            obs = self.get_obs(none_obs=True)
        else:
            obs = self.get_obs()

            if self.container.container_type == 'single' or self.container.init_ctn_num is not None:
                if self.for_test:
                    ems_num = len(self.container.empty_max_spaces)
                else:
                    ems_num = self.max_ems_num

                filter_mask = np.einsum('j,aj->aj', obs['access_mask'] * obs['valid_mask'], obs['ems_to_box_mask'].reshape(ems_num, len(obs['valid_mask'])))
                if filter_mask.max() == False:
                    terminated = True

        reward, delta_float, delta_int = self.get_reward(terminated, self.reward_type)

        if self.for_test:
            pos = self.container.positions[-1]
            info = { 'ctn': len(self.container.Cs), 'box':box, 'pos': pos, 'box_id': box_id, 'axis_id': axis_id, 'is_rotate':is_rotate, 'ems_id': ems_id, 'box_state_id': box_state_id, 'box_state_num':box_state_num, 'delta_float': delta_float, 'delta_int': delta_int }
        else:
            info = { 'ctn': len(self.container.Cs), 'box_num': len(self.container.boxes), 'delta_float': delta_float, 'delta_int': delta_int}

        return obs, reward, terminated, truncated, info

    def random_action(self, obs):
        ems_to_box_mask = obs['ems_to_box_mask']
        valid_actions = np.where(ems_to_box_mask > 0)[0]

        if len(valid_actions) == 0:
            ems_size_mask = obs['ems_size_mask']
            valid_actions = np.where(ems_size_mask > 0)[0]
            obs['ems_to_box_mask'] = ems_size_mask

        if len(valid_actions) == 0:
            action = None
        else:
            action = np.random.choice(valid_actions)
        
        return action


    def get_heightmaps( self ):
        return self.container.get_heightmap()
        
    def set_init_height( self, heightmap ):
        if heightmap is None: return
    
        self.container.set_init_height(heightmap)

        min_ems_width = None
        # if self.min_ems_width > 0 and len(self.box_list) > 0:
        #     current_boxes = np.array(self.box_list)
        #     min_ems_width = np.min(current_boxes[:,:2])        
        self.container.update_ems(min_ems_width)

    def set_packing_mask(self, mask):
        self.container.set_packing_mask(mask)
        
    def reset(self, seed=None, options=None):
        self.container.clear_container(with_reward=True)
        self.factory.reset()

        self.container_num = 1

        self.generate_new_task()
        
        obs = self.get_obs()
        info = self.get_info()
        return obs, info

    def get_reward(self, terminated, reward_type='C'):
        
        reward = 0
        delta_float = 0
        delta_int = 0

        if 'E' in reward_type:
            # c in Each step
            r, delta_float, delta_int = self.container.calc_ratio()
            reward += r
            return reward

        if 'N' in reward_type:
            # container Num
            current_container_num = len(self.container.each_container_boxes)
            if current_container_num > self.container_num:
                reward += -1
                self.container_num = current_container_num
            else:
                reward += 0
        
        if 'A' in reward_type:
            # All c, not mean

            if terminated:
                self.container.calc_ratio()
                reward += np.sum(self.container.Cs)
            
            else:
                reward += 0

        if 'S' in reward_type:
            # Stable
            
            # if 'soft' in self.stable_rule:
            #     penalty = 0.05 * (self.container.each_container_stable[self.container.last_pack_container][-1] == False)
            #     reward -= penalty
            if terminated:
                penalty = 0.1 * self.container.new_by_unstable
                reward -= penalty

        if 'H' in reward_type:
            # Height 根据高度计算  

            # Compactness
            if terminated:
                r, delta_float, delta_int = self.container.calc_ratio('current')
                reward += r
            else:
                reward += 0

        if 'D' in reward_type:
            # Differ height

            # NOTE 惩罚高低差

            # if self.min_height_diff > 0 and not self.for_test:
            min_height_diff = self.box_range[1] # 51 71
            min_ems_width = self.box_range[0] # 

            # min_height_diff = 50 # 51 71
            # min_ems_width = self.box_range[0] # 

            if min_height_diff > 0 and not self.for_test:
                if self.gripper_size is not None:
                    gw = int(self.gripper_size[0] / 2)
                else:
                    gw = int(self.container_size[0] * 0.2)

                height_diff = self.container.get_max_height_diff(min_ems_width, gw)

                if height_diff > min_height_diff:
                    # reward -= 0.03
                    reward -= 0.02
                    # reward -= 0.5

        if 'C' in reward_type:
            # Compactness
            if terminated:
                r, delta_float, delta_int = self.container.calc_ratio()
                reward += r
            
            else:
                reward += 0

        if 'T' in reward_type:
            # Touch TODO 加贴合奖励
            if not self.for_test and False:
                pack_pos = pack_pos.astype('int')
                if pack_pos[0] == 0 or self.container.heightmap[pack_pos[0] - 1, pack_pos[1]] > pack_pos[2]:
                    if pack_pos[1] == 0 or self.container.heightmap[pack_pos[0], pack_pos[1]-1] > pack_pos[2]:
                        # pass
                        reward += 0.05
                #     else:
                #         reward -= 0.05
                # else:
                #     reward -= 0.05

        # reward = self.container.calc_ratio()
        return reward, delta_float, delta_int
    
    def save_container_state(self):

        container = self.container
        Cs = container.Cs

        # Rs = container.CPSs
        # Ps = container.Ps
        # Ss = container.Ss

        stables = container.stable
        container_nums = len(container.Cs)

        # heights = container.height_list
        # box_sizes = container.box_size_list
        # valid_sizes = container.valid_size_list
        # empty_sizes = container.empty_size_list
        # stable_nums = container.stable_num_list
        return container_nums, Cs, stables

    def render(self):
        boxes=None
        pos=None
        # save_dir="./pack/test/100-100-note-test"
        # note=''
        self.container.save_states(boxes, pos, self.save_dir + self.note)

    def set_render(self, save_dir="./pack/test/100-100-note-test-source", note=''):
        self.save_dir = save_dir
        self.note = note

    def render_source(self, boxes=None, pos=None, save_dir="./pack/test/100-100-note-test-source", note=''):
        if self.save_dir is not None:
            save_dir = self.save_dir
            note = self.note
        # assert self.source_container is not None
        self.factory.source_container.save_states(boxes, pos, save_dir + note)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]
