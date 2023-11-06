import time
import os.path as osp
import sys 
import numpy as np
import requests

import random
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
# from maya_infer import gen_simu_seq
from web_service.vis_tools import render_all
# from web_service.test_post import POST, ip, prefix, port
import torch
import datetime
from web_service.args import get_args


def POST(url, post_data):
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Mobile Safari/537.36",
        "Cookie": None}
    req = requests.post(url,json=post_data,headers=headers)
    
    print(req.status_code)
    print(req.text)
    return req.status_code, req.text

ip = "127.0.0.1"
prefix = "speedbot"
port = "8105"

def gen_simu_seq(simu_len, item_size_set, fixed_sequence):
    boxes_proportion = [i / sum(fixed_sequence) for i in fixed_sequence]
    print(item_size_set, fixed_sequence, boxes_proportion)

    simu_seq = []
    for i, box in enumerate(item_size_set):
        for j in range(int(boxes_proportion[i] * simu_len + 0.5)):
            simu_seq.append(box.copy())
    random.shuffle(simu_seq)
    # r = random.randint(0, 10)
    # t = random.randint(simu_len - 10, simu_len - 1)
    # simu_seq[r:t] = sorted(simu_seq[r:t])
    return simu_seq, boxes_proportion


def case_test(args, data_path=None, fixed_sequence=None):
    import random

    random.seed(666)
    traj_nums = args.test_num
    
    frame_id = 10
    frame_size_mm = [1200, 1000, 1040]
    frame_size = frame_size_mm
    # center_point_mm_7e = [220, 2367, 50]
    center_point_mm_7e = [-180, -2367, -500]
    center_point_mm = center_point_mm_7e

    c_list = []
    height_list = []
    var_list = []

    if data_path != None:
        print("dataset path: ", data_path)
        # box_trajs = torch.load(data_path)

        # box_trajs = [[[26,20,16]]]
        # box_trajs = [box_trajs[0], box_trajs[0]]
        
        box_traj = np.load(data_path)[:,:3].astype('int')
        box_trajs = [ box_traj ]
        
        traj_nums = len(box_trajs)
        print("total trajectory: ", traj_nums)
        # eval_freq = 2
    elif fixed_sequence != None:
        simu_len = args.simu_len
        box_trajs = []
        for _ in range(traj_nums):
            box_traj, _ = gen_simu_seq(simu_len, args.item_size_set, fixed_sequence) 
            box_trajs.append(box_traj)
        # eval_freq = 1
        # traj_nums = 1
    
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    all_time = []

    for step_counter in range(traj_nums):
        
        init_data = {
                    "task_id":  step_counter,
                    "frame_id":  frame_id,
                    "frame_length": frame_size[1],
                    "frame_width": frame_size[0],
                    "frame_height": frame_size[2],
                    "gripper_x": 25,
                    "gripper_y": 10,
                    "base_point": center_point_mm,
                    "time":  time.time(),
                    'now': now
                    }
        url = "http://" + ip + ":" + port + "/" + prefix + "/frame/init"
        # print(url, init_data)
        code, _ = POST(url, init_data)
        
        if code == 200:
            # print("Good")
            print('Episode {}'.format(step_counter))
            box_set = box_trajs[step_counter]
            
            simu_box_set_mm = []
            for box_id, box in enumerate(box_set):
                simu_length = box[0] * 10 + random.gauss(0, 1)
                simu_width  = box[1] * 10 + random.gauss(0, 1)
                simu_height = box[2] * 10 + random.gauss(0, 5)
                simu_box_set_mm.append([simu_length, simu_width, simu_height ])
                
            # print(box_set)
            # box_set.append(frame_size + 1)
        
            packed = []
            post_id = 0

            start = time.time()
            
            while len(simu_box_set_mm) > 0:
                print("".join(["-" for i in range(100)]))
                multi_boxes = simu_box_set_mm[:args.preview_simu]
                stack_data = {
                    "frame_id": frame_id,
                    "box_id": "%d-%d" % (step_counter, post_id),
                    "task_id": "1",
                    "time":  time.time()
                }
                box_infos = []
                for box_id, box in enumerate(multi_boxes):
                    box_infos.append({
                        "point": '0, 0, 0, 0, 0, 0',
                        "length": box[0],
                        "width": box[1],
                        "height": box[2], 
                        "box_id": box_id,
                    })
                stack_data["box_info"] = box_infos
                
                url = "http://" + ip + ":" + port + "/" + prefix + "/frame/maduo"
                # print(url, stack_data)
                code, text = POST(url, stack_data)

                end = time.time()
                print('[time] ',  (end-start)  )
                all_time.append(end-start)
                
                start = time.time()

                
                post_id += 1
                if code != 200:
                    break
                
                text = eval(text)
                # print(code, text)
                if 'data' not in text: break

                ret_idx = text['data']['box_id']
                box_mm = simu_box_set_mm[ret_idx]
                position = [i/10 for i in text['data']['point']]
                
                if text['data']['rotation'] == 0:
                    pack = [box_mm[0]/10, box_mm[1]/10, box_mm[2]/10] + position
                else:
                    pack = [box_mm[1]/10, box_mm[0]/10, box_mm[2]/10] + position

                pack[3] = pack[3] - pack[0] / 2 - center_point_mm[0] / 10
                pack[4] = pack[4] - pack[1] / 2 - center_point_mm[1] / 10
                pack[5] = pack[5] - pack[2] - center_point_mm[2] / 10
                # print(pack)
                
                packed.append(pack)
                
                simu_box_set_mm.pop( ret_idx )

                if 'msg' in text:
                    if text['msg'] == 'failed':
                        break
            
            print("".join(["$" for i in range(100)]))

            if traj_nums == 1:
                render_all(packed)

            stack_data = { 
                    "frame_id": frame_id,
                    "box_id": "%d-%d" % (step_counter, box_id),
                    "task_id": "1",
                    "box_info": [],
                    "time":  time.time()
            }
            url = "http://" + ip + ":" + port + "/" + prefix + "/frame/maduo"
            code, text = POST(url, stack_data)
            if code != 200:
                break
            text = eval(text)
            c_list.append(text['data']['c'])
            height_list.append(text['data']['max_height'])
            var_list.append(text['data']['var_height'])

        else:
            print("Initialization Failed", code)

    # np.save("./web_service/logs/maya_v3/%s/C" % str(now), c_list)
    # np.save("./web_service/logs/maya_v3/%s/H" % str(now), height_list)
    # np.save("./web_service/logs/maya_v3/%s/V" % str(now), var_list)

    print(now)
    print('Test num: ', traj_nums, '   Mean C:', np.mean(c_list),  \
          '    Mean H: ', np.mean(height_list), '    Mean Var: ', np.mean(var_list))
    
    print('Mean time of each step: ', np.mean(all_time[1:]))

if __name__ == '__main__':

    args = get_args()

    case_test(args, fixed_sequence=args.fixed_sequence)

    # case_test(args, data_path="./web_service/logs/RL/231008-155628/logs_0_packed_seq.npy", fixed_sequence=args.fixed_sequence)

    