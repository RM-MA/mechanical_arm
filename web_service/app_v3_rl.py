from flask import Flask, request, abort
import time
import datetime
import numpy as np
import copy

from gymnasium.envs.registration import register
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from tapnet.gym_tap import TAPer
from web_service.args import get_args
from oblique import get_oblique_point

prefix = 'speedbot'
DATA_DIR = 'data'
tic = time.time()

app = Flask(__name__, static_folder=DATA_DIR)
app._static_folder = os.path.abspath(DATA_DIR)
app.debug = False  # TODO: make sure this is False in production

G = {}

def registration_envs():
    register(
        id='PackingPreview-v0',
        entry_point='envs:PackingPreview',
    )


def printer(data='', prefix='', code_1='', code_2='', level=0):
    if code_1 != '':
        print(code_1 * 40)

    string = '' + ' ' * level
    if prefix != '': string += '[%s] ' % prefix

    string += str(data)
    print(string)

    if code_2 != '':
        print(code_2 * 40)
    pass



def get_volume_ratio(bin_size, pd):
    p = np.array(pd)
    maxh = max(p[:, 2] + p[:, 5])
    packed_v = sum([i[0] * i[1] * i[2] for i in pd])
    return packed_v / (bin_size[0] * bin_size[1] * maxh), packed_v / (bin_size[0] * bin_size[1] * bin_size[2]), maxh


def init_frame_size(frame_size_mm, base_point_mm):
    # frame_size_mm = [2150.0-1150.0, 750.0-(-450), 1800.0]

    # NOTE TODO
    # frame_size_mm = [1000.0, 1200, 1800.0]
    # frame_size_mm = [1200.0, 1000, 1800.0]

    # frame_size_mm = [1200.0, 1000, 1450.0]
    # # center_point_mm = [1760.0, 60.0, -140.0]
    # center_point_mm = [1700.0, 100.0, -355.0]
    top_point_mm = [1700.0, 100.0, -355.0+2500.0]

    inflate_size_cm = G["inflate_size"]

    frame_dec_x = 0
    frame_dec_y = 0
    # frame_dec_y = 200
    frame_size_mm[0] -= frame_dec_x
    frame_size_mm[1] -= frame_dec_y

    # add extra width of frame
    frame_size_mm[0] += inflate_size_cm * 10
    frame_size_mm[1] += inflate_size_cm * 10
    
    ################################################
    
    frame_size_cm = [int(i / 10) for i in frame_size_mm]

    # corner_pos = G["corner_pos"]

    center_point_mm = [  base_point_mm[0] + frame_size_mm[0] / 2, base_point_mm[1] + frame_size_mm[1] / 2, base_point_mm[2]   ]

    base_point_mm_list = []
    for i in range(4):
        if i == 0:
            base_point_mm = [int(center_point_mm[0] - frame_size_mm[0] / 2), int(center_point_mm[1] - frame_size_mm[1] / 2), center_point_mm[2]]
        elif i == 1:
            base_point_mm = [int(center_point_mm[0] + frame_size_mm[0] / 2), int(center_point_mm[1] - frame_size_mm[1] / 2), center_point_mm[2]]
        elif i == 2:
            base_point_mm = [int(center_point_mm[0] - frame_size_mm[0] / 2), int(center_point_mm[1] + frame_size_mm[1] / 2), center_point_mm[2]]
        elif i == 3:
            base_point_mm = [int(center_point_mm[0] + frame_size_mm[0] / 2), int(center_point_mm[1] + frame_size_mm[1] / 2), center_point_mm[2]]
        base_point_mm_list.append(base_point_mm)

    assert frame_size_cm[0] > 0 and frame_size_cm[1] > 0, "\n!!!!!!!!!!!!!!!!\nFrame length and width should LARGER than 0 !!"
    
    return base_point_mm_list, frame_size_mm, frame_size_cm, top_point_mm

def read_boxes(box_info):
    boxes = []
    
    for ii, box in enumerate(box_info):
        # print("box_id", ii, box, box['box_id'])
        box_id = int(box['box_id'])
        box_length_mm = float(box['length'])
        box_width_mm = float(box['width'])
        box_height_mm = float(box['height'])
        box_length = int(box_length_mm / 10 + 0.5)
        box_width  = int(box_width_mm  / 10 + 0.5)
        box_height = int(box_height_mm / 10 + 0.5)
        # box_angle = box_info[box_id]['point'].split(',')[3:6]
        
        boxes.append({'box_id': box_id, 
                      'box_cm': [box_length, box_width, box_height], 
                      'box_mm': [box_length_mm, box_width_mm, box_height_mm]})
    
    return boxes


def preprocess_boxes(box_info, item_size_set):
    boxes = read_boxes(box_info)

    new_boxes = []
    
    printer('', 'input')
    printer("+ %.2f cm" % G['capping_dilate'], 'preprocess', level=3)
    
    for bb in boxes[::-1]:
        normal_box_cm = bb['box_cm']
        
        normal_box_cm[0] += G['capping_dilate']
        normal_box_cm[1] += G['capping_dilate']

        bb['box_cm'] = normal_box_cm
        new_boxes.append(bb)

        printer(["%.2f" % e for e in bb['box_cm']], 'box cm', level=6)
        
    return new_boxes


def postprocess_packed(in_box, packed, base_point_mm_list, frame_id, pre_heightmap):
    # convert grid into real position
    
    place_cm = packed[-1]
    rotation = 0 #float(box_angle[0])
    
    bx = int(in_box[0])
    by = int(in_box[1])
    px = int(place_cm[0] + 0.5)
    py = int(place_cm[1] + 0.5)
    
    # if in_box[0] == place_cm[1] and in_box[1] == place_cm[0]:
    if bx == py and by == px:
        rotation = 1
    

    x = (place_cm[3] + place_cm[0] / 2) * 10
    y = (place_cm[4] + place_cm[1] / 2) * 10
    z = (place_cm[5] + place_cm[2]) * 10

    printer(["%.1f" %e for e in in_box], 'pack box', level=3)
    printer(["%.2f" %e for e in place_cm[3:6]], 'place cm', level=3)
    # printer(["%.2f" %e for e in [x, y, z]], 'my center', level=3)

    display_info = copy.deepcopy([place_cm[3] * 10, place_cm[4] * 10, place_cm[5] * 10, place_cm[0]*10, place_cm[1]*10, place_cm[2]*10])

    oblique_pos = get_oblique_point(place_cm[3:6], pre_heightmap,  place_cm[:3] )

    grid_place_pos = np.array(place_cm[3:6]).copy()
    grid_oblique_pos = np.array(oblique_pos).copy() - [ place_cm[0]/2, place_cm[1]/2, 0 ]

    grid_place_pos = grid_place_pos.astype('int')
    grid_oblique_pos = grid_oblique_pos.astype('int')

    printer(oblique_pos, 'oblique_init', level=3)
    oblique_pos = oblique_pos * 10.0

    corner_pos = G["corner_pos"]
    base_point_mm = base_point_mm_list[corner_pos]
    
    if corner_pos == 0:
        x += base_point_mm[0]
        y += base_point_mm[1]
        
        oblique_pos[0] += base_point_mm[0]
        oblique_pos[1] += base_point_mm[1]

    elif corner_pos == 1:
        x = base_point_mm[0] - x
        y += base_point_mm[1]

        oblique_pos[0] = base_point_mm[0] - oblique_pos[0]
        oblique_pos[1] += base_point_mm[1]
        
    elif corner_pos == 2:
        x += base_point_mm[0]
        y = base_point_mm[1] - y
        
        oblique_pos[0] += base_point_mm[0]
        oblique_pos[1] = base_point_mm[1] - oblique_pos[1]
        
    elif corner_pos == 3:
        x = base_point_mm[0] - x
        y = base_point_mm[1] - y
        
        oblique_pos[0] = base_point_mm[0] - oblique_pos[0]
        oblique_pos[1] = base_point_mm[1] - oblique_pos[1]



    grid_place_x, grid_place_y, grid_place_z = grid_place_pos
    grid_oblique_x, grid_oblique_y, grid_oblique_z = grid_oblique_pos
    if grid_oblique_x < 0: grid_oblique_x = 0
    if grid_oblique_y < 0: grid_oblique_y = 0

    # if G['capping_net'] is not None:
    #     hm = G["capping_net"].env.container.each_container_heightmap[0].copy()
    # else:
    #     hm = np.zeros(G['frame_size_cm'][:2])
    hm = pre_heightmap

    place_max = hm[  grid_place_x:, grid_place_y: ].max()
    oblique_max = hm[  grid_oblique_x:, grid_oblique_y: ].max()
    h_max = max(place_max, oblique_max)

    if h_max == 0:
        h_max = 20
    
    oblique_pos[2] = (h_max + place_cm[2] + 12) * 10

    z += base_point_mm[2]
    oblique_pos[2] += base_point_mm[2]

    z_offset = 153 # mm
    z_offset = 149 # mm
    z += z_offset
    oblique_pos[2] += z_offset

    # add z offset to fix detection error: n * log(n/2) * 0.1 mm
    if args.real == 1:
        packed_z = place_cm[5]
        if packed_z > 1:
            # z_offset = packed_z * np.log((packed_z / 2.0)) * 0.15
            z_offset = 0
        else:
            z_offset = 0
        z += z_offset
        # print("[current step] z offset", z_offset)



    print("------- base point", base_point_mm)
    
    printer(["%.2f" %e for e in [x, y, z]], 'hand center', level=3)
    printer(h_max, 'grid_h_max', level=3)
    printer(grid_place_pos, 'grid_place_pos', level=3)
    printer(grid_oblique_pos, 'grid_oblique_pos', level=3)
    printer(oblique_pos, 'oblique_base', level=3)
    printer(rotation, 'rotation', level=3)

    return place_cm, float(x), float(y), float(z), rotation, oblique_pos, display_info



def try_to_pack(boxes, pre_heightmap, frame_id):
    
    done, info = do_pack(boxes)
    
    if done:
        pbox, pbox_mm, pbox_req_id, container_id, alg_pbox_id = info

        G["capping_cnt"] += 1
        printer('', 'packing')
        printer(pbox_req_id, 'pbox_req_id', level=3)
        printer(alg_pbox_id, 'alg_pbox_id', level=3)
        printer(container_id, 'container_id', level=3)
        printer(G["capping_cnt"], 'pack_num', level=3)
        
        printer()

        place_cm, x, y, z, rotation, oblique_pos, display_info = postprocess_packed(pbox, G["packed"], G["base_point_mm"], container_id, pre_heightmap)
        ox, oy, oz = oblique_pos

        G["packed_seq"].append(place_cm[0:6])
        # display_mm = [x, y, z] + pbox_mm
        # display_mm = place_cm[3:6] * 10

        ratio, ratio_z, max_height = get_volume_ratio(G["frame_size_cm"], G["packed"])
        printer(ratio, "ratio", level=3)
        printer(max_height, "height", level=3, code_2='*')

        ret = { 'code': 200,
            'msg': 'success',
            'data': { 
                    'frame_id': frame_id, 
                    'box_id': pbox_req_id, 
                    'rotation': rotation,
                    'point': [x, y, z], 
                    'display': [str(i) for i in display_info],
                    # 'oblique': G['top_point_mm'],
                    'oblique': [ox, oy, oz],
                }
            }
        printer(ret, 'msg', level=3)

    else:
        ratio, ratio_z, max_height = get_volume_ratio(G["frame_size_cm"], G["packed"])

        ret = {'code': 202,
            'msg': 'failed',
            'ratio': str(ratio),
            'height': str(max_height),
        }

    return done, ret

def all_over():
    printer("", 'over', '*')

    ratio, ratio_z, max_height = get_volume_ratio(G["frame_size_cm"], G["packed"])

    printer(ratio, "ratio", level=3)
    printer(max_height, "height", level=3, code_2='*')

    ret = {'code': 202,
        'msg': 'failed',
        'ratio': str(ratio),
        'height': str(max_height),
    }
    return ret

###############################################################
# flask functions

@app.route('/' + prefix + '/frame/init', methods=['POST'])
def init_frame():
    global G

    if 'task_time' in G:
        task_time = G['task_time']
    else:
        task_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    G = {}
    
    data = request.get_json()
    
    printer(data, code_2='=')

    try:
        frame_id = int(data["frame_id"])

        if 'task_id' in data:
            task_id = int(data["task_id"])
        else:
            task_id = 0
        
        frame_length_mm = float(data["frame_length"])
        frame_width_mm = float(data["frame_width"])
        frame_height_mm = float(data["frame_height"])
        # frame_size_mm = [frame_length_mm, frame_width_mm, frame_height_mm]
        frame_size_mm = [frame_width_mm, frame_length_mm, frame_height_mm]
        base_point_mm = [float(x) for x in data["base_point"]]
        
        str_time = data["time"]

        if 'task_time' in data:
            task_time = data["task_time"]
        else:
            task_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

        ret = { "code": 200, "msg": "success", "data": "0" }

    except Exception as e:
        print(e)
        return abort(404, e)
    
    G["corner_pos"] = args.corner_pos ## 
    G["inflate_size"] = args.inflate_size
    G['vision_tolerant'] = args.vision_tolerant
    G['capping_dilate'] = args.capping_dilate

    base_point_mm_list, frame_size_mm, frame_size_cm, top_point_mm = init_frame_size(frame_size_mm, base_point_mm)
    
    printer( [frame_size_cm, base_point_mm_list[0]],  'frame_size_cm, base_point_mm')
    
    G['base_point_mm'] = base_point_mm_list
    G['frame_size_mm'] = frame_size_mm
    G['frame_size_cm'] = frame_size_cm
    G['top_point_mm'] = top_point_mm
    
    G["task_id"] = task_id
    G['task_time'] = task_time

    G["init"] = True
    G["req_id"] = 0
    G["packed"] = []
    G["viewer"] = None
    G["input_seq"] = []
    G["packed_seq"] = []
    G["heightmap"] = []

    G["capping_net"] = None
    assert args.capping_method == 'pct' or args.capping_method == 'tap'
    G["capping_method"] = args.capping_method
    G["pct_init_flag"] = True

    # 手动调整的提升高度
    G["cut_height"] = 0

    G["capping_cnt"] = 0

    save_folder =  "./web_service/logs/RL/%s" % str(task_time)
    os.makedirs(save_folder, exist_ok=True)

    log_name_input = '%s/logs_%d_input_seq.npy' % (save_folder, G['task_id'])
    log_name_packed = '%s/logs_%d_packed_seq.npy' % (save_folder, G['task_id'])
    log_heightmap = '%s/logs_%d_heightmap.npy' % (save_folder, G['task_id'])

    printer(log_name_packed, 'save folder', code_2='#')

    G['log_name_input'] = log_name_input
    G['log_name_packed'] = log_name_packed
    G['log_heightmap'] = log_heightmap

    G['is_over'] = False

    return ret

def do_pack(boxes):
    preview = [bb['box_cm'] for bb in boxes]
    
    invert_xy = False
    # invert w and l because network do better when width > length

    if G["capping_net"] == None:

        printer('', 'START', '+', '+')

        cut_topo = G["packed"].copy()
        
        # TODO cut_topo is the init_heightmap
        for cc in cut_topo:
            cc[2] = int(round(cc[2]))
            cc[5] = int(round(cc[5]))
        # print("cut_topo", cut_topo)

        # input(args.resume_path_capping)
        w, l = G['frame_size_cm'][0], G['frame_size_cm'][1]
        if invert_xy:
            w, l = G['frame_size_cm'][1], G['frame_size_cm'][0]
        
        G["capping_net"] = TAPer(w, l, container_height=G['frame_size_cm'][2],
                                    preview_num=args.preview_capping, 
                                    hidden_dim=args.hidden_dim, init_ctn_num=1,
                                    checkpoint_path=[args.model1, args.model2, args.model3, args.model4], device=args.device, stable_scale_factor=0.5, use_bridge=False, 
                                    same_height_threshold=args.same_height_threshold, min_ems_width=args.min_ems_width, min_height_diff=args.min_height_diff,
                                     scale_to_large=False, gripper_size=[70,65,45],
                                     stable_predict =False, prec_type = 'none', rotate_axes = ['z'], data_type = 'rand', fact_type = 'box', world_type = 'real'
                                    )
        G["capping_net"].init_packing_state(cut_topo, reset=True)

    else:
        pass
    
    done, pbox_cm, pbox_id, container_id, extra = G["capping_net"].pack_box(preview[:args.preview_capping], allow_preview=True, check_box_stable=True, invert=invert_xy)
    # pbox_cm 包括： box_x, box_y, box_z, pos_x, pos_y, pos_z
    
    
    G["heightmap"].append(  G["capping_net"].env.container.each_container_heightmap[0] )

    if pbox_id == None:
        return False, None

    pbox_id = int(pbox_id)

    pbox_cm[5] += G["cut_height"]
    G["packed"].append(pbox_cm)

    pbox = boxes[pbox_id]['box_cm']
    pbox_mm = boxes[pbox_id]['box_mm']
    pbox_req_id = boxes[pbox_id]['box_id']
    
    return True, (pbox, pbox_mm, pbox_req_id, container_id, pbox_id)

@app.route('/' + prefix + '/frame/maduo', methods=['POST'])
def stack():
    global G

    if not G["init"]:
        return abort(401, "System has not been initialized.")

    G["req_id"] += 1
    # print(G["req_id"])

    data = request.get_json()

    printer(data, 'msg', code_2='=')
    
    try:
        frame_id = int(data["frame_id"])
        box_info = data["box_info"]
        # place_pose = data["place_pose"]  # [x,y,z, rx,ry,rz]
        # stack_info = data["stack_info"]  # [[px, py, pz, bx, by, bz],[], ...]
        str_time = data["time"]

    except Exception as e:
        print(e)
        return abort(400, e)

    if len(box_info) == 0 or G['is_over'] == True:
        printer('', 'NO box')

        if G["capping_net"] != None:
            if G["capping_method"] == 'tap':
                c, max_height, var_height = G["capping_net"].close()
            elif G["capping_method"] == 'pct':
                c = ratio_z
                var_height = 0
        else:
            c, max_height, var_height = 0,0,0

        ratio, ratio_z, max_height = get_volume_ratio(G["frame_size_cm"], G["packed"])

        printer(ratio, "ratio", level=3)
        printer(max_height, "height", level=3, code_2='*')

        ret = { 'code': 200,
            'msg': 'success',
            'data': {
                    'c': c,
                    'max_height': max_height * 1.0,
                    'var_height': var_height * 1.0
                }
            }

        return ret

    boxes = preprocess_boxes(box_info, args.item_size_set)

    if False:
        print(boxes)
        new_boxes = np.load("/home/speedbot/binpacking/web_service/logs/RL/231025-145253/logs_0_packed_seq.npy")

        sid = G["capping_cnt"]
        nb = new_boxes[sid][:3]
        if sid == 5:
            nb[0], nb[1] = nb[1], nb[0]    
        print(nb)
        boxes[0]['box_cm'] = nb
    
    if G['capping_net'] is not None:
        pre_heightmap = G["capping_net"].env.container.each_container_heightmap[0].copy()
    else:
        pre_heightmap = np.zeros(G['frame_size_cm'][:2])
    
    G["input_seq"].append([G["req_id"], boxes])

    # NOTE new
    done, ret = try_to_pack(boxes, pre_heightmap, frame_id)

    if not done:
        ret = all_over()

    # smaller container
    new_heightmap = G["capping_net"].env.container.each_container_heightmap[0].copy()

    if new_heightmap.max() > 45:
        new_size = [ G['frame_size_cm'][0] - G['inflate_size'], G['frame_size_cm'][1] - G['inflate_size'], G['frame_size_cm'][2]  ]
        new_size = np.array(new_size).astype('int')
        G['frame_size_cm'] = new_size
        G["capping_net"].env.change_container_size( new_size )
        G['inflate_size'] = 0

    print()
    
    # np.save(G["log_name_input"], G["input_seq"])
    np.save(G["log_name_packed"], G["packed_seq"])

    # np.save(G["log_heightmap"], G["heightmap"])
    printer(f"save {G['log_name_packed']}", 'stack info')

    return ret

@app.route('/' + prefix + '/frame/clearpallet', methods=['POST'])
def clear():
    ret = { "code": 200, "msg": "success", "data": "" }

    print(request.data, request.access_route, ret)
    return ret

@app.route('/' + prefix + '/frame/ratio', methods=['POST'])
def ratio():
    ret = { "code": 200, "msg": "success", "data": "" }

    print(request.data, request.access_route, ret)
    return ret

@app.route('/' + prefix + '/frame/status', methods=['POST'])
def status():
    ret = { "code": 200, "msg": "success", "data": "" }

    print(request.data, request.access_route, ret)
    return ret

@app.route('/' + prefix + '/frame/delete', methods=['POST'])
def delete_frame():
    print(request.data, request.access_route)
    ret = { "code": 200, "msg": "success", "data": "" }

    print(request.data, request.access_route, ret)
    return ret

@app.route('/' + prefix + '/frame/success', methods=['POST'])
def success_frame():
    print(request.data, request.access_route)
    ret = { "code": 200, "msg": "success", "data": "" }

    print(request.data, request.access_route, ret)
    return ret

if __name__ == '__main__':
    print()
    registration_envs()
    args = get_args()

    # TODO maybe we dont need item_size
    # assert args.item_size_set != None

    printer(args, code_2='=')
    print()
    app.run(host='0.0.0.0', port=8100, threaded=True)
