# import random
import numpy as np
import json

basic_color = [
    [0.98, 0.02, 0.02], # red
               [0.02, 0.98, 0.02], # green
               [0.02, 0.02, 0.98], # blue
               [0.98, 0.98, 0.02], # yellow
               [0.98, 0.02, 0.98], # purple
               [0.02, 0.98, 0.98], # cyan   
               [0.40, 0.02, 0.02], # dark red
               [0.98, 0.50, 0.02], # gold
               [0.54, 0.16, 0.82], # voilet
               [0.10, 0.10, 0.60], # navi blue
               [0.02, 0.02, 0.02], # black
               [0.3, 0.3, 0.3], # gray
               [0.7, 0.7, 0.7], # white
               ]
color_dist = 0.01

VIEWPORT_W = 1024
VIEWPORT_H = 768

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def render(viewer, pbox):
    # import open3d as o3d
    import time

    return

    # print(viewer, render_mode)

    if viewer is None:
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(
            width=VIEWPORT_W * 2, height=VIEWPORT_H * 2,
            window_name='bin packing'
        )
    
    mesh = o3d.geometry.TriangleMesh()
    ob = mesh.create_box(pbox[0], pbox[1], pbox[2])
    ob.compute_vertex_normals()
    if len(pbox) == 7:
        color = basic_color[pbox[6]] + np.random.uniform(-color_dist, color_dist, size=[1, 3])[0]
        color = [max(min(1, i), 0) for i in color]
        ob.paint_uniform_color(color)
    else:
        color = 1 - np.random.uniform(0.0, 1.0, size=[1, 3])
        ob.paint_uniform_color(color[0])
    ob.translate((pbox[3], pbox[4], pbox[5]))
    
    viewer.add_geometry(ob)
    # viewer.update_geometry(mesh)

    viewer.poll_events()
    # viewer.update_renderer()

    time.sleep(0.4)
    return viewer

def render_onebyone(boxes):
    viewer = None
    for box in boxes:
        viewer = render(viewer, box)
    
    viewer.close()

def get_draw_box_obj(bin_size, colorid=0):
    import open3d as o3d
    
    bx = bin_size[0]
    by = bin_size[1]
    bz = bin_size[2]
    points = [[0, 0, 0], [bx, 0, 0], [0, by, 0], [bx, by, 0], [0, 0, bz], [bx, 0, bz],
          [0, by, bz], [bx, by, bz]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [basic_color[colorid] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    text = "BIN:%dx%dx%d" % (bx, by, bz)
    tt = o3d.t.geometry.TriangleMesh.create_text(text, depth=1)
    # tt.paint_uniform_color((1,0,0))
    ss = 0.4
    tt.scale(ss, (0, 0, 0))
    tt.translate((0, by, bz))
    return line_set, tt

def render_all(boxes, bin_size=None):
    # return
    
    import open3d as o3d
    import open3d.visualization as vis

    draw_obj = []
    
    if bin_size:
        bin_obj, text_obj = get_draw_box_obj(bin_size, -1)
        draw_obj.append(bin_obj)
        draw_obj.append(text_obj)

    for id, pbox in enumerate(boxes):
        # print(pbox)
        ob = o3d.geometry.TriangleMesh.create_box(pbox[0], pbox[1], pbox[2])
        ob.compute_vertex_normals()
        if len(pbox) == 7:
            color = basic_color[pbox[6]] + np.random.uniform(-color_dist, color_dist, size=[1, 3])[0]
            color = [max(min(1, i), 0) for i in color]
            ob.paint_uniform_color(color)
        else:
            color = 1 - np.random.uniform(0.0, 1.0, size=[1, 3])
            ob.paint_uniform_color(color[0])
        ob.translate((pbox[3], pbox[4], pbox[5]))
        draw_obj.append(ob)

        # text = "#%d:h%.2f" % (id, pbox[2]+pbox[5])
        text = "#%d:%dx%dx%d" % (id, pbox[0], pbox[1], pbox[2])
        tt = o3d.t.geometry.TriangleMesh.create_text(text, depth=1)
        ss = 0.2
        tt.translate((pbox[3], pbox[4]+15, pbox[5]+pbox[2]))
        tt.scale(ss, (pbox[3], pbox[4], pbox[5]+pbox[2]))
        # tt.paint_uniform_color((0,0,0))
        draw_obj.append(tt)
        text = "@%dx%dH%d" % (pbox[3], pbox[4], pbox[2]+pbox[5])
        tt = o3d.t.geometry.TriangleMesh.create_text(text, depth=1)
        ss = 0.2
        tt.translate((pbox[3], pbox[4], pbox[5]+pbox[2]))
        tt.scale(ss, (pbox[3], pbox[4], pbox[5]+pbox[2]))
        # tt.paint_uniform_color((0,0,0))
        draw_obj.append(tt)
    vis.draw(draw_obj)