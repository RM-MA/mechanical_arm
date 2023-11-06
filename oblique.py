############### oblique point ###############
import numpy as np

def get_oblique_point(pos, plain, size_seq):
    '''
    pos: 码放位置
    plain: 平面高度信息
    size_seq: 箱子尺寸
    '''
    x_center = int(pos[0] + int(size_seq[0] / 2))
    y_center = int(pos[1] + int(size_seq[1] / 2))

    print("center", x_center, y_center)


    box_boardline_x = [int(x_center - size_seq[0]/2 ), int(x_center + size_seq[0]/2 )]
    box_boardline_y = [int(y_center - size_seq[1]/2 ), int(y_center + size_seq[1]/2 )]

    oblique_left =      [x_center - int(size_seq[0] ),     y_center,         pos[2]]
    oblique_right =     [x_center + int(size_seq[0]) ,     y_center,         pos[2]]
    oblique_up =        [x_center,                          y_center + int(size_seq[1]),     pos[2]]
    oblique_down =      [x_center,                          y_center - int(size_seq[1]),     pos[2]]
    oblique_left_up =   [x_center - int(size_seq[0]),     y_center + int(size_seq[1]),     pos[2]]
    oblique_left_down = [x_center - int(size_seq[0]),     y_center - int(size_seq[1]),     pos[2]]
    oblique_right_up =  [x_center + int(size_seq[0]),     y_center + int(size_seq[1]),     pos[2]]
    oblique_right_down= [x_center + int(size_seq[0]),     y_center - int(size_seq[1]),     pos[2]]


    oblique_left_boardline = int(pos[0] - size_seq[0]/2 )
    oblique_right_boardline = int(pos[0] + size_seq[0]*3/2)
    oblique_up_boardline = int(pos[1] + size_seq[1]*3/2)
    oblique_down_boardline = int(pos[1] - size_seq[1]/2)

    oblique_left_box = int(pos[0])
    oblique_right_box = int(pos[0] + size_seq[0])
    oblique_up_box = int(pos[1] + size_seq[1])
    oblique_down_box = int(pos[1])
    
    print("oblique_left",oblique_left )
    print("oblique_right",oblique_right )
    print("oblique_up",oblique_up )
    print("oblique_down",oblique_down )
    print("oblique_left_up",oblique_left_up )
    print("oblique_left_down",oblique_left_down )
    print("oblique_right_up",oblique_right_up )
    print("oblique_right_down",oblique_right_down )

    judge_oblique = [0, 0, 0, 0, 0, 0, 0, 0]
    judge_boardline = [0, 0, 0, 0, 0, 0, 0, 0]

    print("cur height", pos[2])


    # left 0
    print("---left", oblique_left[0], oblique_left[1])
    try:
        if oblique_left_boardline < 0:
            judge_boardline[0] = 1
            height = np.max(plain[0:oblique_right_box , oblique_down_box: oblique_up_box])
            if height <= pos[2] + 2:
                judge_oblique[0] = 1
            else:
                judge_boardline[0] = 0
        else:
            height = np.max(plain[oblique_left_boardline:oblique_right_box , oblique_down_box : oblique_up_box]) 
            # print(height1, height2, height3)
            if height <= pos[2] + 2:
                judge_oblique[0] = 1
            print("left", height)
    except:
        raise
        try:
            height = plain[oblique_left[0] + int(size_seq[0]/2), oblique_left[1]] 
            if height <= pos[2] + 2 :
                judge_oblique[0] = 1
            print("except left", height)
        except:
            judge_oblique[0] = 1


    # right 1
    # height = plain[45, 45]
    try: 
        height = np.max(plain[oblique_left_box:oblique_right_boardline , oblique_down_box : oblique_up_box])
        print("right", height)
        # print(height)
        if height <= pos[2] + 2:
            judge_oblique[1] = 1
            if oblique_right_boardline >= plain.shape[0]:
                judge_boardline[1] = 1
    except:
        raise
        # print("right except")
        try:
            # print(oblique_right[0] - size_seq[0]/2, oblique_right[1])
            height = plain[int(oblique_right[0] - size_seq[0]/2), oblique_right[1]]
            # print("except height", height)
            if height <= pos[2] + 2:
                judge_oblique[1] = 1
            print("except right", height)
        except:
            judge_oblique[1] = 1
            judge_boardline[1] = 1


    # up 2
    try: 
        height = np.max( plain[oblique_left_box:oblique_right_box , oblique_down_box : oblique_up_boardline ] )

        if height <= pos[2] + 2:
            judge_oblique[2] = 1
            if oblique_up_boardline >= plain.shape[1]:
                judge_boardline[2] = 1
        print("up", height)
    except:
        raise
        try:
            height = plain[oblique_up[0],  int(oblique_up[1]- size_seq[1]/2)]
            if height <= pos[2] + 2:
                judge_oblique[2] = 1
            print("except up", height)
        except:
            judge_oblique[2] = 1
            judge_boardline[2] = 1

    # down 3
    try: 
        if oblique_down_boardline < 0:
            height = np.max(plain[oblique_left_box:oblique_right_box , 0 : oblique_up_box])
            if height < pos[2] + 2:
                judge_oblique[3] = 1
                judge_boardline[3] = 1
        else:
            height = np.max(plain[oblique_left_box:oblique_right_box , oblique_down_boardline : oblique_up_box])
            if height <= pos[2] + 2:
                judge_oblique[3] = 1
            print("down", height)
    except:
        raise
        try:
            height = plain[oblique_down[0], oblique_down[1] + int(size_seq[1]/2)]
            if height <= pos[2] + 2:
                judge_oblique[3] = 1
            print("execpt down", height)
        # raise
        except:
            judge_oblique[3] = 1
            judge_boardline[3] = 1

    # left - up 4
    try:
        if oblique_left_boardline < 0:
            height = np.max(plain[0:oblique_right_box , oblique_down_box: oblique_up_boardline])
        else:
            height = np.max(plain[oblique_left_boardline:oblique_right_box , oblique_down_box: oblique_up_boardline])
        if height < pos[2] + 2:
            judge_oblique[4] = 1
        print("left up", height)
    except:
        raise
        try:
            height = plain[oblique_left_up[0] + int(size_seq[0]/2), oblique_left_up[1] - int(size_seq[1]/2)]
            if height <= pos[2] + 2:
                judge_oblique[4] = 1
            print("except left up", height)
        except:
        # raise
            judge_oblique[4] = 1


    # left - down 5
    try: 
        if oblique_left_boardline < 0 and oblique_down_boardline < 0:
            height = np.max(plain[0:oblique_right_box, 0:oblique_up_box])
        elif oblique_left_boardline < 0: 
            height = np.max(plain[0:oblique_right_box, oblique_down_boardline:oblique_up_box])
        elif oblique_down_boardline < 0: 
            height = np.max(plain[oblique_left_boardline:oblique_right_box, 0:oblique_up_box])
        else:
            height = np.max(plain[oblique_left_boardline:oblique_right_box, oblique_down_boardline:oblique_up_box])

        if height <= pos[2] + 2:
            judge_oblique[5] = 1
        print("left down", height)
    except:
        raise
        try:
            height = plain[oblique_left_down[0] + int(size_seq[0]/2), oblique_left_down[1]+int(size_seq[1]/2)]
            if height <= pos[2] + 2:
                judge_oblique[5] = 1
            print("except left down", height)
        except:
            judge_oblique[5] = 1

    # right - up 6
    try: 
        height = np.max(plain[oblique_left_box:oblique_right_boardline, oblique_down_box:oblique_up_boardline])
        if height <= pos[2] + 2:
            judge_oblique[6] = 1
        print("right up", height)
    except:
        raise
        try:
            height = plain[oblique_right_up[0] - int(size_seq[0]/2), oblique_right_up[1] - int(size_seq[1]/2)]
            if height <= pos[2] + 2:
                judge_oblique[6] = 1
            print("except right up", height)
        except:
            judge_oblique[6] = 1

    # right - down 7     (need to check)
    try: 
        if  oblique_down_boardline < 0:
            height = np.max(plain[oblique_left_box:oblique_right_boardline, 0:oblique_up_box])
        else: 
            height = np.max(plain[oblique_left_box:oblique_right_boardline, oblique_down_boardline:oblique_up_box])
        if height <= pos[2] + 2:
            judge_oblique[7] = 1
        print("right down", height)
    except:
        raise
        try:
            height = plain[oblique_right_down[0]-int(size_seq[0]/2), oblique_right_down[1]+int(size_seq[1]/2)]
            if height <= pos[2] + 2:
                judge_oblique[7] = 1
            print("except right down", height)
        except: 
            judge_oblique[7] = 1



    print("judge_oblique", judge_oblique)
    # raise
    if judge_oblique[6] == 1 and judge_boardline[2] == 1 and judge_boardline[1] == 1:
        oblique_pos = oblique_right_up
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2 - 2.5
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2 - 2.5
        print("  oblique: right up")
    elif judge_oblique[7] == 1 and judge_boardline[1] == 1 and judge_boardline[3] == 1:
        oblique_pos = oblique_right_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2 - 2.5
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2 + 2.5
        print("  oblique: right down")
    elif judge_oblique[4] == 1 and judge_boardline[2] == 1 and judge_boardline[0] == 1:
        oblique_pos = oblique_left_up
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2 + 2.5
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2 - 2.5
        print("  oblique: left up")
    elif judge_oblique[5] == 1 and judge_boardline[0] == 1 and judge_boardline[3] == 1  :
        oblique_pos = oblique_left_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2  + 2.5
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2 + 2.5
        print("  oblique: left down")
    # ---
    elif judge_oblique[6] == 1 and ((judge_oblique[2] == 1 and judge_boardline[1] == 1) or (judge_boardline[2] == 1 and judge_oblique[1] == 1)):
        oblique_pos = oblique_right_up
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2 - 2.5
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2 - 2.5
        print("  oblique: right up")
    elif judge_oblique[7] == 1 and ((judge_oblique[1] == 1 and judge_boardline[3] == 1) or (judge_boardline[1] == 1 and judge_oblique[3] == 1)):
        oblique_pos = oblique_right_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2 - 2.5
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2 + 2.5
        print("  oblique: right down")
    elif judge_oblique[4] == 1 and ((judge_oblique[2] == 1 and judge_boardline[0] == 1) or (judge_boardline[2] == 1 and judge_oblique[0] == 1)):
        oblique_pos = oblique_left_up
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2 + 2.5
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2 - 2.5
        print("  oblique: left up")
    elif judge_oblique[5] == 1 and ((judge_oblique[0] == 1 and judge_boardline[3] == 1) or (judge_boardline[0] == 1 and judge_oblique[3] == 1)) :
        oblique_pos = oblique_left_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2  + 2.5
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2 + 2.5
        print("  oblique: left down")
    # oblique 45
    elif judge_oblique[6] == 1 and judge_oblique[2] == 1 and judge_oblique[1] == 1:
        oblique_pos = oblique_right_up
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2 - 2.5
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2 - 2.5
        print("  oblique: right up")
    elif judge_oblique[7] == 1 and judge_oblique[1] == 1 and judge_oblique[3] == 1:
        oblique_pos = oblique_right_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2 - 2.5
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2 + 2.5
        print("  oblique: right down")
    elif judge_oblique[4] == 1 and judge_oblique[2] == 1 and judge_oblique[0] == 1:
        oblique_pos = oblique_left_up
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2 + 2.5
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2 - 2.5
        print("  oblique: left up")
    elif judge_oblique[5] == 1 and judge_oblique[0] == 1 and judge_oblique[3] == 1  :
        oblique_pos = oblique_left_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2  + 2.5
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2 + 2.5
        print("  oblique: left down")
    # ---- boardline
    elif judge_boardline[2] == 1:
        oblique_pos = oblique_up
        print("  oblique:  up")
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2
    elif judge_boardline[1] == 1:
        oblique_pos = oblique_right
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2
        print("  oblique: right")
    elif judge_boardline[3] == 1:
        oblique_pos = oblique_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2
        print("  oblique: down")
    elif judge_boardline[0] == 1:
        oblique_pos = oblique_left
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2
        print("  oblique: left")
    # ------
    elif judge_oblique[2] == 1:
        oblique_pos = oblique_up
        print("  oblique:  up")
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[1] = oblique_pos[1] - int(size_seq[1])/2
    elif judge_oblique[1] == 1:
        oblique_pos = oblique_right
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] - int(size_seq[0])/2
        print("  oblique: right")
    elif judge_oblique[3] == 1:
        oblique_pos = oblique_down
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[1] = oblique_pos[1] + int(size_seq[1])/2
        print("  oblique: down")
    elif judge_oblique[0] == 1:
        oblique_pos = oblique_left
        if [int(size_seq[0]), int(size_seq[1])] != [30, 30]:
            oblique_pos[0] = oblique_pos[0] + int(size_seq[0])/2
        print("  oblique: left")
    else:
        oblique_pos = [x_center, y_center, pos[2]]

    # 四周都是障碍物
    if all(item == 0 for item in judge_oblique):
        oblique_pos = [x_center, y_center, pos[2]]
        print("  oblique: all 0")
        print("  stright")

    # 四周很空旷
    if all(item == 1 for item in judge_oblique):
        oblique_pos = [x_center, y_center, pos[2]]
        print("  oblique: all 1")
        print("  stright")

    oblique_center_pos = []
    # oblique_center_pos.append(int(oblique_pos[0] - int(size_seq[0] / 2)))
    # oblique_center_pos.append(int(oblique_pos[1]- int(size_seq[1] / 2)))

    
    oblique_center_pos.append(int(oblique_pos[0]))
    oblique_center_pos.append(int(oblique_pos[1]))
    oblique_center_pos.append(int(oblique_pos[2] + size_seq[2]))

    print('    obl corner: ', int(oblique_pos[0] - int(size_seq[0] / 2)), int(oblique_pos[1] - int(size_seq[1] / 2)), int(oblique_pos[2] + size_seq[2]))
    
    oblique_center_pos = np.array(oblique_center_pos)
    print("------ size_seq", size_seq)
    print("------ pos", x_center, y_center)
    print("------ oblique", oblique_pos[0], oblique_pos[1])

    return oblique_center_pos

	