import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import math
import random
plt.rcParams['font.family']=['SimHei']
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False

CENTER = np.array((0, 0))
R = 50
NIL = -1

TARGET_COORDS = [  (0, 0), # point zero
            (86.60, 330),
            (50, 0),
            (50, 300),
            (50, 60),
            (0, 0),
            (50, 240),
            (86.6025, 90),
            (50, 120),
            (50, 180),
            (86.6025, 210),
            (132.2875, 100.893),
            (100, 120),
            (86.6025, 150),
            (100, 180),
            (132.2875, 199.107)
            ]

COORDS = [  (0, 0), # point zero
            (85, 330.20),
            (50, 0),
            (54, 299.61),
            (49, 60.10),
            (0, 0),
            (56, 240.41),
            (85, 89.63),
            (50, 120.27),
            (53, 180.48),
            (88, 209.55),
            (137, 100.87),
            (99, 120.34),
            (84, 150.23),
            (104, 179.41),
            (137, 198.56)
]

COORDS_2 = [
            (0,0), 
            (80, 333.20),
            (50, 0),
            (63, 297.61),
            (41, 63.10),
            (0, 0),
            (61, 243.41),
            (80, 84.63),
            (54, 122.27),
            (63, 184.48),
            (91, 207.55),
            (147, 100.87),
            (92, 125.34),
            (80, 155.23),
            (107, 176.41),
            (141, 196.56),

]

def rad2dec(x): # 极坐标转为直角坐标
    r, theta = x[0], x[1]
    return np.array((r*np.cos(theta), r*np.sin(theta)))

def dec2rad(v):
    x, y = v[0], v[1]
    r = np.sqrt(x * x + y * y), 
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2 * np.pi
    return np.array((r, theta), dtype=object)


def plot_dots(lst, mk=False, mk_offset=0, **kwargs):
    x = [v[0] for v in lst]
    y = [v[1] for v in lst]
    plt.scatter(x, y, **kwargs)
    if mk:
        marks = [str(i+mk_offset) for i in range(0, len(x))]
        for i in range(len(x)):
            plt.annotate(marks[i], xy = (x[i], y[i]), xytext = (x[i]+0.5, y[i]+0.5))

def calc_angle(v1, v2):
    cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(cos_angle)
    if theta < 0:
        theta += np.pi
    return theta

def calc_pos(i, j, k, a1, a2, b1, b2):
    print(f"i_:{i}, j_:{j}, k_:{k}")
    # case 1
    # assert k != 0, "k should be 1~9"
    # print(math.degrees(a1), math.degrees(a2), math.degrees(b1), math.degrees(b2),sep=',')
    b = b2 - b1
    yl0 = np.sin(a1)*np.sin(a2)
    yr0 = np.sin(b + a2)*np.sin(a1)
    yr1 = np.sin(b - a2)*np.sin(a1)
    xl0 = np.cos(a1)*np.sin(a2)
    xr0 = np.cos(b+a2)*np.sin(a1)
    xr1 = np.cos(b-a2)*np.sin(a1)
    
    is_inside = i < k < j
    # gt_180_cnt = 0
    if np.abs(j - k) > 3:
        is_inside = not is_inside
        # gt_180_cnt += 1
    if np.abs(k - i) > 3:
        is_inside = not is_inside
        # gt_180_cnt += 1
    
    up_down = k in ((i)%6+1, (i+1)%6+1)
    # if gt_180_cnt == 2:
    #     up_down = not up_down
    
    if up_down:
        yl, xl = yl0, xl0
        if is_inside:
            print("case 1")
            yr, xr = -yr0, xr0
        else:
            yr, xr = yr1, -xr1
            print("case 2")
        kappa = (yl + yr) / (xl + xr)
        print(f"k:{kappa}")
        theta_hat = -np.arctan(kappa) if kappa <= 0 else np.pi - np.arctan(kappa)
        print(f"thetahat{math.degrees(theta_hat)}")
        d =  R * np.sin(theta_hat + a1) / np.sin(a1)
        theta = theta_hat + b1
        if theta < 0:
            theta += 2 * np.pi
    else:
        yl, xl = yl0, -xl0
        if is_inside:
            yr, xr = yr1, -xr1
            print("case 3")
        else:
            yr, xr = -yr0, xr0
            print("case 4")
        kappa = (yl + yr) / (xl + xr)
        # print(f"kappa:{kappa}")
        theta_hat = np.pi + np.arctan(kappa) if kappa <= 0 else np.arctan(kappa)
        d =  R * np.sin(theta_hat + a1) / np.sin(a1)
        theta = b1 - theta_hat
        if theta < 0:
            theta += 2 * np.pi
    # y = (np.sin(a1-b1)*np.sin(a2) - np.sin(a2+b2)*np.sin(a1))
    # x = np.cos(a1-b1)*np.sin(a2) + np.cos(a2+b2)*np.sin(a1)
    
    # print("theta:",theta)
    return d, theta
    
class DroneGroupTri:
    def __init__(self, coords_target, coords) -> None:
        self.N = len(coords) - 1
        self.pos = [rad2dec((c[0], math.radians(c[1]))) for c in coords] # 极坐标
        self.target = [rad2dec((c[0], math.radians(c[1]))) for c in coords_target]
    
    def calc_delta(self, i, j, k, ref1, ref2, ref0, mapping, debug=False):
        print(i,j,k)
        p_real = self.pos[k]
        p_tgt = self.target[k]
        a1 = calc_angle(ref0 - p_real, ref1 - p_real)
        a2 = calc_angle(ref0 - p_real, ref2 - p_real)
        b1 = dec2rad(ref1 - ref0)[1]
        b2 = dec2rad(ref2 - ref0)[1]
        print(math.degrees(a1), math.degrees(a2), math.degrees(b1), math.degrees(b2),sep=',')
        i_, j_, k_ = mapping[i], mapping[j], mapping[k]
        r_, theta_ = calc_pos(i_, j_, k_, a1, a2, b1, b2)
        # if debug:
        #     print(f"r_:{r_}, θ: {math.degrees(theta_)}")
        p_est = rad2dec((r_, theta_)) + ref0 # 估计的坐标（笛卡尔）
        if debug:
            print(f"r_:{r_}, theta_:{math.degrees(theta_)}")
            print(f"Estimated: {p_est}, actual: {p_real}, target: {p_tgt}")
        delta = p_tgt - p_est
        if np.linalg.norm(delta) > 2e1:
            raise ValueError("Prediction Error!")
        return delta
    
    def one_hex(self, rounds, center, ref, around, lr, color, has_legend=False):
        mapping = dict()
        for i in range(6):
            mapping[around[i]] = i + 1
        dots_x, dots_y = self.adjust_random(rounds, center, ref, around, mapping, lr)
        # for i in range(1, DG.N+1):
        #     plt.plot(dots_x[i], dots_y[i], marker='.', color=color, markersize=0.5)
        #     # plt.plot(dots_x[i][-1], dots_y[i][-1], marker='*', color='cyan')
        for i in range(1, self.N+1):
            label1 = '调整移动轨迹' if i == 1 and has_legend else None 
            label2 = '初始位置' if i ==1 and has_legend else None
            label3 = '最终位置' if i ==1 and has_legend else None
            plt.plot(dots_x[i], dots_y[i], marker='.', markersize=2, color='blue', label=label1)
            plt.plot(dots_x[i][0], dots_y[i][0], marker='+', markersize=7, color='orange', label=label2)
            plt.plot(dots_x[i][-1], dots_y[i][-1], marker='+', markersize=10, color='cyan', label=label3)
        return dots_x, dots_y
    
        
    def traingle_adjust(self, rounds, lr=0.5):
        center = 5
        around = [2, 4, 8, 9, 6, 3]
        ref = 2
        
        
        dots_x, dots_y = self.one_hex(rounds, center, ref, around, lr, color='blue')
        X, Y = dots_x, dots_y
        # done = input("done!------------------------------------")
        
        center = 8
        around = [4, 7, 12, 13, 9, 5]
        ref = 4
        dots_x, dots_y = self.one_hex(rounds, center, ref, around, lr, color='green')
        for i in range(1, self.N+1):
            X[i] += dots_x[i]
            Y[i] += dots_y[i]
        
        center = 9
        around = [5, 8, 13, 14, 10, 6]
        ref = 5
        dots_x, dots_y = self.one_hex(rounds, center, ref, around, lr, color='purple', has_legend=True)
        for i in range(1, self.N+1):
            X[i] += dots_x[i]
            Y[i] += dots_y[i]
        # 调整三个角上的
        # k 是三角形角上的点
        def adjust_angle(i, j, k, center, mapping, color):
            delta = self.calc_delta(i, j, k, self.pos[i], self.pos[j], ref0=self.pos[center], mapping=mapping, debug=True)
            print(delta)
            self.pos[k] += 1.0 * delta
            dots_x[k].append(self.pos[k][0])
            dots_y[k].append(self.pos[k][1])
            plt.plot(dots_x[k], dots_y[k], marker='.', color=color, markersize=0.5)
        
        # adjust 15
        i, j, k = 9, 14, 15
        center = 10
        mapping = {6:1, 9:2, 14:3, 15:4}
        if mapping[i] > mapping[j]:
            i, j = j, i
        adjust_angle(i, j, k, center, mapping, color='pink')
        X[k] += dots_x[k]
        Y[k] += dots_y[k]
        
        # adjust 11
        i, j, k = 8, 7, 11
        center = 12
        mapping = {7:1, 11:2, 13:5, 8:6}
        if mapping[i] > mapping[j]:
            i, j = j, i
        adjust_angle(i, j, k, center, mapping, color='pink')
        X[k] += dots_x[k]
        Y[k] += dots_y[k]
        
        # adjust 1
        i, j, k = 2, 5, 1
        center = 3
        mapping = {1:1, 2:2, 5:3, 6:4}
        if mapping[i] > mapping[j]:
            i, j = j, i
        adjust_angle(i, j, k, center, mapping, color='pink')
        X[k] += dots_x[k]
        Y[k] += dots_y[k]
        
        
        for i in range(1, self.N+1):
            plt.plot(dots_x[i][-1], dots_y[i][-1], marker='*', color='cyan')   
        
        plt.legend()
        plt.show()
        
        plt.clf()
        
        n = self.N
        for i in range(1, n+1):
            ep = len(X[i])
            t = np.array(list(range(ep))) * 6
            print(DG.target[i])
            print(np.array((X[i][-1],Y[i][-1])))
            dist = [np.linalg.norm(np.array((X[i][j],Y[i][j])) - DG.target[i]) for j in range(ep)]
            plt.plot(t, dist, label=f"FY0{i}")    


        plt.xlabel("迭代次数t")
        plt.ylabel("无人机当前位置和目标位置的距离d")
        plt.legend()
        plt.show()
        
    
    def adjust_random(self, rounds, center, ref, around, mapping, lr=0.5):
        dots_x = [[] for _ in range(self.N+1)]
        dots_y = [[] for _ in range(self.N+1)]
        for idx in range(1, self.N+1):
            dots_x[idx].append(self.pos[idx][0])
            dots_y[idx].append(self.pos[idx][1])
        tmp = around[:] # copy
        while rounds > 0:
            rounds -= 1
            random.shuffle(tmp)
            i, j, k = tmp[0], tmp[1], tmp[2]
            if k == ref:
                rounds += 1
                continue
            
            
            i_, j_, k_ = mapping[i], mapping[j], mapping[k]
            if np.abs(i_ - j_) == 3 or np.abs(j_ - k_) == 3 or np.abs(i_ -k_) == 3:
                rounds += 1
                continue
            if i_ > j_:
                i, j = j, i
            
            print(f"i:{i}, j:{j}, k:{k}")
            
            try:
                delta = self.calc_delta(i, j, k, self.pos[i], self.pos[j], ref0=self.pos[center], mapping=mapping, debug=True)
                print(f"delta:{delta}")
                # except Exception as e:
                #     upd = False


                # 更新
                DG.pos[k] += lr * delta

                dots_x[k].append(self.pos[k][0])
                dots_y[k].append(self.pos[k][1])
            except Exception as e:
                print(e)
                
        return dots_x, dots_y

def line_seg(v1, v2):
    x = [v1[0], v2[0]]
    y = [v1[1], v2[1]]
    plt.plot(x, y, '--', color='gray',linewidth=1)

if __name__ == "__main__":
    
    # random.seed(114514)
    # random.seed(1919)
    random.seed(2022)
    
    DG = DroneGroupTri(TARGET_COORDS, COORDS)
    print(DG.pos)
    
    # draw_circle = plt.Circle(CENTER, R, fill=False)
    # plt.gcf().gca().add_artist(draw_circle)
    # plot_dots(DG.pos, color='blue')
    plot_dots(DG.target[1:], mk=True, mk_offset=1, marker='x', s=50, color='red', label="调整目标")
    tgt = DG.target
    line_seg(tgt[1], tgt[15])
    line_seg(tgt[2], tgt[14])
    line_seg(tgt[4], tgt[13])
    line_seg(tgt[7], tgt[12])
    line_seg(tgt[1], tgt[11])
    line_seg(tgt[3], tgt[12])
    line_seg(tgt[6], tgt[13])
    line_seg(tgt[10], tgt[14])
    line_seg(tgt[11], tgt[15])
    line_seg(tgt[7], tgt[10])
    line_seg(tgt[4], tgt[6])
    line_seg(tgt[2], tgt[3])
    # circle1 = plt.Circle(tgt[5], R, fill=False, color='red')
    # circle2 = plt.Circle(tgt[8], R, fill=False, color='blue')
    # circle3 = plt.Circle(tgt[9], R, fill=False, color='green')
    
    # plt.gcf().gca().add_artist(circle1)
    # plt.gcf().gca().add_artist(circle2)
    # plt.gcf().gca().add_artist(circle3)
    
    
    DG.traingle_adjust(rounds=500)
    # print(dots_x, dots_y)
    
    # print("t", DG.target[3])
    # k = 9
    # for i in range(1,10):
    #     for j in range(1, 10):
    #         if i < j and i != k and j != k: 
    #             print(i, j)
    #             print(DG.calc_delta(i, j, k, DG.target[i], DG.target[j],debug=True))
    
    
    
    # print(DG.target)