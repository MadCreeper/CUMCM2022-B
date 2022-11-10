# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import math
import random
plt.rcParams['font.family']=['SimHei']
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False

CENTER = np.array((0, 0))
R = 100

COORDS = [(0,0),
          (100,0),
          (98,40.10),
          (112,80.21),
          (105,119.75),
          (98,159.86),
          (112,199.96),
          (105,240.07),
          (98,280.17),
          (112,320.28)]

COORDS_2 = [
    (0,0),
    (100, 0),
    (125, 35.23),
    (112, 71.27),
    (87, 109.11),
    (129, 166.48),
    (72, 210.89),
    (110, 229.56),
    (94, 295.12),
    (130, 330.02)]

COORDS_3 = [(0,0),
            (100, 0),
            (129, 18.23),
            (130, 59.27),
            (79, 99.11),
            (129, 178.48),
            (72, 207.89),
            (125, 200.56),
            (79, 295.12),
            (130, 340.02)
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


def plot_dots(lst, mk=False, **kwargs):
    x = [v[0] for v in lst]
    y = [v[1] for v in lst]
    plt.scatter(x, y, **kwargs)
    if mk:
        marks = [str(i) for i in range(0, len(x))]
        for i in range(len(x)):
            plt.annotate(marks[i], xy = (x[i], y[i]), xytext = (x[i] + 5, y[i] + 5))

def calc_angle(v1, v2):
    cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(cos_angle)
    if theta < 0:
        theta += np.pi
    return theta

def calc_pos(i, j, k, a1, a2, b1, b2):
    # case 1
    assert k != 0, "k should be 1~9"
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
    if np.abs(j - k) > 4.5:
        is_inside = not is_inside
        # gt_180_cnt += 1
    if np.abs(k - i) > 4.5:
        is_inside = not is_inside
        # gt_180_cnt += 1
    
    up_down = k in ((i)%9+1, (i+1)%9+1, (i+2)%9+1, (i+3)%9+1)
    # if gt_180_cnt == 2:
    #     up_down = not up_down
    
    if up_down:
        yl, xl = yl0, xl0
        if is_inside:
            # print("case 1")
            yr, xr = -yr0, xr0
        else:
            yr, xr = yr1, -xr1
            # print("case 2")
        kappa = (yl + yr) / (xl + xr)
        theta_hat = -np.arctan(kappa) if kappa <= 0 else np.pi - np.arctan(kappa)
        d =  R * np.sin(theta_hat + a1) / np.sin(a1)
        theta = theta_hat + b1
        if theta < 0:
            theta += 2 * np.pi
    else:
        yl, xl = yl0, -xl0
        if is_inside:
            yr, xr = yr1, -xr1
            # print("case 3")
        else:
            yr, xr = -yr0, xr0
            # print("case 4")
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
    
class DroneGroup:
    def __init__(self, coords) -> None:
        self.N = len(coords) - 1
        self.pos = [rad2dec((c[0], math.radians(c[1]))) for c in coords] # 极坐标
        self.target = [np.zeros(2)] + [rad2dec((R, math.radians(w))) for w in range(0, 360, 360//self.N)]
    
    def calc_delta(self, i, j, k, ref1, ref2, ref0=CENTER,debug=False):
        p_real = self.pos[k]
        p_tgt = self.target[k]
        a1 = calc_angle(ref0 - p_real, ref1 - p_real)
        a2 = calc_angle(ref0 - p_real, ref2 - p_real)
        b1 = dec2rad(ref1)[1]
        b2 = dec2rad(ref2)[1]
        
        r_, theta_ = calc_pos(i, j, k, a1, a2, b1, b2)
        # if debug:
        #     print(f"r_:{r_}, θ: {math.degrees(theta_)}")
        p_est = rad2dec((r_, theta_)) # 估计的坐标（笛卡尔）
        if debug:
            print(f"Estimated: {p_est}, actual: {p_real}, target: {p_tgt}")
        delta = p_tgt - p_est
        if np.linalg.norm(p_est - p_tgt) > 1e2:
            raise ValueError("Prediction Error!")
        return delta
    
    # def adjust(self, k): # 调整k号无人机
    #     tot_delta = np.zeros(2)
    #     cnt = 0
    #     for i in range(1, self.N + 1):
    #         for j in range(1, self.N + 1):
    #             if i < j and i != k and j != k:
    #                 delta = self.calc_delta(self.pos[k], self.pos[i], self.pos[j])
    #                 tot_delta += delta
    #                 cnt += 1
    #     avg_delta = tot_delta / cnt
    #     return avg_delta
    
    def adjust_random(self, rounds, lr=0.5):
        REFPT = 1
        dots_x = [[] for _ in range(10)]
        dots_y = [[] for _ in range(10)]
        for idx in range(1, 10):
            dots_x[idx].append(self.pos[idx][0])
            dots_y[idx].append(self.pos[idx][1])
        tmp = list(range(1,10))
        while rounds > 0:
            rounds -= 1
            random.shuffle(tmp)
            i, j, k = tmp[0], tmp[1], tmp[2]
            if k == REFPT:
                rounds += 1
                continue
            if i > j:
                i, j = j, i
            
            try:
                delta = self.calc_delta(i, j, k, self.pos[i], self.pos[j], debug=True)
            
                print(f"delta:{delta}")
                # 更新
                DG.pos[k] += lr * delta
                
                dots_x[k].append(self.pos[k][0])
                dots_y[k].append(self.pos[k][1])
            except Exception as e:
                print(e)
                
        return dots_x, dots_y
            
if __name__ == "__main__":
    
    # random.seed(114514)
    # random.seed(1919)
    random.seed(810)
    
    DG = DroneGroup(COORDS_2)
    print(DG.pos)
    
    draw_circle = plt.Circle(CENTER, R, fill=False)
    plt.gcf().gca().add_artist(draw_circle)
    # plot_dots(DG.pos, color='blue')
    plot_dots(DG.target, mk=True, marker='x',color='red',s=100,label='调整目标')
    
    EPOCHS = 1000
    
    dots_x, dots_y = DG.adjust_random(EPOCHS, lr=0.1)
    # print(dots_x, dots_y)
    for i in range(1, 10):
        label1 = '调整移动轨迹' if i == 1 else None 
        label2 = label='初始位置'if i ==1 else None
        label3 = label='最终位置'if i ==1 else None
        plt.plot(dots_x[i], dots_y[i], marker='.', markersize=2, color='blue', label=label1)
        plt.plot(dots_x[i][0], dots_y[i][0], marker='+', markersize=7, color='orange', label=label2)
        plt.plot(dots_x[i][-1], dots_y[i][-1], marker='+', markersize=10, color='cyan', label=label3)
        
    # print("t", DG.target[3])
    # k = 9
    # for i in range(1,10):
    #     for j in range(1, 10):
    #         if i < j and i != k and j != k: 
    #             print(i, j)
    #             print(DG.calc_delta(i, j, k, DG.target[i], DG.target[j],debug=True))
    
    plt.legend()
    plt.show()
    
    plt.clf()
    
    n = 9
    for i in range(1, n+1):
        ep = len(dots_x[i])
        t = np.array(list(range(ep))) * n
        print(DG.target[i])
        print(np.array((dots_x[i][-1],dots_y[i][-1])))
        dist = [np.linalg.norm(np.array((dots_x[i][j],dots_y[i][j])) - DG.target[i]) for j in range(ep)]
        plt.plot(t, dist, label=f"FY0{i}")    
    
    t = np.array(list(range(50, EPOCHS)))
    plt.plot(t, 400 * 1/(t), '--', label='d=400/t', )
    
    plt.xlabel("迭代次数t")
    plt.ylabel("无人机当前位置和目标位置的距离d")
    plt.legend()
    plt.show()
    # print(DG.target)