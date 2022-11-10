from tkinter import INSIDE
from turtle import up
import matplotlib.pyplot as plt
import numpy as np
import math
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


def plot_dots(lst, **kwargs):
    x = [v[0] for v in lst]
    y = [v[1] for v in lst]
    plt.scatter(x, y, **kwargs)

def calc_angle(v1, v2):
    cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(cos_angle)
    if theta < 0:
        theta += np.pi
    return theta

def calc_pos(i, j, k, a1, a2, b1, b2):
    # case 1
    assert k != 0, "k should be 1~9"
    print(math.degrees(a1), math.degrees(a2), math.degrees(b1), math.degrees(b2),sep=',')
    yl0 = np.sin(a1-b1)*np.sin(a2)
    yl1 = np.sin(a1+b1)*np.sin(a2)
    yr0 = np.sin(a2-b2)*np.sin(a1)
    yr1 = np.sin(a2+b2)*np.sin(a1)
    
    xl0 = np.cos(a1-b1)*np.sin(a2)
    xl1 = np.cos(a1+b1)*np.sin(a2)
    xr0 = np.cos(a2-b2)*np.sin(a1)
    xr1 = np.cos(a2+b2)*np.sin(a1)
    is_inside = i < k < j
    gt_180_cnt = 0
    if np.abs(j - k) > 4.5:
        is_inside = not is_inside
        gt_180_cnt += 1
    if np.abs(k - i) > 4.5:
        is_inside = not is_inside
        gt_180_cnt += 1
    
    up_down = k in (1, 2, 3, 4, 5)
    # if gt_180_cnt == 2:
    #     up_down = not up_down
    
    if up_down:
        yl, xl = yl0, xl0
        if is_inside:
            yr, xr = -yr1, xr1
            print("case 1")
        else:
            yr, xr = -yr0, -xr0
            print("case 2")
    else:
        yl, xl = yl1, -xl1
        is_inside = i < k < j
        if np.abs(j - k) > 4.5:
            is_inside = not is_inside
        if np.abs(k - i) > 4.5:
            is_inside = not is_inside
        if is_inside:
            yr, xr = -yr0, -xr0
            print("case 3")
        else:
            yr, xr = -yr1, xr1
            print("case 4")
    
    # y = (np.sin(a1-b1)*np.sin(a2) - np.sin(a2+b2)*np.sin(a1))
    # x = np.cos(a1-b1)*np.sin(a2) + np.cos(a2+b2)*np.sin(a1)
    y, x = yl + yr, xl + xr
    # print(y, x)
    theta = - np.arctan(y/x) if y / x <= 0 else np.pi - np.arctan(y/x)
    # print("theta:",theta)
    d = R * np.sin(a1 - b1 + theta) / np.sin(a1)
    return d, theta
    
class DroneGroup:
    def __init__(self, coords) -> None:
        self.N = len(coords) - 1
        self.pos = [rad2dec((c[0], math.radians(c[1]))) for c in coords] # 极坐标
        self.target = [np.zeros(2)] + [rad2dec((R, math.radians(w))) for w in range(0, 360, 360//self.N)]
    
    def calc_delta(self, i, j, k, ref1, ref2, ref0=CENTER):
        p_real = self.pos[k]
        p_tgt = self.target[k]
        a1 = calc_angle(ref0 - p_real, ref1 - p_real)
        a2 = calc_angle(ref0 - p_real, ref2 - p_real)
        b1 = dec2rad(ref1)[1]
        b2 = dec2rad(ref2)[1]
        
        r_, theta_ = calc_pos(i, j, k, a1, a2, b1, b2)
        print(f"r_:{r_}, θ: {math.degrees(theta_)}")
        p_est = rad2dec((r_, theta_)) # 估计的坐标（笛卡尔）
        
        print(f"Estimated: {p_est}, actual: {p_real}, target: {p_tgt}")
        delta = p_tgt - p_est
        return delta
    
    def adjust(self, k): # 调整k号无人机
        tot_delta = np.zeros(2)
        cnt = 0
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                if i < j and i != k and j != k:
                    delta = self.calc_delta(self.pos[k], self.pos[i], self.pos[j])
                    tot_delta += delta
                    cnt += 1
        avg_delta = tot_delta / cnt
        return avg_delta
        
        
if __name__ == "__main__":
    DG = DroneGroup(COORDS)
    print(DG.pos)
    
    draw_circle = plt.Circle(CENTER, R, fill=False)
    plt.gcf().gca().add_artist(draw_circle)
    plot_dots(DG.pos, color='blue')
    plot_dots(DG.target, color='green')
    
    # print("t", DG.target[3])
    k = 2
    for i in range(1,10):
        for j in range(1, 10):
            if i < j and i != k and j != k: 
                print(i, j)
                print(DG.calc_delta(i, j, k, DG.target[i], DG.target[j]))
    plt.show()
    
    
    # print(DG.target)