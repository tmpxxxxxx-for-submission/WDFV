import matplotlib.pyplot as plt
import numpy as np


class LineChart:
    def __init__(self, vec):
        self.vec = vec
        self.len = len(vec)

    def Show(self):
        # x = np.arange(img_for_line_chart.shape[0])  # 点的横坐标
        x = np.arange(self.len)

        plt.plot(x, self.vec, 'o-', color='r', label="???")  # o-:圆形

        plt.xlabel("index")  # 横坐标名字
        # plt.ylabel("intensity")  # 纵坐标名字
        plt.legend(loc="best")  # 图例
        plt.show()
