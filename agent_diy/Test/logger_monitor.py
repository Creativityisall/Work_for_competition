import logging
import matplotlib.pyplot as plt

import logging
import numpy as np

def init_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():          # 防止重复初始化
        return logger

    logger.setLevel(logging.DEBUG)
    log_file = f"./Test/backup/test.log"

    # 新建 FileHandler，mode='w' 表示覆盖写
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_fmt = "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s"
    file_handler.setFormatter(logging.Formatter(file_fmt))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    return logger

class Monitor:
    def __init__(self):
        self.data = {}

    def put_data(self, data):
        """将数据存入监控器"""
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def clear(self):
        """清空监控数据"""
        self.data.clear()

    def draw(self):
        """绘制监控数据（支持多条曲线）"""
        if not self.data:
            print("No data to draw.")
            return

        plt.figure(figsize=(8, 4.5))
        for label, values in self.data.items():
            # 统一成 (x, y)
            if isinstance(values, tuple) and len(values) == 2:
                x, y = values
            else:
                y = np.asarray(values).flatten()   # 兼容 list / ndarray
                x = np.arange(len(y))              # 0,1,2...

            plt.plot(x, y, label=str(label), marker='o', markersize=2)

        plt.title("Monitor Data")
        plt.xlabel("index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def init_monitor():
    return Monitor()