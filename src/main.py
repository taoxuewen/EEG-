"""
EEG数据分析 - 主入口
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    print("EEG数据分析 项目已就绪")
    # 示例：生成模拟EEG信号
    fs = 256  # 采样频率 (Hz)
    duration = 2  # 时长 (秒)
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    # 10Hz alpha波 + 噪声
    signal = np.sin(2 * np.pi * 10 * t) + 0.3 * np.random.randn(len(t))

    print(f"生成了 {len(t)} 个样本点，采样率 {fs}Hz")
    print(f"信号均值: {signal.mean():.4f}, 标准差: {signal.std():.4f}")


if __name__ == "__main__":
    main()
