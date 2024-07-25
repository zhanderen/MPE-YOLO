import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

def plot_results_model(dir=''):
    file = r'C:\Users\qinyi\Desktop\ultralyticsPro0515\ultralytics\script\print_plot\YOLOv8s.csv' # 训练完模型的results.csv的完整路径，建议放同一个文件夹下面
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('*.csv')) # 其他对比模型的csv文件放在同一个文件夹下。命名格式比如model_01.csv、model_02.csv
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype('float')
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / '将多个results文件的图画出来.png', dpi=1000) # merge models
    plt.close()

if __name__ == '__main__':
    plot_results_model()