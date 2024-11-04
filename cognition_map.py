import os 
import numpy as np
from PIL import Image

def merge_map(semseg_path, rational_path):
    semseg_img = Image.open(semseg_path).convert('RGB')
    rational_img = Image.open(rational_path).convert('RGB')

    semseg_array = np.array(semseg_img)
    rational_array = np.array(rational_img)
    merge_array = np.add(semseg_array, rational_array)
    # 保存
    merge_img = Image.fromarray(merge_array)
    merge_img.save(semseg_path.replace('semseg', 'merge'))
    print('OK')


def split_map(merge_path):
    if os.path.exists(merge_path):
        merge_img = Image.open(merge_path).convert('RGB')
        merge_array = np.array(merge_img)
        # 获取语义分割地图
        semseg_array = np.where(merge_array >= 125, merge_array - 125, merge_array - 1)
        rational_array = np.subtract(merge_array, semseg_array)
        # 保存图片
        semseg_img = Image.fromarray(semseg_array)
        semseg_img.save(merge_path.replace('merge', 'semseg'))
        rational_img = Image.fromarray(rational_array)
        rational_img.save(merge_path.replace('merge', 'rational'))
        print('OK')
    else:
        print('{} -> 文件不存在'.format(merge_path))

if __name__ == "__main__":
    # semseg_path = 'running/2024-01-05_V2.0_rational_k1k2_1/visual/bedroom_night_nobody_passable_00000144_semseg.png'
    # rational_path = 'running/2024-01-05_V2.0_rational_k1k2_1/visual/bedroom_night_nobody_passable_00000144_rational.png'
    semseg_path = 'running/2024-04-11_V2.0_rational_e3000/visual/balcony_7_rational.png'
    rational_path = 'running/2024-04-11_V2.0_rational_e3000/visual/balcony_7_semseg.png'
    merge_map(semseg_path, rational_path)
    merge_path = '/mnt/dt01/datasets/indoor_rational_data_v2/merge/balcony_7.png'
    split_map(merge_path)

