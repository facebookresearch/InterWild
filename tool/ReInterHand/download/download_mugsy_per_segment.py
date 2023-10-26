import os
import os.path as osp
from tqdm import tqdm

capture_id_list = [
        'm--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands',
        'm--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands',
        'm--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221110--1033--TQH976--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221111--0944--JFQ550--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20230313--1433--TXB805--pilot--ProjectGoliath--Hands--two-hands',
        'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands',
        'm--20230317--1433--TRO760--pilot--ProjectGoliath--Hands--two-hands'
]

name_list = {
        'm--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl'],
        'm--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av'],
        'm--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs'],
        'm--20221110--1033--TQH976--pilot--ProjectGoliathScript--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at'],
        'm--20221111--0944--JFQ550--pilot--ProjectGoliathScript--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at'],
        'm--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as'],
        'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au'],
        'm--20230313--1433--TXB805--pilot--ProjectGoliath--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg'],
        'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd'],
        'm--20230317--1433--TRO760--pilot--ProjectGoliath--Hands--two-hands': ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd']
}

def download_cam_params(capture_id):
    # change the working directory
    os.makedirs(osp.join(capture_id, 'Mugsy_cameras'), exist_ok=True)
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(osp.join(current_path, capture_id, 'Mugsy_cameras'))

    # download
    cmd = 'wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand/' + capture_id + '/Mugsy_cameras/cam_params.json'
    os.system(cmd)

    # back to the current path
    os.chdir(current_path)

def download_images_masks(capture_id):
    # change the working directory
    os.makedirs(osp.join(capture_id, 'Mugsy_cameras'), exist_ok=True)
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(osp.join(current_path, capture_id, 'Mugsy_cameras'))

    # download
    for name in name_list[capture_id]:
        cmd = 'wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand/' + capture_id + '/Mugsy_cameras/envmap_per_segment.tar.gz' + name
        os.system(cmd)

    # decompress
    cmd = 'cat envmap_per_segment.tar.gz* | tar zxvf -'
    os.system(cmd)

    # back to the current path
    os.chdir(current_path)

for capture_id in tqdm(capture_id_list):
    os.makedirs(capture_id, exist_ok=True)

    download_cam_params(capture_id)
    download_images_masks(capture_id)


