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

def download_checksum(capture_id):
    # change the working directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(osp.join(current_path, capture_id))

    # download
    cmd = 'wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand/' + capture_id + '/CHECKSUM'
    os.system(cmd)

    # back to the current path
    os.chdir(current_path)

def download_frame_list(capture_id):
    # change the working directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(osp.join(current_path, capture_id))

    # download
    cmd = 'wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand/' + capture_id + '/frame_list.txt'
    os.system(cmd)
    cmd = 'wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand/' + capture_id + '/frame_list_orig.txt'
    os.system(cmd)

    # back to the current path
    os.chdir(current_path)

for capture_id in tqdm(capture_id_list):
    os.makedirs(capture_id, exist_ok=True)
    
    download_checksum(capture_id)
    download_frame_list(capture_id)


