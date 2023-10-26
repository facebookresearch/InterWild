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
        'm--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands': ['aa'],
        'm--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands': ['aa'],
        'm--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221110--1033--TQH976--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221111--0944--JFQ550--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20230313--1433--TXB805--pilot--ProjectGoliath--Hands--two-hands': ['aa'],
        'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands': ['aa'],
        'm--20230317--1433--TRO760--pilot--ProjectGoliath--Hands--two-hands': ['aa']
}

def download(capture_id):
    # change the working directory
    os.makedirs(osp.join(capture_id, 'mano_fits'), exist_ok=True)
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(osp.join(current_path, capture_id, 'mano_fits'))

    # download
    for name in name_list[capture_id]:
        cmd = 'wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand/' + capture_id + '/mano_fits/mano_fits.tar.gz' + name
        os.system(cmd)

    # decompress
    cmd = 'cat mano_fits.tar.gz* | tar zxvf -'
    os.system(cmd)

    # back to the current path
    os.chdir(current_path)

for capture_id in tqdm(capture_id_list):
    os.makedirs(capture_id, exist_ok=True)
    
    download(capture_id)


