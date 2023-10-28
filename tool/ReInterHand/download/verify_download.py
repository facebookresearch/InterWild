import os
import os.path as osp
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
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

for capture_id in capture_id_list:
    print('Checking ' + capture_id)
    
    os.chdir(osp.join(current_path, capture_id))
    checksum_filename = 'CHECKSUM'
    if not osp.isfile(checksum_filename):
        print(osp.join(current_path, capture_id, checksum_filename) + ' is missing. Please download it again.')
        exit()
    with open(checksum_filename) as f:
        checksums = f.readlines()

    good = True
    results = []
    for line in tqdm(checksums):
        filename, md5sum = line.split()

        if not osp.isfile(filename):
            good = False
            results.append(filename + ': missing. Please download it again.')
            continue
        
        os.system('md5sum ' + filename + ' > YOUR_CHECKSUM')
        with open('YOUR_CHECKSUM') as f:
            md5sum_yours, _ = f.readline().split()
        os.system('rm YOUR_CHECKSUM')
        
        if md5sum == md5sum_yours:
            results.append(filename + ': md5sum is correct.')
        else:
            good = False
            results.append(filename + ': md5sum is wrong. Please download it again.')

    if good:
        print('All of downloaded files are verified.')
    else:
        print('Some of downloaded files are not verified.')

    result_path = 'download_verify_results.txt'
    with open(result_path, 'w') as f:
        for result in results:
            f.write(result + '\n')
    print('The verification results are saved in ' + osp.join(current_path, capture_id, result_path))
    os.chdir(current_path)

