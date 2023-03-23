import os

for seq_idx in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']:
    cmd = 'wget http://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/Hand_Hand___All_Files/' + seq_idx + '.zip'
    os.system(cmd)

for seq_idx in ['15', '16', '17', '18', '19', '20', '21']:
    cmd = 'wget http://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/Hand_Object___All_Files/' + seq_idx + '.zip'
    os.system(cmd)

cmd = 'wget http://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/MANO_compatible/IJCV16___Results_MANO___parms_for___joints21.zip'
os.system(cmd)
