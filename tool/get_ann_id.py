import json

split = 'train'
with open('InterHand2.6M_' + split + '_data.json') as f:
    data = json.load(f)

f = open('aid_human_annot_' + split + '.txt', 'w')
for ann in data['annotations']:
    aid = ann['id']
    f.write(str(aid) + '\n')
f.close()
