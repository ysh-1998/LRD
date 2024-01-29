import pickle
import torch
from tqdm import tqdm
def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    index2unit = {value:key for key,value in unit2index.items()}
    return unit2index,index2unit
def load_text(file):
    item2text = {}
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            item, text = line.strip().split('\t', 1)
            item2text[item] = text
    return item2text
item2index,index2item = load_unit2index("../data/Office/Office.item2index")
item2text = load_text("../data/Office/Office.title.text")

with open('lrd_result.pkl', 'rb') as f:
    data = pickle.load(f)
all_pairs = data["history_target"]
topk = torch.tensor(data["predict"]).topk(50, dim=0)

for rel in tqdm(range(13)):
    outf = open(f"rel{rel}_item_pairs.txt","w")
    item_pair_set = set()
    for pair_index in topk[1][:,rel]:
        pair = all_pairs[pair_index]
        item_text1, item_text2 = item2text[index2item[pair[0]]], item2text[index2item[pair[1]]]
        if (item_text1,item_text2) not in item_pair_set:
            item_pair_set.add((item_text1,item_text2))
            outf.write("item1: "+item_text1+"\t""item2: "+item_text2+"\n")
