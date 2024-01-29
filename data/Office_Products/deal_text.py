import gzip
import pandas as pd
import re
import html
from tqdm import tqdm
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
meta_df = get_df("/home/ysh/project/KDA_LRD/data/Office/meta_Office_Products.json.gz")


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'["\n\r]*', '', text)
    return text
outf = open("/home/ysh/project/KDA_LRD/data/Office/Office.title.text", "w")
with open("/home/ysh/project/KDA_LRD/data/Office/Office.text") as f:
    lines = f.readlines()[1:]
    for line in tqdm(lines):
        item_id = line.split("\t")[0]
        # print(item_id)
        categories = meta_df[meta_df['asin'] == item_id]['categories']
        if len(categories) > 0:
            for cate in categories:
                if cate[0][0] == "Office Products":
                    category = clean_text(cate[0][-1])
                    break
            outf.write(item_id+"\t"+category + "\n")
        else:
            outf.write(item_id+"\t"+"" + "\n")