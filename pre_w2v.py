'''
STEP 1 : 构建词向量前，导入csv并处理成一行一词的语料库 
'''
import numpy as np
import pandas as pd
from numpy.core.numeric import NaN
from tqdm import tqdm

# 没有用停用词

def to_wordlist(txt):
    return txt.split()

f_path = 'isear_v2/'

train = pd.read_csv(f_path + 'train.csv', sep=',', error_bad_lines=False)
valid = pd.read_csv(f_path + 'valid.csv', sep=',', error_bad_lines=False)

sentences = []
text = train['sentence'] + valid['sentence']

for t in tqdm(text):
    if (type(t) == str):
        try:
            sentences += to_wordlist(t)
        except Exception as e:
            print('Exception:', e)
            exit()

print('数据载入并预处理完毕')
print('len(sentences):', len(sentences))

f = open('vec/sentences.txt', 'w', encoding='utf-8')

print('写入数据...')
for s in tqdm(sentences):
    f.write(s + '\n')
f.close()

print('数据写入完毕')
