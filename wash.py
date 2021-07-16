'''
STEP 0 : 数据清洗
'''
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from numpy.core.numeric import NaN

f_path = 'isear_v2/'

def add_blank(matched):
    return ' ' + str(matched.group()) + ' '

def wash(sentence):
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", ' ', sentence)
    sentence = re.sub(r"[(),!?\'\`]", add_blank, sentence)
    sentence = re.sub(r"\s{2,}", ' ', sentence)
    return sentence.strip().lower()

def operate(dataset):
    res1, res2 = [], []
    labels = dataset['label']
    sentences = dataset['sentence']
    for s in tqdm(sentences):
        res2.append(wash(s))
    for l in labels:
        res1.append(l)
    return res1, res2

if __name__ == "__main__":
    train = pd.read_csv(f_path + 'isear_train.csv', index_col=0, sep=',', error_bad_lines=False)
    valid = pd.read_csv(f_path + 'isear_valid.csv', index_col=0, sep=',', error_bad_lines=False)
    test = pd.read_csv(f_path + 'isear_test.csv', index_col=0, sep=',', error_bad_lines=False)

    labels, sentences = operate(valid)
    valid_df = pd.DataFrame({
        'label' : labels,
        'sentence' : sentences
    })
    valid_df.to_csv(f_path + 'valid.csv')

    labels, sentences = operate(test)
    test_df = pd.DataFrame({
        'label' : labels,
        'sentence' : sentences
    })
    test_df.to_csv(f_path + 'test.csv')

    labels, sentences = operate(train)
    train_df = pd.DataFrame({
        'label' : labels,
        'sentence' : sentences
    })
    train_df.to_csv(f_path + 'train.csv')