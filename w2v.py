'''
STEP 3 : 导入语料库并构建word2vec词向量
'''
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Word2Vec
num_features = 128
min_word_count = 1
num_workers = 4
context = 10
downsampling = 1e-3

sentences = LineSentence('vec/sentences.txt')

print('训练模型中...')
model = Word2Vec(sentences, workers=num_workers,
                 size=num_features, min_count=min_word_count,
                 window=context, sample=downsampling)
print('模型训练完毕')

print('保存模型...')
model.init_sims(replace=True)
model_name = 'vec/my_w2v_' + str(num_features) + '.w2v'
model.wv.save_word2vec_format(model_name, binary=False)
print('模型保存完毕')
