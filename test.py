# STEP.C5 在测试集上检测准确率
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
from torch.autograd import Variable 
from sklearn.metrics import f1_score
import configparser as cp
import json
from model import TextCNN, GlobalMaxPool1d, LSTM, LSTM_ATT, GRU, GRU_ATT, MLP

if_cuda = False
device = ''
model_type = ''
data_path = ''
vec_path = ''
model_path = ''
batch_size = 32
embedding_dim = 128
lr = 0.001
num_epochs = 8
kernel_sizes = [3, 4, 5]
num_channels = [100, 100, 100]
hidden_size = 100
num_hidden = 128
stop_words = []
sentence, label = None, None
train, valid, test = None, None, None
vectors = None
vocab_size = 0
label_num = 0
test_iter = None

def init(type):
    print('Init...')
    global if_cuda, device
    global model_type, data_path, vec_path, model_path
    global batch_size, embedding_dim, lr, num_epochs
    global kernel_sizes, num_channels
    global hidden_size, stop_words
    global num_hidden
    print('Version of Torch:', torch.__version__)
    config_file = 'config.ini'
    con = cp.ConfigParser()
    con.read(config_file, encoding='utf-8')
    if type == 'cnn':
        cnn_items = dict(con.items('CNN'))
        batch_size = json.loads(cnn_items['batch_size'])
        kernel_sizes = json.loads(cnn_items['kernel_sizes'])
        num_channels = json.loads(cnn_items['num_channels'])
        embedding_dim = json.loads(cnn_items['embedding_dim'])
        lr = json.loads(cnn_items['lr'])
        num_epochs = json.loads(cnn_items['num_epoches'])
    elif type == 'lstm':
        lstm_items = dict(con.items('LSTM'))
        batch_size = json.loads(lstm_items['batch_size'])
        hidden_size = json.loads(lstm_items['hidden_size'])
        embedding_dim = json.loads(lstm_items['embedding_dim'])
        lr = json.loads(lstm_items['lr'])
        num_epochs = json.loads(lstm_items['num_epoches'])
    elif type == 'lstm_attention':
        lstm_items = dict(con.items('LSTM_ATT'))
        batch_size = json.loads(lstm_items['batch_size'])
        hidden_size = json.loads(lstm_items['hidden_size'])
        embedding_dim = json.loads(lstm_items['embedding_dim'])
        lr = json.loads(lstm_items['lr'])
        num_epochs = json.loads(lstm_items['num_epoches'])
    elif type == 'gru':
        gru_items = dict(con.items('GRU'))
        batch_size = json.loads(gru_items['batch_size'])
        hidden_size = json.loads(gru_items['hidden_size'])
        embedding_dim = json.loads(gru_items['embedding_dim'])
        lr = json.loads(gru_items['lr'])
        num_epochs = json.loads(gru_items['num_epoches'])
    elif type == 'gru_attention':
        gru_items = dict(con.items('GRU_ATT'))
        batch_size = json.loads(gru_items['batch_size'])
        hidden_size = json.loads(gru_items['hidden_size'])
        embedding_dim = json.loads(gru_items['embedding_dim'])
        lr = json.loads(gru_items['lr'])
        num_epochs = json.loads(gru_items['num_epoches'])
    elif type == 'mlp':
        mlp_items = dict(con.items('MLP'))
        batch_size = json.loads(mlp_items['batch_size'])
        num_hidden = json.loads(mlp_items['num_hidden'])
        embedding_dim = json.loads(mlp_items['embedding_dim'])
        lr = json.loads(mlp_items['lr'])
        num_epochs = json.loads(mlp_items['num_epoches'])
    else:
        return
    all_items = dict(con.items('ALL'))
    if_cuda = json.loads(all_items['if_cuda'])
    if (if_cuda):
        if torch.cuda.is_available():
            device = 'cuda:0'
            torch.cuda.set_device(device=device)
        else:
            if_cuda = False
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print('device:', device)
    data_path = all_items['data_path']
    vec_path = all_items['vec_path']
    model_path = all_items['model_path']
    with open(data_path + 'stopwords.txt', 'r') as f:
        stop_words.append(f.readline().strip('\n'))

def tokenizer(text):
    return text.split()

def set_field():
    global sentence, label
    sentence = data.Field(sequential=True,
                        lower=True,
                        tokenize=tokenizer)
    label = data.Field(sequential=False)

def set_dataset():
    print('Set Dataset...')
    global data_path, train, valid, test, label, sentence
    train, valid = data.TabularDataset.splits(
        path=data_path,
        skip_header=True,
        train='train.csv',
        validation='valid.csv',
        format='csv',
        fields=[('index', None), ('label', label), ('sentence', sentence)]
    )
    test = data.TabularDataset(
        path=data_path + 'test.csv',
        skip_header=True,
        format='csv',
        fields=[('index', None), ('label', label), ('sentence', sentence)]
    )

def build_vocab():
    print('Build Vocab...')
    global train, valid, vec_path, embedding_dim, vectors, vocab_size, label_num
    sentence.build_vocab(train, valid, vectors=Vectors(name=vec_path))
    label.build_vocab(train)
    embedding_dim = sentence.vocab.vectors.size()[-1]
    vectors = sentence.vocab.vectors
    vocab_size = len(sentence.vocab)
    label_num = len(label.vocab) - 1
    print('vocab_size:', vocab_size)
    print('label_num:', label_num)
    print('label:', label.vocab.freqs)

def build_iter():
    print('Build Iterator...')
    global batch_size, test_iter, test
    test_iter = data.Iterator(dataset=test, batch_size=len(test), shuffle=False,
                            sort=False, repeat=False, train=False, device=device,
                            sort_key=lambda x: len(x.sentence))

def load_model(net_name):
    print('Loading the model ...')
    net = torch.load(model_path + net_name, map_location=device)
    return net


# 在测试集上进行预测
def test_predict(data_iter, net):
    acc_sum, n = 0, 0
    std, res = [], []
    with torch.no_grad():
        net.eval()
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.sentence, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1) #X转置 y减1
            if isinstance(net, torch.nn.Module):
                y_hat = net(X)
                y_hat = y_hat.argmax(dim=1)
                std += y.numpy().tolist()
                res += y_hat.numpy().tolist()
                acc_sum += (y_hat == y).float().sum().item()
            n += y.shape[0]
    with open('out/' + model_type + '.txt', 'w') as f:
        for l in res:
            f.write(label.vocab.itos[l + 1])
            f.write('\n')
    # with open('out/std.txt', 'w') as f:
    #     for l in std:
    #         f.write(label.vocab.itos[l + 1])
    #         f.write('\n')

    test_acc = acc_sum / n
    macro_f1 = f1_score(std, res, average='macro')
    micro_f1 = f1_score(std, res, average='micro')
    weighted_f1 = f1_score(std, res, average='weighted')
    print(
        'type: %s, total %d, acc sum %d, test acc %.4f, macro f1 %.4f, micro f1 %.4f, weighted f1 %.4f'
        % (model_type, n, acc_sum, test_acc, macro_f1, micro_f1, weighted_f1)
    )

if __name__ == '__main__':
    print('Begin')
    model_type, net_name = '', ''
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        print('Invalid model!')
        exit()
    if len(sys.argv) > 2:
        net_name = sys.argv[2]
    else:
        print('Invalid model!')
        exit()
    model_list = ['cnn', 'lstm', 'lstm_attention', 'gru', 'gru_attention', 'mlp']
    if model_type in model_list:
        init(model_type)
    else:
        print('Invalid model!')
        exit()
    set_field()
    set_dataset()
    build_vocab()
    build_iter()
    print('Build Net...')
    net = load_model(net_name)
    print(net)
    print('Testing...')
    test_predict(test_iter, net)
    print('Done.')
