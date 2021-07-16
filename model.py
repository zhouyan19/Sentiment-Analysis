import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
from torch.autograd import Variable
import time
from tqdm import tqdm
import configparser as cp
import json
from EarlyStopping import EarlyStopping as es

if_cuda = False
device = ''
model_name = ''
data_path = ''
vec_path = ''
model_path = ''
batch_size = 32
embedding_dim = 128
lr = 0.001
num_epochs = 8
kernel_sizes = [3, 4, 5]
num_channels = [100, 100, 100]
hidden_size = 128
num_hidden = 128
stop_words = []
sentence, label = None, None
train, valid = None, None
vectors = None
vocab_size = 0
label_num = 0
train_iter, val_iter = None, None
patience = 5


def init(type):
    print('Init...')
    global if_cuda, device
    global model_name, data_path, vec_path, model_path
    global batch_size, embedding_dim, lr, num_epochs
    global kernel_sizes, num_channels
    global hidden_size, stop_words
    global num_hidden
    global patience
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
    patience = json.loads(all_items['patience'])
    with open(data_path + 'stopwords.txt', 'r') as f:
        stop_words.append(f.readline().strip('\n'))

def tokenizer(text):
    # words = text.split()
    # return [word for word in words if word not in stop_words]
    return text.split()

def set_field():
    print('Set Field...')
    global sentence, label
    sentence = data.Field(sequential=True,
                        lower=True,
                        tokenize=tokenizer)
    label = data.Field(sequential=False)

def set_dataset():
    print('Set Dataset...')
    global data_path, train, valid, label, sentence
    train, valid = data.TabularDataset.splits(
        path=data_path,
        skip_header=True,
        train='train.csv',
        validation='valid.csv',
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
    global batch_size, train_iter, val_iter, train, valid
    train_iter, val_iter = data.Iterator.splits(
        (train, valid),
        batch_sizes=(batch_size, len(valid)),
        sort_key=lambda x : len(x.sentence)
    )


# 普通池化来实现全局池化
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])
        # shape: (batch_size, channel, 1)


# TextCNN Model:
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding之后的shape: torch.Size([, , ])
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), label_num)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embedding_dim, 
                                        out_channels = c, 
                                        kernel_size = k))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


# LSTM Model:
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=1, dropout=0, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_size, self.label_num)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(1, 0, 2)
        this_size = embeds.shape[1]
        h0 = Variable(torch.zeros(2, this_size, self.hidden_size))
        c0 = Variable(torch.zeros(2, this_size, self.hidden_size))
        lstm_out, (h, c) = self.lstm(embeds, (h0, c0))
        out = torch.cat((h[-2, : ,:], h[-1, : ,:]), dim=1)
        out = self.linear(out)
        predict = F.softmax(out, dim=1)
        return predict


# LSTM Model (Attention):
class LSTM_ATT(nn.Module):
    def __init__(self):
        super(LSTM_ATT, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=1, dropout=0, bidirectional=True)
        self.init_w = Variable(torch.Tensor(1, 2*self.hidden_size), requires_grad=True)
        self.init_w = nn.Parameter(self.init_w)
        self.linear = nn.Linear(2*self.hidden_size, self.label_num)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(1, 0, 2)
        embeds = embeds[:, :, :512]
        this_size = embeds.shape[1]
        h0 = Variable(torch.zeros(2, this_size, self.hidden_size))
        c0 = Variable(torch.zeros(2, this_size, self.hidden_size))
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        lstm_out = torch.tanh(lstm_out) # [len, bach_size, hidden_size]
        M = torch.matmul(self.init_w, lstm_out.permute(1, 2, 0))
        alpha = F.softmax(M, dim=2) # [batch_size, 1, len]
        outputs = torch.matmul(alpha, lstm_out.permute(1, 0, 2)).squeeze() # outputs:[batch_size, 2*hidden_size]
        predict = self.linear(outputs) # predict:[batch_size, hidden_size]
        return predict


# GRU Model:
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=1, dropout=0, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_size, self.label_num)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(1, 0, 2)
        this_size = embeds.shape[1]
        h0 = Variable(torch.zeros(2, this_size, self.hidden_size))
        lstm_out, h = self.gru(embeds, h0)
        out = torch.cat((h[-2, : ,:], h[-1, : ,:]), dim=1)
        out = self.linear(out)
        predict = F.softmax(out, dim=1)
        return predict


# GRU Model:
class GRU_ATT(nn.Module):
    def __init__(self):
        super(GRU_ATT, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=1, dropout=0, bidirectional=True)
        self.init_w = Variable(torch.Tensor(1, 2*self.hidden_size), requires_grad=True)
        self.init_w = nn.Parameter(self.init_w)
        self.linear = nn.Linear(2*self.hidden_size, self.label_num)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(1, 0, 2)
        this_size = embeds.shape[1]
        h0 = Variable(torch.zeros(2, this_size, self.hidden_size))
        gru_out, h = self.gru(embeds, h0)
        gru_out = torch.tanh(gru_out) # [len, bach_size, hidden_size]
        M = torch.matmul(self.init_w, gru_out.permute(1, 2, 0))
        alpha = F.softmax(M, dim=2) # [batch_size, 1, len]
        out = torch.matmul(alpha, gru_out.permute(1, 0, 2)).squeeze() # outputs:[batch_size, 2*hidden_size]
        out = self.linear(out)
        predict = F.softmax(out, dim=1)
        return predict


# MLP Model:
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding之后的shape: torch.Size([, , ])
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.num_hidden = num_hidden
        self.label_num = label_num
        self.linear1 = nn.Linear(embedding_dim, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.linear3 = nn.Linear(self.num_hidden, self.label_num)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.mean(dim=1)
        x = F.softmax(self.linear1(embeds), dim=1)
        x = F.softmax(self.linear2(x), dim=1)
        out = F.softmax(self.linear3(x), dim=1)
        return out


# 训练
def trainer(train_iter, valid_iter, net, loss, optimizer, num_epochs, early_stopping):
    for epoch in tqdm(range(num_epochs)):
        batch_count = 0
        train_l_sum, train_acc_sum, n, start = 0, 0, 0, time.time()
        for batch_idx, batch in tqdm(enumerate(train_iter)):
            X, y = batch.sentence, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.detach().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc, valid_loss = evaluate_accuracy(valid_iter, net)
        print(
            "epoch %d, train loss %.4f, valid loss %.4f, train acc %.3f, valid acc %.3f, time %.1f sec"
            % (epoch + 1, train_l_sum / batch_count, valid_loss, train_acc_sum / n,
                test_acc, time.time() - start)
        )
        early_stopping(valid_loss, net)
        if early_stopping.early_stop:
            print("Early stopping, epoch ", epoch + 1)
            break

# 计算准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.sentence, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1) #X转置 y减1
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                y_hat = net(X)
                l = loss(y_hat, y)
                acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                net.train()
            n += y.shape[0]
    return acc_sum / n, l

if __name__ == '__main__':
    print('Begin')
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        print('Invalid model!')
        exit()
    model_list = ['cnn', 'lstm', 'lstm_attention', 'gru', 'gru_attention', 'mlp']
    if model_name in model_list:
        init(model_name)
    else:
        print('Invalid model!')
        exit()
    set_field()
    set_dataset()
    build_vocab()
    build_iter()
    print('Build Net...')
    net = None
    if model_name == 'cnn':
        net = TextCNN()
    elif model_name == 'lstm':
        net = LSTM()
    elif model_name == 'lstm_attention':
        net = LSTM_ATT()
    elif model_name == 'gru':
        net = GRU()
    elif model_name == 'gru_attention':
        net = GRU_ATT()
    elif model_name == 'mlp':
        net = MLP()
    else:
        pass
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    early_stopping = es(patience)
    print('Training...')
    trainer(train_iter, val_iter, net, loss, optimizer, num_epochs, early_stopping)
    if model_name == 'cnn':
        torch.save(net, model_path + 'cnn_' + str(num_channels[0]) + '_' + str(lr) + '.pkl')
    elif model_name == 'lstm':
        torch.save(net, model_path + 'lstm.pkl')
    elif model_name == 'lstm_attention':
        torch.save(net, model_path + 'lstm_attention_' + str(hidden_size) + '.pkl')
    elif model_name == 'gru':
        torch.save(net, model_path + 'gru.pkl')
    elif model_name == 'gru_attention':
        torch.save(net, model_path + 'gru_attention.pkl')
    elif model_name == 'mlp':
        torch.save(net, model_path + 'mlp.pkl')
    else:
        pass
    print('Model already saved.')
    print('Done.')  