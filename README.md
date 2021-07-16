# 人智导·情感分析实验

<div style="text-align:right">
    <font style="font-size:20px;">
        计94 周䶮 2019011301
        <br>
        2021.6.6
    </font>
</div>

## 一、文件目录

文件目录如下：

<img src="https://img.wzf2000.top/image/2021/06/06/image-20210606145000368.png" alt="image-20210606145000368" style="zoom:50%;" />

**/isear_v2/**

存有数据集、停用词以及经清洗过的数据集 ( train.csv, valid.csv 与 test.csv ) 。

**/model/**

训练完的不同模型，包含以下几个：

<img src="https://img.wzf2000.top/image/2021/06/06/image-20210606143933291.png" alt="image-20210606143933291" style="zoom:50%;" />

**/out/**

不同模型的输出结果 (label) ，其中 std.txt 为标准输出。

**/vec/**

包含了预训练完的 word2vec ，以及构建词向量所用的词库。

**config.ini**

配置文件，包含了网络所需要的参数，分为以下 7 块：

[ALL]，[CNN]，[LSTM]，[GRU]，[LSTM_ATT]，[GRU_ATT]，[MLP] 

分别对应全部模型共同的参数以及每个模型各自的参数。

**wash.py**

对原始数据进行清洗的 python 源文件。

**pre_w2v.py**

预训练词向量前先构建词库的 python 源文件。

**w2v.py**

预训练 word2vec 词向量的 python 源文件。

**EarlyStopping.py**

用于训练提前终止的模块，参考了一个 github 项目，具体在报告中已经说明。

**model.py**

模型训练所用的 python 源文件。

**test.py** 

模型测试所用的 python 源文件。

## 二、运行方法

### 1. 数据清洗以及预训练

#### (1) 数据清洗

命令行运行：

```bash
python wash.py
```

则可以看到在 /isear_v2/ 目录下出现了 train.csv, valid.csv 与 test.csv 三个经过清洗的 csv 文件。

#### (2) 构建词库

命令行运行：

```bash
python pre_w2v.py
```

则可以看到 /vec/ 目录下出现 sentence.txt ，包含了语料库的所有词语。

#### (3) 预训练 word2vec

命令行运行：

```bash
python w2v.py
```

则可以看到 /vec/ 目录下出现了 my_w2v_128.w2v ，里面存有预训练的 128 维词向量。

### 2. 模型训练

在 config.ini 中，可以设置不同模型的参数。

参数设定完毕后，在命令行运行：

```bash
python model.py MODEL
```

这里的 MODEL 必须是下列模型名中的一种：

cnn，lstm，gru，lstm_attention，gru_attention，mlp

示例：

```bash
python model.py cnn
```

该 python 源文件会根据命令行参数自动训练对应种类的模型。

训练完毕后，模型的输出结果位于 /model/ 目录下。

### 3. 模型测试

命令行运行：

```bash
python test.py MODEL FILE
```

这里的 MODEL 和训练中一样，必须是上面的六个中的一个。而第二个参数 FILE 则表示模型文件名字，是在 /model/ 目录下的文件的名字。

示例：

```bash
python test.py lstm_attention lstm_attention_128.pkl
```

该 python 文件将在命令行中显示 accuracy 以及 f1-score 。预测得到的标签，将输出到 /out/ 目录下对应模型名字的 txt 文件中。
