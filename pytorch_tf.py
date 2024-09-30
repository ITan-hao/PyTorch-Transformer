import pickle
import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch import optim
from torchnet import meter
from tqdm import tqdm

# 模型输入参数，需要自己根据需要调整
hidden_dim = 100  # 隐层大小
epochs = 20  # 迭代次数
batch_size = 32  # 每个批次样本大小
embedding_dim = 20  # 每个字形成的嵌入向量大小
output_dim = 2  # 输出维度，因为是二分类
lr = 0.003  # 学习率
device = 'cpu'
input_shape = 180  # 每句话的词的个数，如果不够需要使用0进行填充


# 加载文本数据
def load_data(file_path, input_shape=20):
    df = pd.read_csv(file_path, sep='\t')
    # 标签及词汇表
    labels, vocabulary = list(df['label'].unique()), list(df['text'].unique())
    # 构造字符级别的特征
    string = ''
    for word in vocabulary:
        string += word

    # 所有的词汇表
    vocabulary = set(string)
    # word2idx 将字映射为索引
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)
    # idx2word 将索引映射为字
    inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}
    # label2idx 将正反面映射为0和1
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open('label_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)
    # idx2label 将0和1映射为正反面
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    # 训练数据中所有词的个数
    vocab_size = len(word_dictionary.keys())  # 词汇表大小
    # 标签类别，分别为正、反面
    label_size = len(label_dictionary.keys())  # 标签类别数量
    # 序列填充，按input_shape填充，长度不足的按0补充
    # 将一句话映射成对应的索引 [0,24,63...]
    x = [[word_dictionary[word] for word in sent] for sent in df['text']]
    # 如果长度不够input_shape，使用0进行填充
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    # 形成标签0和1
    y = [[label_dictionary[sent]] for sent in df['label']]
    #     y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
    y = np.array(y)
    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class, feedforward_dim=256, num_head=2, num_layers=3, dropout=0.1,
                 max_len=128):
        super(Transformer, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置编码层
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)
        # 编码层
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, feedforward_dim, dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        # 输出层
        self.fc = nn.Linear(embedding_dim, num_class)

    def forward(self, x):
        # 输入的数据维度为【批次，序列长度】，需要交换因为transformer的输入维度为【序列长度，批次，嵌入向量维度】
        x = x.transpose(0, 1)
        # 将输入的数据进行词嵌入，得到数据的维度为【序列长度，批次，嵌入向量维度】
        x = self.embedding(x)
        # 维度为【序列长度，批次，嵌入向量维度】
        x = self.positional_encoding(x)
        # 维度为【序列长度，批次，嵌入向量维度】
        x = self.transformer(x)
        # 将每个词的输出向量取均值，也可以随意取一个标记输出结果，维度为【批次，嵌入向量维度】
        x = x.mean(axis=0)
        # 进行分类，维度为【批次，分类数】
        x = self.fc(x)
        return x


# 1.获取训练数据
x_train, y_train, output_dictionary_train, vocab_size_train, label_size, inverse_word_dictionary_train = load_data(
    "./train.tsv", input_shape)
x_test, y_test, output_dictionary_test, vocab_size_test, label_size, inverse_word_dictionary_test = load_data(
    "./test.tsv", input_shape)
idx = 0
word_dictionary = {}
for k, v in inverse_word_dictionary_train.items():
    word_dictionary[idx] = v
    idx += 1
for k, v in inverse_word_dictionary_test.items():
    word_dictionary[idx] = v
    idx += 1
# 3.将numpy转成tensor
x_train = torch.from_numpy(x_train).to(torch.int32)
y_train = torch.from_numpy(y_train).to(torch.float32)
x_test = torch.from_numpy(x_test).to(torch.int32)
y_test = torch.from_numpy(y_test).to(torch.float32)
# 4.形成训练数据集
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
# 5.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size,
                                           True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size,
                                          False)
# 6.模型训练
model = Transformer(len(word_dictionary), embedding_dim, output_dim)
Configimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
criterion = nn.CrossEntropyLoss()  # 多分类损失函数
model.to(device)
loss_meter = meter.AverageValueMeter()
best_acc = 0  # 保存最好准确率
best_model = None  # 保存对应最好准确率的模型参数
for epoch in range(epochs):
    model.train()  # 开启训练模式
    epoch_acc = 0  # 每个epoch的准确率
    epoch_acc_count = 0  # 每个epoch训练的样本数
    train_count = 0  # 用于计算总的样本数，方便求准确率
    loss_meter.reset()

    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        x_input = x_train.long().contiguous()
        x_input = x_input.to(device)
        Configimizer.zero_grad()

        # 形成预测结果
        output_ = model(x_input)

        # 计算损失
        loss = criterion(output_, y_train.long().view(-1))
        loss.backward()
        Configimizer.step()

        loss_meter.add(loss.item())

        # 计算每个epoch正确的个数
        epoch_acc_count += (output_.argmax(axis=1) == y_train.view(-1)).sum()
        train_count += len(x_train)

    # 每个epoch对应的准确率
    epoch_acc = epoch_acc_count / train_count

    # 打印信息
    print("【EPOCH: 】%s" % str(epoch + 1))
    print("训练损失为%s" % (str(loss_meter.mean)))
    print("训练精度为%s" % (str(epoch_acc.item() * 100)[:5]) + '%')
    # 保存模型及相关信息
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = model.state_dict()

    # 在训练结束保存最优的模型参数
    if epoch == epochs - 1:
        # 保存模型
        torch.save(best_model, './best_model.pkl')
word2idx = {}
for k, v in word_dictionary.items():
    word2idx[v] = k
label_dict = {0: "非谣言", 1: "谣言"}
try:
    input_shape = 180  # 序列长度，就是时间步大小，也就是这里的每句话中的词的个数
    #     sent = "电视刚安装好，说实话，画质不怎么样，很差！"
    # 用于测试的话
    sent = "凌晨的长春，丢失的孩子找到了，被偷走的车也找到，只是偷车贼没找到，看来，向雷锋同志学习50周年的今天，还是一个有效果的日子啊。"
    # 将对应的字转化为相应的序号
    x = [[word2idx[word] for word in sent]]
    # 如果长度不够180，使用0进行填充
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    x = torch.from_numpy(x)
    # 加载模型
    model_path = './best_model.pkl'
    model = Transformer(len(word_dictionary), embedding_dim, output_dim)
    model.load_state_dict(torch.load(model_path, 'cpu'))
    # 模型预测，注意输入的数据第一个input_shape,就是180
    y_pred = model(x.long())
    print('输入语句: %s' % sent)
    print('谣言检测结果: %s' % label_dict[y_pred.argmax().item()])
except KeyError as err:
    print("您输入的句子有汉字不在词汇表中，请重新输入！")
    print("不在词汇表中的单词为：%s." % err)