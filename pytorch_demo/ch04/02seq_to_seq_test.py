from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")

MAX_LENGTH = 10  # Maximum sentence length

# 默认的词向量
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # 统计SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # 统计默认的令牌
        for word in keep_words:
            self.addWord(word)

    # 小写并删除非字母字符
    def normalizeString(s):
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    # 使用字符串句子，返回单词索引的句子
    def indexesFromSentence(voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# 编码器
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 初始化GRU;input_size和hidden_size参数都设置为'hidden_size'
        # 因为我们输入的大小是一个有多个特征的词向量== hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 将单词索引转换为向量
        embedded = self.embedding(input_seq)
        # 为RNN模块填充批次序列
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # 正向通过GRU
        outputs, hidden = self.gru(packed, hidden)
        # 打开填充
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 将双向GRU的输出结果总和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # 返回输出以及最终的隐藏状态
        return outputs, hidden


# Luong的注意力层
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力权重（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # 转置max_length和batch_size维度
        attn_energies = attn_energies.t()

        # 返回softmax归一化概率分数（增加维度）
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# 解码器
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 保持参考
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 定义层
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 注意：我们这步只运行一次
        # 获取当前输入字对应的向量映射
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 通过单向GRU转发
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 通过当前GRU的输出计算注意力权重
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # 注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 使用Luong的公式5来连接加权上下文向量和GRU输出
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # 使用Luong的公式6来预测下一个单词
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # 返回输出和最终的隐藏状态
        return output, hidden


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # 格式化输入句子作为批处理
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 创建长度张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置批量的维度以匹配模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 使用适当的设备
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # 用earcher解码句子s
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# 评估来自用户输入的输入(stdin)
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while (1):
        try:
            # 获取输入的句子
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 规范化句子
            input_sentence = normalizeString(input_sentence)
            # 评估句子
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 格式化和打印回复句
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# 规范化输入句子并调用evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print("> " + sentence)
    # 规范化句子
    input_sentence = normalizeString(sentence)
    # 评估句子
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))
