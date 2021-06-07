import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

# 选择前 30000 最大的值
MAX_VAB_SIZE = 30000
# 中心次
C = 3
# 批处理
BATCH_SIZE = 128

with open('text8/text8.train.txt', 'r') as f:
    text = f.read()

print(text[0:100])
text = text.split()
print(text[0:10])

vocab = dict(Counter(text).most_common(MAX_VAB_SIZE - 1))
vocab['unk'] = len(text) - np.sum(list(vocab.values()))

# word
idx_to_word = [word for word in vocab.keys()]
# word:count
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

word_counts = [count for count in vocab.values()]
word_freqs = word_counts / np.sum(word_counts)
# 按照论文
word_freqs = word_freqs * 3.0 / 4.0

print(idx_to_word[0:10])
print(word_freqs[0:10])
print(word_to_idx)
# print(vocab)
