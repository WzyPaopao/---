import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import gensim
import jieba
import pandas as pd

# 读取训练集标签
train_label = np.loadtxt('./train_label_text.txt', delimiter=',', dtype=np.int16)
train_label_onehot = tf.one_hot(list(train_label), depth=10)

# 读取训练集特征
train_data_temp = []
train_data = []
with open('./train_data_text_clear.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_data_temp.append(line.strip().strip('[]').split(','))
for line in train_data_temp:
    new_line = []
    for word in line:
        word = word.strip().strip('\'')
        new_line.append(word)
    train_data.append(new_line)
del train_data_temp

# 读取验证集
valid_label = np.loadtxt('./valid_label_text.txt', delimiter=',', dtype=np.int16)
valid_label_onehot = tf.one_hot(list(valid_label), depth=10)

# 读取验证集特征
valid_data_temp = []
valid_data = []
with open('./valid_data_text.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        valid_data_temp.append(line.strip().strip('[]').split(','))
for line in valid_data_temp:
    new_line = []
    for word in line:
        word = word.strip().strip('\'')
        new_line.append(word)
    valid_data.append(new_line)
del valid_data_temp

# 读取测试集
test_label = np.loadtxt('./test_label_text.txt', delimiter=',')

# 读取测试集特征
test_data_temp = []
test_data = []
with open('./test_data_text.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        test_data_temp.append(line.strip().strip('[]').split(','))
for line in test_data_temp:
    new_line = []
    for word in line:
        word = word.strip().strip('\'')
        new_line.append(word)
    test_data.append(new_line)
del test_data_temp

# 读取词向量模型
word2vec_model = gensim.models.word2vec.Word2Vec.load('./myNewsModel_pure')

# 使用自己的词库
word_list = word2vec_model.wv.index2word

idx2word = {}
for i in range(word_list.__len__()):
    idx2word[i+2] = word_list[i]
idx2word[0] = ' '
idx2word[1] = '?'

word2idx = {word: idx for idx, word in idx2word.items()}

# 词条数目
numOfWord = len(word2idx)

# 词向量维度
wordvector_dim = len(word2vec_model.wv['大海'])

fill_vector = np.zeros([wordvector_dim, ])

# 词向量矩阵
embedding_mat = []
for i in range(numOfWord):
    try:
        embedding_mat.append(word2vec_model[idx2word[i]])
    except:
        embedding_mat.append(fill_vector)

del word2vec_model

# 句子的最大长度，超过截断，不足补零
max_len = 50

# 转换训练集
train_data_idx = []
for sentencen in train_data:
    temp = []
    for word in sentencen:
        if word in word2idx:
            temp_word = word2idx[word]
        else:
            temp_word = word2idx['?']
        temp.append(temp_word)
    train_data_idx.append(temp)

del train_data

# 转换验证集
valid_data_idx = []
for sentencen in valid_data:
    temp = []
    for word in sentencen:
        if word in word2idx:
            temp_word = word2idx[word]
        else:
            temp_word = word2idx['?']
        temp.append(temp_word)
    valid_data_idx.append(temp)

del valid_data

# 转换测试集
test_data_idx = []
for sentencen in test_data:
    temp = []
    for word in sentencen:
        if word in word2idx:
            temp_word = word2idx[word]
        else:
            temp_word = word2idx['?']
        temp.append(temp_word)
    test_data_idx.append(temp)

del test_data

train_data_idx_matrix = keras.preprocessing.sequence.pad_sequences(train_data_idx, maxlen=max_len)
valid_data_idx_matrix = keras.preprocessing.sequence.pad_sequences(valid_data_idx, maxlen=max_len)
test_data_idx_matrix = keras.preprocessing.sequence.pad_sequences(test_data_idx, maxlen=max_len)

np_embedding_mat = np.array(embedding_mat)

# 神经网络模型
model = keras.Sequential()

model.add(layers.Embedding(
    len(np_embedding_mat),
    wordvector_dim,
    weights = [np_embedding_mat],
    input_length = max_len,
    trainable = False
))

model.add(layers.LSTM(128, dropout = 0.2))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

callbacks = [
    keras.callbacks.EarlyStopping(patience=2, min_delta=1e-2)
]

# 编译模型
model.compile(
    loss = tf.losses.sparse_categorical_crossentropy,
    optimizer = 'adam',
    metrics = ['accuracy'],
    callbacks = callbacks
)

history = model.fit(
    train_data_idx_matrix,
    train_label,
    epochs = 10,
    validation_data = [valid_data_idx_matrix, valid_label],
    batch_size = 500
)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

pre = model.evaluate(test_data_idx_matrix, test_label)
print(pre)