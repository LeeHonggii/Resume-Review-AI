import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from collections import Counter
from konlpy.tag import Okt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
datapath = "./App/initdata/"

class Classfication_model:
  def __init__(self,
               model_file=datapath+'simple_lstm.h5',
               data_file=datapath+'data_v3.csv',
               test_file=datapath+'data_test.csv',
               ):   # REPLACABLE

    np.set_printoptions(suppress=True, precision=2)

    self.vocab = []
    self.word_to_index = {}
    self.index_to_word = {}
    self.maxlen = 0
    self.num_classes = 0
    self.vlen = 0
    self.class_name = {1: '성공', 2: '포부', 0: '부정'}

    self.stop_words = ['의', '가', '이', '은', '들',
                  '는', '좀', '잘', '걍', '과', '를',
                  '으로', '자', '에', '와', '한', '하다'
                  # , '…', '·'
                  ]

    self.tokenizer = Okt()
    self.enc = preprocessing.LabelBinarizer()

    if os.path.exists(model_file):
      traindf = self.prepare_data(data_file)
      self.prepare_training(traindf)
      self.model = load_model(model_file)
      print("Simple LSTM classification model loaded.")
    else:
      print(f"LSTM classification model not found. Training starts.: {model_file}")

      traindf = self.prepare_data(data_file)
      testdf = self.prepare_data(test_file)

      x_train, y_train = self.prepare_training(traindf)
      train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.2)
      # print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

      self.model = keras.Sequential()
      # self.model.add(keras.layers.Embedding(self.vlen, 100, input_length=self.maxlen))
      self.model.add(keras.layers.Embedding(self.vlen, 100, input_shape=(self.maxlen, )))
      self.model.add(keras.layers.LSTM(128))
      self.model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
      self.model.summary()

      self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      self.model.fit(train_x, train_y, epochs=3, batch_size=34, validation_data=(val_x, val_y))

      self.model.save(model_file)
      print((f"Simple LSTM classification model '{model_file}' saved."))

      pred = self.model.predict(val_x)
      # print(pred[:5])
      print("Validation Accuracy: ", np.mean(np.argmax(val_y, axis=1) == np.argmax(pred, axis=1)))

      test_x, test_y = self.prepare_test(testdf)
      pred = self.model.predict(test_x)
      # print(len(pred))
      # pred[:5]
      print("Test Accuracy: ", np.mean(np.argmax(test_y, axis=1) == np.argmax(pred, axis=1)))
      print('\n')

  def prepare_data(self, data_file, verbose=False):
    print("processing", data_file) if verbose else None
    df = pd.read_csv(data_file)
    print(df.head()) if verbose else None
    print(df.describe()) if verbose else None
    print(df['Label'].unique()) if verbose else None
    sns.set_theme(style='darkgrid') if verbose else None
    ax = sns.countplot(x="Label", data=df) if verbose else None
    print(len(df)) if verbose else None
    df = df[df['Label'] != '강점']
    print(len(df)) if verbose else None
    df.drop_duplicates(subset=['Sentence'], inplace=True)
    print(len(df)) if verbose else None
    df = df.dropna(how='any')
    print(len(df)) if verbose else None
    return df

  def tokenize_list(self, sentence_list):   # REPLACABLE
    tokenized_list = []
    # co = 0
    for sentence in sentence_list:
      temp_x = self.tokenizer.morphs(sentence)
      temp_x = [word for word in temp_x if word not in self.stop_words]
      tokenized_list.append(temp_x)
      # co =+ 1
      # if co % 1000 == 0:
      #   print(co, end=' ')

    return tokenized_list

  def make_vocab(self, data, verbose=False):   # REPLACABLE
    words = np.concatenate(data).tolist()
    print(len(words)) if verbose else None
    print(words[:10]) if verbose else None
    counter = Counter(words)
    print(len(counter)) if verbose else None
    counter = counter.most_common(30000 - 4)  # reserve for 4 special tokens
    print(len(counter)) if verbose else None
    print(counter[:10]) if verbose else None
    vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter]
    print(vocab[:10]) if verbose else None

    return vocab

  def wordlist_to_indexlist(self, wordlist):   # REPLACABLE
    return [self.word_to_index[word] if word in self.word_to_index else self.word_to_index['<UNK>'] for word in wordlist]

  def unk_count(self, indexlist):   # REPLACABLE
    return indexlist.count(self.word_to_index['<UNK>'])

  def prepare_training(self, df, verbose=False):
    tokenized_x = self.tokenize_list(df['Sentence'])
    print(len(tokenized_x)) if verbose else None
    print(len(tokenized_x)) if verbose else None
    print(tokenized_x[:2]) if verbose else None
    # print(traindf['Sentence'][:1]) if verbose else None

    self.vocab = self.make_vocab(tokenized_x, True)
    self.word_to_index = {word: index for index, word in enumerate(self.vocab)}
    self.index_to_word = {index: word for index, word in enumerate(self.vocab)}

    # def wordlist_to_indexlist(wordlist):
    #   return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in wordlist]

    x_data = list(map(self.wordlist_to_indexlist, tokenized_x))
    print(x_data[:3]) if verbose else None

    num = [len(word) for word in x_data]
    print(num[:5]) if verbose else None
    print(f'AVE length: {np.mean(num)}') if verbose else None
    print(f'MAX length: {np.max(num)}') if verbose else None
    print(f'STD length: {np.std(num)}') if verbose else None
    self.maxlen = np.max(num) + 1
    self.num_classes = len(df['Label'].unique())
    self.vlen = len(self.vocab)
    print(self.maxlen, self.num_classes, self.vlen) if verbose else None

    x_train = pad_sequences(x_data, maxlen=self.maxlen, value=self.word_to_index['<PAD>'], padding='pre')
    print(x_train[:2]) if verbose else None

    y_train = np.array(self.enc.fit_transform(df['Label']))
    print(y_train[:3]) if verbose else None

    return x_train, y_train

  def prepare_x(self, sentence_list):
    tx = self.tokenize_list(sentence_list)
    tx = list(map(self.wordlist_to_indexlist, tx))
    tx = pad_sequences(tx, maxlen=self.maxlen, value=self.word_to_index['<PAD>'], padding='pre')
    return tx

  def prepare_test(self, df, verbose=False):
    test_x = self.prepare_x(df['Sentence'])
    print(test_x[:2]) if verbose else None
    test_y = np.array(self.enc.fit_transform(df['Label']))
    print(test_y[:2]) if verbose else None

    return test_x, test_y


  # np.set_printoptions(suppress=True, precision=2)
  # for i in range(len(pred)):
  #   print(i, pred[i], classification[np.argmax(pred[i])], unk_count(list(test_x[i])), testdf['Sentence'][i])

  def prepare_sample(self, filename):
    inList = []
    with open(filename, "r", encoding="utf-8") as inFp:
        text = inFp.read()

    return text

  def prepare_target(self, text, verbose=False):
    inList = text.split('\n')
    outList = []
    for inStr in inList:
        if inStr == '\n':
            continue
        # print(inStr)
        inStr = inStr.replace("<p>", "").replace("</p>", "")
        if len(inStr) == 0:
            continue
        # if inStr[-1] != '.':
        #   inStr = inStr + '.'
        tList = inStr.split('.')
        tList = tList[:-1]
        # if len(tList) != 1:
        #   tList = tList[:-1]
        for tStr in tList:
            t = tStr.strip() + '.'
            outList.append(t)

    if verbose:
      for s in outList:
          print(s)

    target_x = self.prepare_x(outList)

    return target_x, outList

  def sample_classification_test(self, file_name=datapath+'sample.txt', verbose=False):
    text = self.prepare_sample(file_name)
    print(text) if verbose else None

    predict, outList = self.classify(text)
    for i in range(len(predict)):
      print(predict[i], self.class_name[np.argmax(predict[i])], outList[i])

    print(datapath+file_name, "finised\n")
    print(predict) if verbose else None

  def classify(self, inList):
    # print(inList)
    target_x, outList = self.prepare_target(inList)
    # logger.debug(f"target_x shape: {target_x.shape}")
    # logger.debug(f"target_x: {target_x}")

    if target_x.shape[0] == 0:
      return [], outList

    predict = self.model.predict(target_x)
    # logger.debug(f"predict: {predict}")

    return predict, outList