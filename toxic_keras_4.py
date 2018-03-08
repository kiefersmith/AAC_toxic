import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers

EMBEDDING_FILE='../input/glove-200/glove.6B.300d.txt'
TRAIN_DATA_FILE='../input/jigsaw-toxic-comment-classification-challenge/train.csv'
TEST_DATA_FILE='../input/jigsaw-toxic-comment-classification-challenge/test.csv'

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

embed_size=300
max_features=20000
maxlen=100

label_cols = train.columns[2:8]

comment = 'comment_text'
sent_train = train[comment].fillna('empty_comment').values
sent_test = test[comment].fillna('empty_comment').values

y=train[label_cols].values


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(sent_train))
tokenized_train = tokenizer.texts_to_sequences(sent_train)
tokenized_test = tokenizer.texts_to_sequences(sent_test)
X_train = pad_sequences(tokenized_train, maxlen=maxlen)
X_test = pad_sequences(tokenized_test, maxlen=maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split())for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(275, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(175, 2)(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(6, activation='sigmoid')(x)
model = Model(inputs=inp,
              outputs=x)

stop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

X_tra, X_val, y_tra, y_val = train_test_split(X_train, y, train_size=0.9)

model.fit(X_tra, y, batch_size=64, epochs=1, validation_data=(X_val, y_val), callbacks=[stop])
model.save('toxic_keras.h5')

y_test = model.predict([X_test], batch_size=1024, verbose=1)
sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
sample_submission[label_cols] = y_test
sample_submission.to_csv('submission_keras.csv', index=False)
