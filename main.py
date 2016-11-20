import numpy as np
from loop import make_generator
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, Flatten

EMBEDDING_DIMS = 100
CONTEXT_LENGTH = 700
QUESTION_LENGTH = 40

cenc = Sequential()
cenc.add(LSTM(128, input_shape=(CONTEXT_LENGTH, EMBEDDING_DIMS), return_sequences=True))
cenc.add(Dropout(0.2))

qenc = Sequential()
qenc.add(LSTM(128, input_shape=(QUESTION_LENGTH, EMBEDDING_DIMS), return_sequences=True))
qenc.add(Dropout(0.2))

aenc = Sequential()
aenc.add(LSTM(128, input_shape=(1, EMBEDDING_DIMS), return_sequences=True))                   
aenc.add(Dropout(0.2))

facts = Sequential()
facts.add(Merge([cenc, qenc], mode="dot", dot_axes=[2, 2]))

attn = Sequential()
attn.add(Merge([aenc, qenc], mode="dot", dot_axes=[2, 2]))

model = Sequential()
model.add(Merge([facts, attn], mode="concat", concat_axis=1))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

dev_gen = make_generator(mode="dev")
train_gen = make_generator(mode="train")
for cycle in range(10):
	model.save_weights("model/weights." + str(cycle) + ".h5")
	model.fit_generator(train_gen, 87599, 3, validation_data=dev_gen, nb_val_samples=34726)
