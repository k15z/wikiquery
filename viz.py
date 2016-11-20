from loop import vector
from keras.models import model_from_json
from data import embedding, dev_data, train_data

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


#MODEL_JSON = open("model/arch.json","rt").read()
#model = model_from_json(MODEL_JSON)
model.load_weights("model/iw-lstm.h5")

for i in range(20):
	c, q, a = dev_data[i]
	results = []
	c_vec = vector(c, pad_to=700)
	q_vec = vector(q, pad_to=40)
	for word in c:
		a_vec = vector([word], pad_to=1)
		result = model.predict([np.array([c_vec]), np.array([q_vec]), np.array([a_vec])])
		results.append((result.flatten()[0], word))
	results = list(reversed(sorted(results)))[:10]
	print(" ".join(c))
	print(" ".join(q))
	print(" ".join(a))
	print(results)
	print("")
