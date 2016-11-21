from loop import vector
from keras.models import model_from_json
from data import embedding, dev_data, train_data
import tqdm
import json
import numpy as np
from loop import make_generator
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, Flatten

EMBEDDING_DIMS = 300
CONTEXT_LENGTH = 700
QUESTION_LENGTH = 40

cenc = Sequential()
cenc.add(LSTM(128, input_shape=(CONTEXT_LENGTH, EMBEDDING_DIMS), return_sequences=True))
cenc.add(Dropout(0.25))

qenc = Sequential()
qenc.add(LSTM(128, input_shape=(QUESTION_LENGTH, EMBEDDING_DIMS), return_sequences=True))
qenc.add(Dropout(0.25))

aenc = Sequential()
aenc.add(LSTM(128, input_shape=(1, EMBEDDING_DIMS), return_sequences=True))                   
aenc.add(Dropout(0.25))

facts = Sequential()
facts.add(Merge([cenc, qenc], mode="dot", dot_axes=[2, 2]))

attn = Sequential()
attn.add(Merge([aenc, qenc], mode="dot", dot_axes=[2, 2]))

model = Sequential()
model.add(Merge([facts, attn], mode="concat", concat_axis=1))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights("model/iw-lstm.h5")

results = []
for i in tqdm.tqdm(range(250)):
	prediction = []
	c, q, a = train_data[i+233]
	c_vec = vector(c, pad_to=700)
	q_vec = vector(q, pad_to=40)
	
 	C = []
	Q = []
	A = []
	for word in c:
		C.append(c_vec)
		Q.append(q_vec)
		A.append(vector([word], pad_to=1))
	C, Q, A = map(np.array, (C, Q, A))
	P = model.predict([C, Q, A])
  
	for i, word in enumerate(c):
		prediction.append({
			"word": word,
			"probability": float(P[i][0])
		})

	results.append({
		"text": " ".join(c),
		"question": " ".join(q),
		"answer": " ".join(a),
		"prediction": prediction
	})
	json.dump(results, open("viz.json", "wt"), indent=4)
