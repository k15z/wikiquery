import random
import numpy as np
from data import embedding, dev_data, train_data

EMBEDDING_DIMS = len(embedding.values()[0])
def vector(words, pad_to=False):
	vector = []
	for word in words:
		if word in embedding:
			vector.append(embedding[word])
	if pad_to:
		assert len(vector) <= pad_to
		while len(vector) < pad_to:
			vector.append([0] * EMBEDDING_DIMS)
	return np.array(vector)

def generate(data):
	while True:
		for c, q, a in data:
			if len(a) < 1:
				continue
			if random.random() < 0.5:
				a = random.choice(a)
				a = vector([a], pad_to=1)
				y = [1.0, 0.0]
			else:
				a = random.choice(list(set(c).difference(a)))
				a = vector([a], pad_to=1)
				y = [0.0, 1.0]
			c = vector(c, pad_to=700)
			q = vector(q, pad_to=40)
			yield (c, q, a, y)

def make_generator(mode="train", batch_size=32):
	assert mode == "train" or mode == "dev"
	generator = generate(dev_data) if mode == "dev" else generate(train_data)

	while True:
		C = []
		Q = []
		A = []
		Y = []
		for i in range(batch_size):
			c, q, a, y = next(generator)
			C.append(c)
			Q.append(q)
			A.append(a)
			Y.append(y)
		C, Q, A, Y = map(np.array, [C, Q, A, Y])
		yield ([C, Q, A], Y)

if __name__ == "__main__":
	X, Y = next(make_generator())
	C, Q, A = X
	print(C.shape)
	print(Q.shape)
	print(A.shape)
	print(Y.shape)
