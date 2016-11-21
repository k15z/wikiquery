import os
import re
import json
import random
import cPickle as pickle

DATA_DIR = os.path.dirname(__file__)

EMBEDDING_DIMS = 300
EMBEDDING_FILE = os.path.join(DATA_DIR, 'glove/glove.6B.300d.txt')
EMBEDDING_PICKLE = os.path.join(DATA_DIR, 'embedding.pickle')
if os.path.isfile(EMBEDDING_PICKLE):
	print("Restoring embedding...")
	embedding = pickle.load(open(EMBEDDING_PICKLE, 'rb'))
	for value in embedding.values():
		assert len(value) == EMBEDDING_DIMS
else:
	print("Building embedding...")
	embedding = {}
	for line in open(EMBEDDING_FILE, 'rt'):
		line = line.split()
		embedding[line[0]] = map(float, line[1:])
	for value in embedding.values():
		assert len(value) == EMBEDDING_DIMS
	print("Saving embedding...")
	pickle.dump(embedding, open(EMBEDDING_PICKLE, 'wb'))

DEV_FILE = os.path.join(DATA_DIR, 'squad/dev-v1.1.json')
TRAIN_FILE = os.path.join(DATA_DIR, 'squad/train-v1.1.json')
def load_dataset(target):
	assert target == DEV_FILE or target == TRAIN_FILE
	data = json.load(open(os.path.join(DATA_DIR, target), "rt"))["data"]
	pqas = []
	for article in data:
		for paragraph in article['paragraphs']:
			for question in paragraph["qas"]:
				for answer in question["answers"]:
					p = re.sub('[^0-9a-zA-Z]+', ' ', paragraph["context"].lower()).split()
					q = re.sub('[^0-9a-zA-Z]+', ' ', question["question"].lower()).split()
					a = re.sub('[^0-9a-zA-Z]+', ' ', answer["text"].lower()).split()
					pqas.append((p, q, a))
	random.shuffle(pqas)
	return pqas

DEV_PICKLE = os.path.join(DATA_DIR, 'data_dev.pickle')
if os.path.isfile(DEV_PICKLE):
	print("Restoring dev...")
	dev_data = pickle.load(open(DEV_PICKLE, 'rb'))
else:
	print("Building dev...")
	dev_data = load_dataset(DEV_FILE)
	pickle.dump(dev_data, open(DEV_PICKLE, 'wb'))

TRAIN_PICKLE = os.path.join(DATA_DIR, 'data_train.pickle')
if os.path.isfile(TRAIN_PICKLE):
	print("Restoring train...")
	train_data = pickle.load(open(TRAIN_PICKLE, 'rb'))
else:
	print("Building train...")
	train_data = load_dataset(TRAIN_FILE)
	pickle.dump(train_data, open(TRAIN_PICKLE, 'wb'))

assert embedding and dev_data and train_data, "Error loading data."
