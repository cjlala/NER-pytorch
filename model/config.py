#Author: CJ
#Date: 2019-02-24 20:39:14
import json

EMBEDDING_DIM = 200
HIDDEN_DIM = 100
DATA_PATH = 'tagged_data.txt'
VOCAB_SIZE = 1753
TAGET_SIZE = 19
lr = 0.01
TOTAL_WORD_COUNT = 261035

with open('../tagged_data.txt', 'r') as f:
    data = f.read().decode('utf8').split('\n\r')
    
with open('../tag2idx.json', 'r') as j:
    tag2idx = json.load(j)

with open('../char2idx.json', 'r') as j:
    char2idx = json.load(j)