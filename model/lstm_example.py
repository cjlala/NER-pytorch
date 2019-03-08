import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from config import *
import json

torch.manual_seed(1)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def prepare_sequence(pair_in, char2idx, tag2idx):
    pairs = pair_in.split('\n')
    sent, targets = [], []
    for pair in pairs:
        if pair.split('\t')[0]:
            sent.append(char2idx[pair.split('\t')[0]])
            targets.append(tag2idx[pair.split('\t')[1]])
    return torch.tensor(sent, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

with open('../tagged_data.txt', 'r') as f:
    data = f.read().decode('utf8').split('\n\r')
        
with open('../tag2idx.json', 'r') as j:
    tag2idx = json.load(j)

with open('../char2idx.json', 'r') as j:
    char2idx = json.load(j)

if __name__ == "__main__":

    model = LSTMTagger(6, 6, len(char2idx), len(tag2idx))
    loss_function = nn.NLLLoss()
    optimizer = opt.SGD(model.parameters(), lr=lr)

    for epoch in range(10):
        for sent in data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in, tags = prepare_sequence(sent, char2idx, tag2idx)
            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()

            print loss.item()