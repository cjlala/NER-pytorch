#encoding=utf-8
#Author: CJ
#Date: 2019-02-28 14:37:11

import torch # 主包
import torch.nn as nn # 包含各个子cell
import torch.nn.functional as F # 包含各种激活函数
import torch.optim as opt # 包含各种激活函数
from config import *
from utils import *
from record import Record 
import json
from tqdm import tqdm

torch.manual_seed(1024)

# lstm = nn.LSTM(3, 3) # (input_size, hidden_size)
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

if __name__ == "__main__":

    model = LSTMTagger(64, 64, len(char2idx), len(tag2idx))
    loss_function = nn.NLLLoss()
    optimizer = opt.Adam(model.parameters(), lr=lr)

    record = Record()

    for epoch in range(20):
        record.clear()
        count_dic = {
            'BODY' : [0,0,0],
            'CUE' : [0,0,0],
            'SYMPTOM' : [0,0,0],
            'CHECK' : [0,0,0],
            'DISEASE' : [0,0,0]
        }
        for sent in tqdm(data[:-1]):

            model.zero_grad()

            model.hidden = model.init_hidden()

            sentence_in, tags = prepare_sequence(sent, char2idx, tag2idx)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()

            predicions = argmax(tag_scores)

            y_true_copy, y_pred_copy = tags.detach().numpy(), predicions.detach().numpy()
            sent_correct_pred, sent_total_pred, sent_total_true = eval_chunk(y_true_copy, y_pred_copy, tag2idx, count_dic)
            # record.update(sent_correct_pred, sent_total_pred, sent_total_true, loss.item()/TOTAL_WORD_COUNT)
        # record.show(epoch+1)
        record.show_category(epoch+1, count_dic)