#encoding=utf-8
#Author: CJ
#Date: 2019-03-06 11:08:26

import torch
import numpy as np
import json

def prepare_sequence(pair_in, char2idx, tag2idx):
    pairs = pair_in.split('\n')
    sent, targets = [], []
    for pair in pairs:
        if pair.split('\t')[0]:
            sent.append(char2idx[pair.split('\t')[0]])
            targets.append(tag2idx[pair.split('\t')[1]])
    return torch.tensor(sent, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx

def cal_char_accuracy(y_true, y_pred):
    return (y_true == y_pred).float().sum().item()

def eval_category_chunk(correct_set, pred_set, true_set, count_dic):
    for chunk in correct_set:
        count_dic[chunk[0]][0] += 1
    for chunk in pred_set:
        count_dic[chunk[0]][1] += 1
    for chunk in true_set:
        count_dic[chunk[0]][2] += 1
    return

def eval_chunk(y_true, y_pred, tag2idx, count_dic):
    y_true_chunks = set(get_chunks(y_true, tag2idx))
    y_pred_chunks = set(get_chunks(y_pred, tag2idx))

    correct_set = y_true_chunks & y_pred_chunks
    eval_category_chunk(correct_set, y_pred_chunks, y_true_chunks, count_dic)
    correct_pred = len(correct_set)
    total_pred = len(y_pred_chunks)
    total_true = len(y_true_chunks)

    return correct_pred, total_pred, total_true

def get_chunk_type(tok, idx2tag):
    tag_name = idx2tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[1]
    return tag_class, tag_type

def get_chunks(seq, tag2idx):
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == tag2idx['O'] and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        elif tok != tag2idx['O']:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx2tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
    
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    with open('../tag2idx.json', 'r') as j:
        tag2idx = json.load(j)
    print get_chunks([17,2,14,9,9,13,1,16], tag2idx)