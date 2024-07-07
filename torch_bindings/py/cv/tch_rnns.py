#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'

import collections

import torch
import torch.nn as nn
import torch.nn.functional as functional


NUM_WORDS = 10000
EMBEDDING_VEC_LEN = 300
SENTENCE_LEN = 300
HIDDEN_SIZE = 512


class LSTM_GRU_BASE(nn.Module):
    RNN_MODEL = None
    def __init__(self):
        super(LSTM_GRU_BASE, self).__init__()
        self.embed = nn.Embedding(NUM_WORDS, EMBEDDING_VEC_LEN)
        self.rnn_model = self.__class__.RNN_MODEL(EMBEDDING_VEC_LEN, HIDDEN_SIZE, batch_first=True)
        self.drop_lstm = nn.Dropout(0.5)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_embed = self.embed(x)

        x_lstm, _ = self.rnn_model(x_embed)
        x_lstm_last_seq = x_lstm[:, -1, :]
        x_lstm_last_seq = self.drop_lstm(x_lstm_last_seq)

        logits = self.fc(x_lstm_last_seq)
        out = self.sig(logits)

        return out


class LSTM(LSTM_GRU_BASE):
    RNN_MODEL = nn.LSTM


class GRU(LSTM_GRU_BASE):
    RNN_MODEL = nn.GRU


class TextCNN(nn.Module):
    FILTER_SIZES = [2, 3, 4, 5]
    NUM_FILTERS = 256

    def __init__(self):
        super(TextCNN, self).__init__()

        self.emb = nn.Embedding(NUM_WORDS, EMBEDDING_VEC_LEN)
        self.convs = nn.ModuleList([])
        self.max_pools = nn.ModuleList([])
        self.relu = nn.ReLU()
        for fs in self.__class__.FILTER_SIZES:
            self.convs.append(nn.Conv2d(1, self.__class__.NUM_FILTERS, (fs, EMBEDDING_VEC_LEN)))
            self.max_pools.append(nn.MaxPool1d(SENTENCE_LEN - fs + 1))

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(
            len(self.__class__.FILTER_SIZES * self.__class__.NUM_FILTERS), 1
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_emb = self.emb(x)
        x_emb_reshaped = x_emb.unsqueeze(1)
        # print(x_emb_reshaped.shape)

        layers_out = []
        for conv, max_p in zip(self.convs, self.max_pools):
            cc = conv(x_emb_reshaped)
            # print(cc.shape)
            t = self.relu(cc).squeeze(3)
            # print(t.shape)
            # o = functional.max_pool1d(t, t.shape[2]).squeeze(2)
            o = max_p(t).squeeze(2)
            # print(o.shape)
            layers_out.append(o)
        concat_out = torch.cat(layers_out, 1)
        concat_out = self.dropout(concat_out)
        concat_out_flat = self.flatten(concat_out)
        logit = self.dense(concat_out_flat)
        out = self.sig(logit)
        return out
