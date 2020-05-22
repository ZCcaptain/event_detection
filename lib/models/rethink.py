import torch.nn as nn
from lib.utils.tools import load_dict
from lib.utils.tools import load_embedding
import os
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from functools import partial
import math
from torch.nn import Parameter

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class RethinkNaacl(nn.Module):
    def __init__(self, hyper):
        super(RethinkNaacl, self).__init__()
        self.hyper = hyper
        self.device = torch.device("cuda:%d" % hyper.gpu if torch.cuda.is_available() else "cpu")
        self.data_root = hyper.data_root
        self.wdict = load_dict(os.path.join(self.data_root, 'ace/dicts/word_dict.txt'))
        self.edict = load_dict(os.path.join(self.data_root, 'ace/dicts/ent_dict.txt'))
        ydict = load_dict(os.path.join(self.data_root, 'ace/dicts/label_dict.txt'))
        self.ydict = {k.lower(): v for k, v in ydict.items()}

        self.input_dropout = nn.Dropout(p=0.5)

        # word embedding
        word_emb_data = load_embedding(os.path.join(self.data_root, 'ace/embeddings/200.txt'))
        # word_weight = torch.FloatTensor(word_emb_data)
        self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(word_emb_data))
        # self.word_embeddings = nn.Embedding(word_emb_data.shape[0], word_emb_data.shape[1])
        # self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb_data))
        # self.embedding.weight.requires_grad = True

        # entity embedding
        self.ent_embedding = nn.Embedding(hyper.n_ent, hyper.dim_ent)

        #event type embedding 1
        self.hidden_dim = hyper.emb_dim + hyper.dim_ent
        self.evt_embedding = nn.Embedding(hyper.n_class, self.hidden_dim)
        #event type embedding 2
        self.evt_embedding_last = nn.Embedding(hyper.n_class, self.hidden_dim)
        # self.evt_embedding_last = self.evt_embedding

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        # self.ln2 = nn.LayerNorm(self.hidden_dim)

        #lstm encoder
        self.encoder = nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=hyper.bidirectional, batch_first=True)

        self.W = nn.Parameter(torch.randn(hyper.n_class, hyper.n_class))

        self.adj = nn.Parameter(torch.randn(hyper.n_class, hyper.n_class))
        self.gc1 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gc2 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.relu = nn.LeakyReLU(0.2)

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.3f}, epoch: {}/{}:".format(
            output['loss'].item(), epoch, epoch_num)

    def forward(self, sample, is_train=True):

        # GCN
        if self.hyper.use_gcn:
            # adj = gen_adj(self.A).detach()
            x = self.gc1(self.evt_embedding.weight, self.adj)
            x = self.relu(x)
            x = self.gc2(x, self.adj)

        # tokens_id = sample.tokens_id
        # entity_tag = sample.entity_tag


        tokens_len = sample.token_len
        tokens = pad_sequence(sample.tokens_id, batch_first=True, padding_value=0).to(self.device)
        entity_tag = pad_sequence(sample.entity_tag, batch_first=True, padding_value=0).to(self.device)
        tooken_embs = self.word_embeddings(tokens)
        ent_tag_embs = self.ent_embedding(entity_tag)
        lstm_in = torch.cat((tooken_embs, ent_tag_embs), dim=2)
        lstm_in = self.ln1(lstm_in)
        lstm_in = self.input_dropout(lstm_in)

        lstm_in_pack = pack_padded_sequence(lstm_in, tokens_len, batch_first=True)

        o, state = self.encoder(lstm_in_pack)
        o, _ = pad_packed_sequence(o, batch_first=True)
        if self.hyper.bidirectional:
            o = (lambda a: sum(a) / 2)(torch.split(o, self.hidden_dim, dim=2))

        # attention score
        mask = tokens != 0
        if self.hyper.use_gcn:
            att_score = torch.einsum('blh,yh->bly', o, x)
        else:
            att_score = torch.einsum('blh,yh->bly', o, self.evt_embedding.weight)
        att_mask = mask.unsqueeze(2).expand(-1, -1, att_score.shape[2])
        att_score[~att_mask] = float('-inf')
        att_prob = F.softmax(att_score, dim=1)
        att_vec = torch.einsum('blh,bly->byh', o, att_prob)
        # att_vec = self.ln2(att_vec)

        if self.hyper.use_gcn:
            score1 = torch.sum(torch.mul(att_vec, x), dim=2)
        else:
            score1 = torch.sum(torch.mul(att_vec, self.evt_embedding.weight), dim=2)

        cell, hidden = state
        # global score
        if self.hyper.bidirectional:
            avg_hidden = torch.sum(hidden, dim=0)/2
        else:
            avg_hidden = hidden.squeeze()

        # score2 = torch.einsum('bf,yf->by', avg_hidden, self.evt_embedding_last.weight)

        # if self.hyper.use_gcn:
        #     score2 = torch.einsum('bf,yf->by', avg_hidden, x)
        # else:
        #     score2 = torch.einsum('bf,yf->by', avg_hidden, self.evt_embedding_last.weight)
        score2 = torch.einsum('bf,yf->by', avg_hidden, self.evt_embedding_last.weight)

        score = score1 * self.hyper.alpha + score2 * (1 - self.hyper.alpha)
        # scoreRethink = score + torch.sigmoid(torch.mm(score, self.W))
        # score2 = score + F.tanh(torch.mm(score, self.W))

        output = {}
        if not is_train:
            predict_event = self.inference(score)
            output['predict'] = predict_event
        loss = 0

        if is_train:
            gold_label = torch.stack(sample.event_id).to(self.device)
            loss1 = F.binary_cross_entropy_with_logits(score, gold_label, reduction='mean')
            # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * settings['l2_weight']
            # pos_weight = torch.ones([35])*3
            # pos_weight[26] = 0
            # loss2 = F.binary_cross_entropy_with_logits(scoreRethink, gold_label, reduction='mean')
            # loss = loss1 + loss2
            loss = loss1
        output['att_vec'] = att_vec
        output['loss'] = loss
        output['att_prob'] = att_prob
        output['description'] = partial(self.description, output=output)
        return output

    def inference(self, score):
        predict_event = torch.sigmoid(score) > self.hyper.threshold
        return predict_event

    # def decode(self, predict_event):
    #     idx = torch.nonzero(predict_event)