# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
from utils import layers
from models.BaseModel import SequentialModel


class RCF(SequentialModel):
    reader = 'KGReader'
    extra_log_args = ['num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--neg_head_p', type=float, default=0.5,
                            help='The probability of sampling negative head entity.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=1,
                            help='Number of attention heads.')
        parser.add_argument('--gamma', type=float, default=-1,
                            help='Coefficient of KG loss (-1 for auto-determine).')
        parser.add_argument('--attention_size', type=int, default=10,
                            help='Size of attention hidden space.')
        parser.add_argument('--pooling', type=str, default='average',
                            help='Method of pooling relational history embeddings: average, max, attention')
        parser.add_argument('--include_val', type=int, default=1,
                            help='Whether include relation value in the relation representation')
        parser.add_argument('--include_kge', type=int, default=1,
                            help='Whether include Knowledge Graph Embedding module')
        parser.add_argument('--message', type=str, default="",
                            help='additional message')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.relation_num = corpus.n_relations
        self.entity_num = corpus.n_entities
        self.emb_size = args.emb_size
        self.neg_head_p = args.neg_head_p
        self.layer_num = args.num_layers
        self.head_num = args.num_heads
        self.attention_size = args.attention_size
        self.pooling = args.pooling.lower()
        self.include_val = args.include_val
        self.gamma = args.gamma
        self.include_kge = args.include_kge
        if self.gamma < 0:
            self.gamma = len(corpus.relation_df) / len(corpus.all_df)
        super().__init__(args, corpus)

    def _define_params(self):
        self.user_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.entity_embeddings = nn.Embedding(self.entity_num, self.emb_size)
        self.relation_embeddings = nn.Embedding(self.relation_num, self.emb_size)
        self.relational_dynamic_aggregation = RelationalDynamicAggregation(
            self.relation_num, self.relation_embeddings, self.include_val, self.device
        )
        self.attn_head = layers.MultiHeadAttention(self.emb_size, self.head_num, bias=False)
        self.W1 = nn.Linear(self.emb_size, self.emb_size)
        self.W2 = nn.Linear(self.emb_size, self.emb_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.emb_size)
        # Pooling
        if self.pooling == 'attention':
            self.A = nn.Linear(self.emb_size, self.attention_size)
            self.A_out = nn.Linear(self.attention_size, 1, bias=False)
        # Prediction
        self.item_bias = nn.Embedding(self.item_num + 1, 1)


    def actions_before_train(self):
        pass

    def forward(self, feed_dict):
        self.check_list = []
        prediction = self.rec_forward(feed_dict)
        out_dict = {'prediction': prediction}
        if feed_dict['phase'] == 'train':
            if self.include_kge:
                kg_prediction = self.kg_forward(feed_dict)
                out_dict['kg_prediction'] = kg_prediction
        return out_dict

    def rec_forward(self, feed_dict):
        u_ids = feed_dict['user_id']  # B
        i_ids = feed_dict['item_id']  # B * -1
        target_i_ids = i_ids[:,0]  # B
        v_ids = feed_dict['item_val']  # B * -1 * R
        history = feed_dict['history_items']  # B * H
        batch_size, seq_len = history.shape

        u_vectors = self.user_embeddings(u_ids)
        i_vectors = self.entity_embeddings(i_ids)
        v_vectors = self.entity_embeddings(v_ids)  # B * -1 * R * V
        his_vectors = self.entity_embeddings(history)  # B * H * V

        valid_mask = (history > 0).view(batch_size, 1, seq_len, 1)
        context, target_attention = self.relational_dynamic_aggregation(
            his_vectors, i_vectors, v_vectors, valid_mask)  # B * -1 * R * V
        for i in range(self.layer_num):
            residual = context
            # self-attention
            context = self.attn_head(context, context, context)
            # feed forward
            context = self.W1(context)
            context = self.W2(context.relu())
            # dropout, residual and layer_norm
            context = self.dropout_layer(context)
            context = self.layer_norm(residual + context)

        """
        Pooling Layer
        """
        if self.pooling == 'attention':
            query_vectors = context * u_vectors[:, None, None, :]  # B * -1 * R * V
            user_attention = self.A_out(self.A(query_vectors).tanh()).squeeze(-1)  # B * -1 * R
            user_attention = (user_attention - user_attention.max()).softmax(dim=-1)
            his_vector = (context * user_attention[:, :, :, None]).sum(dim=-2)  # B * -1 * V
        elif self.pooling == 'max':
            his_vector = context.max(dim=-2).values  # B * -1 * V
        else:
            his_vector = context.mean(dim=-2)  # B * -1 * V

        """
        Prediction
        """
        i_bias = self.item_bias(i_ids).squeeze(-1)
        prediction = ((u_vectors[:, None, :] + his_vector) * i_vectors).sum(dim=-1)
        prediction = prediction + i_bias


        return prediction.view(feed_dict['batch_size'], -1)

    def kg_forward(self, feed_dict):
        head_ids = feed_dict['head_id'].long()  # B * -1
        tail_ids = feed_dict['tail_id'].long()  # B * -1
        value_ids = feed_dict['value_id'].long()  # B
        relation_ids = feed_dict['relation_id'].long()  # B

        head_vectors = self.entity_embeddings(head_ids)
        tail_vectors = self.entity_embeddings(tail_ids)
        value_vectors = self.entity_embeddings(value_ids)
        relation_vectors = self.relation_embeddings(relation_ids)

        # DistMult
        if self.include_val:
            prediction = (head_vectors * (relation_vectors + value_vectors)[:, None, :] * tail_vectors).sum(-1)
        else:
            prediction = (head_vectors * relation_vectors[:, None, :] * tail_vectors).sum(-1)
        return prediction

    def loss(self, out_dict):
        rec_loss, kg_loss = 0,0
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        rec_loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
        loss = rec_loss
        if self.include_kge:
            predictions = out_dict['kg_prediction']
            pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
            neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
            kg_loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
            loss += self.gamma * kg_loss
        
        return loss, [rec_loss,kg_loss]

    class Dataset(SequentialModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            if self.phase == 'train':
                self.kg_data, self.neg_heads, self.neg_tails = None, None, None

        def _prepare(self):
            # Prepare item-to-value dict
            item_val = self.corpus.item_meta_df.copy()
            item_val[self.corpus.item_relations] = 0  # set the value of natural item relations to None
            for idx, r in enumerate(self.corpus.attr_relations):
                base = self.corpus.n_items + np.sum(self.corpus.attr_max[:idx])
                item_val[r] = item_val[r].apply(lambda x: x + base).astype(int)
            item_vals = item_val[self.corpus.relations].values  # this ensures the order is consistent to relations
            self.item_val_dict = dict()
            for item, vals in zip(item_val['item_id'].values, item_vals.tolist()):
                self.item_val_dict[item] = [0] + vals  # the first dimension None for the virtual relation
            super()._prepare()

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            feed_dict['user_id'] = self.data['user_id'][index]
            feed_dict['item_val'] = [self.item_val_dict[item] for item in feed_dict['item_id']]
            if self.phase == 'train':
                feed_dict['head_id'] = np.concatenate([[self.kg_data['head'][index]], self.neg_heads[index]])
                feed_dict['tail_id'] = np.concatenate([[self.kg_data['tail'][index]], self.neg_tails[index]])
                feed_dict['relation_id'] = self.kg_data['relation'][index]
                feed_dict['value_id'] = self.kg_data['value'][index]
                feed_dict["neg_his_items"] = self.neg_his_items[index]
            return feed_dict

        def generate_kg_data(self) -> pd.DataFrame:
            rec_data_size = len(self)
            replace = (rec_data_size > len(self.corpus.relation_df))
            kg_data = self.corpus.relation_df.sample(n=rec_data_size, replace=replace).reset_index(drop=True)
            kg_data['value'] = np.zeros(len(kg_data), dtype=int)  # default for None
            tail_select = kg_data['tail'].apply(lambda x: x < self.corpus.n_items)
            item_item_df = kg_data[tail_select]
            item_attr_df = kg_data.drop(item_item_df.index)
            item_attr_df['value'] = item_attr_df['tail'].values

            # construct shared attribute item to item relation
            # head: item, relation: rlation_type, tail: item, value: rlation_value
            sample_tails = list()
            for head, val in zip(item_attr_df['head'].values, item_attr_df['tail'].values):
                share_attr_items = self.corpus.share_attr_dict[val]
                tail_idx = np.random.randint(len(share_attr_items))
                sample_tails.append(share_attr_items[tail_idx])
            item_attr_df['tail'] = sample_tails
            kg_data = pd.concat([item_item_df, item_attr_df], ignore_index=True)
            return kg_data

        def actions_before_epoch(self):
            super().actions_before_epoch()
            self.kg_data = self.generate_kg_data()
            heads, tails = self.kg_data['head'].values, self.kg_data['tail'].values
            relations, vals = self.kg_data['relation'].values, self.kg_data['value'].values
            self.neg_heads = np.random.randint(1, self.corpus.n_items, size=(len(self.kg_data), self.model.num_neg))
            self.neg_tails = np.random.randint(1, self.corpus.n_items, size=(len(self.kg_data), self.model.num_neg))
            for i in range(len(self.kg_data)):
                item_item_relation = (tails[i] <= self.corpus.n_items)
                for j in range(self.model.num_neg):
                    if np.random.rand() < self.model.neg_head_p:  # sample negative head
                        tail = tails[i] if item_item_relation else vals[i]
                        while (self.neg_heads[i][j], relations[i], tail) in self.corpus.triplet_set:
                            self.neg_heads[i][j] = np.random.randint(1, self.corpus.n_items)
                        self.neg_tails[i][j] = tails[i]
                    else:  # sample negative tail
                        head = heads[i] if item_item_relation else self.neg_tails[i][j]
                        tail = self.neg_tails[i][j] if item_item_relation else vals[i]
                        while (head, relations[i], tail) in self.corpus.triplet_set:
                            self.neg_tails[i][j] = np.random.randint(1, self.corpus.n_items)
                            head = heads[i] if item_item_relation else self.neg_tails[i][j]
                            tail = self.neg_tails[i][j] if item_item_relation else vals[i]
                        self.neg_heads[i][j] = heads[i]
            self.neg_his_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.corpus.history_max))
            for i, u in enumerate(self.data['user_id']):
                user_clicked_set = self.corpus.user_clicked_set[u]
                for j in range(self.corpus.history_max):
                    while self.neg_his_items[i][j] in user_clicked_set:
                        self.neg_his_items[i][j] = np.random.randint(1, self.corpus.n_items)       


class RelationalDynamicAggregation(nn.Module):
    def __init__(self, n_relation, relation_embeddings, include_val, device):
        super().__init__()
        self.relation_embeddings = relation_embeddings
        self.include_val = include_val
        self.n_relation = n_relation
        self.relation_range = torch.from_numpy(np.arange(n_relation)).to(device)

    def forward(self, seq, target, target_value, valid_mask):
        '''
            seq: user history item embeddings, B * H * V
            delta_t_n: time interval between current item and history items, B * H
            target: target item embeddings, B * -1 * V
            target_value: target item value embeddings, B * -1 * R * V
            valid_mask: mask of valid history items, B * H
        '''
        r_vectors = self.relation_embeddings(self.relation_range)  # R * V
        if self.include_val:
            rv_vectors = r_vectors[None, None, :, :] + target_value
            ri_vectors = rv_vectors * target[:, :, None, :]  # B * -1 * R * V
        else:
            ri_vectors = r_vectors[None, None, :, :] * target[:, :, None, :]  # B * -1 * R * V
        attention = (seq[:, None, :, None, :] * ri_vectors[:, :, None, :, :]).sum(-1)  # B * -1 * H * R

        target_attention = attention[:,0]
        target_attention = target_attention - target_attention.max()  # B * H * R
        target_attention = target_attention.softmax(dim=-1)  # B * H * R
        target_attention = torch.where(valid_mask.squeeze(1).repeat(1,1,self.n_relation), target_attention, 0.)
        
        # shift masked softmax
        attention = attention - attention.max()
        attention = attention.masked_fill(valid_mask == 0, -np.inf).softmax(dim=-2)

        # attentional aggregation of history items
        context = (seq[:, None, :, None, :] * attention[:, :, :, :, None]).sum(-3)  # B * -1 * R * V
        return context, target_attention
