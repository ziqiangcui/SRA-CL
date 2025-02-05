import numpy as np
import torch
import torch.nn as nn
from recbole.model.layers import TransformerEncoder
from utils import info_nce, info_nce_single
import torch.nn.functional as F

class TransRec(torch.nn.Module):
    def __init__(self, config):
        super(TransRec, self).__init__()
        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size  # same as embedding_size
        self.inner_size = config.inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attn_dropout_prob = config.attn_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        self.batch_size = config.train_batch_size
        self.initializer_range = config.initializer_range
        self.loss_type = config.loss_type
        self.n_items = config.item_num
        self.max_seq_length = config.maxlen
        # define contrastive params
        self.temperature = config.temperature
        self.item_neighbors = config.sorted_indices
        self.user_neighbors = config.user_sorted_indices
        self.item_semantic_emb = config.item_semantic_emb
        self.user_semantic_emb = config.user_semantic_emb
        # follow original gat
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.W = nn.Parameter(torch.empty(size=(1024, self.hidden_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items +1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.nce_fct = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    

    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        # print("debug output", output.shape)
        output = output[:,-1, :]
        return output  # [B H]

    def calculate_loss(self, item_seq, pos_items):
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight[1:self.n_items+1]  # unpad the augmentation mask
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        pos_items = pos_items-1
        loss = self.loss_fct(logits, pos_items)
        return loss
    
    def calculate_item_cl_loss(self, aug_seq1, aug_seq2, seq_len):
        # Forward pass for both augmented sequences
        seq_output1 = self.forward(aug_seq1)
        seq_output2 = self.forward(aug_seq2)
        
        # Filter sequences based on the length condition
        valid_indices = seq_len > 0
        if valid_indices.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)  # Return zero loss if no valid sequences

        # Select only valid sequences
        valid_seq_output1 = seq_output1[valid_indices]
        valid_seq_output2 = seq_output2[valid_indices]
        
        # Calculate InfoNCE loss only for valid sequences
        nce_logits, nce_labels = info_nce(valid_seq_output1, valid_seq_output2, temp=self.temperature, batch_size=valid_seq_output1.shape[0], sim="dot")
        loss = self.nce_fct(nce_logits, nce_labels)
        
        return loss
    
    def calculate_user_cl_loss(self, seq, neighbor_seqs, seq_unique_id):
        seq_output = self.forward(seq) # bs, 64
        bs, neighbor_num, seq_len = neighbor_seqs.shape
        # print("debug neighbor_num", neighbor_num)
        
        neighbor_seqs = neighbor_seqs.reshape(bs*neighbor_num, seq_len)
        neighbor_seqs_output = self.forward(neighbor_seqs) # bs*neighbor_num, 64
        neighbor_seqs_output = neighbor_seqs_output.reshape(bs, neighbor_num, self.hidden_size)
    
        gumbel_softmax_probs = self.select_similar_user_probs(seq_unique_id)
        
        gumbel_softmax_probs = gumbel_softmax_probs.unsqueeze(-1)  # (bs, neighbor_num, 1)
        neighbor_seqs_output_weighted_sum = torch.sum(gumbel_softmax_probs * neighbor_seqs_output, dim=1)  # (bs, 64)
        
        nce_logits, nce_labels = info_nce(seq_output, neighbor_seqs_output_weighted_sum, temp=self.temperature, batch_size=seq.shape[0], sim="dot")
        loss = self.nce_fct(nce_logits, nce_labels)
        return loss
    
    def select_similar_user_probs(self, seq_unique_id):
        """
        seq: [bs, seq_len]
        neighbor_seq: [bs, 10, seq_len]
        seq_unique_id: [bs, 1]
        """
        
        neighbors =self.user_neighbors[seq_unique_id] # B L
        neighbors_semantic_emb = self.user_semantic_emb[neighbors] # B L d'
        neighbors_semantic_emb = torch.matmul(neighbors_semantic_emb, self.W)  # B L d
        # print("debug neighbors_semantic_emb", neighbors_semantic_emb.shape)
        
        seq_semantic_emb = self.user_semantic_emb[seq_unique_id] # B,d'
        seq_semantic_emb = seq_semantic_emb.unsqueeze(1).repeat(1, neighbors.shape[-1],1)  # B,L,d'
        seq_semantic_emb = torch.matmul(seq_semantic_emb, self.W) # B,L,d
        # print("debug seq_semantic_emb", seq_semantic_emb.shape)
        
        W_concat = torch.cat((seq_semantic_emb, neighbors_semantic_emb), dim=-1) # N,L,2d
        
        attention = torch.matmul(W_concat, self.a).squeeze(-1) # N,L
        attention = self.leakyrelu(attention)
        
        attention = F.softmax(attention, dim=1) # N,L
    
        attention = F.dropout(attention, 0.5, training=self.training) # N,L
        
        return attention

    def predict(self, item_seq, test_item):
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, item_seq):
        seq_output = self.forward(item_seq)
        test_items_emb = self.item_embedding.weight[1:self.n_items+1]  # unpad the augmentation mask
      
        # print("debug seq_output", seq_output.shape)
        # print("debug test_items_emb", test_items_emb.shape)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)