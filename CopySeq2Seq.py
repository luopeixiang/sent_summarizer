import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from Seq2Seq import Seq2SeqSum, len_mask

INIT = 1e-2

class CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim):
        super(CopyLinear, self).__init__()

        self.v_c = nn.Parameter(torch.Tensor(context_dim))
        self.v_s = nn.Parameter(torch.Tensor(state_dim))
        self.v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self.v_c, -INIT, INIT)
        init.uniform_(self.v_s, -INIT, INIT)
        init.uniform_(self.v_i, -INIT, INIT)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, context, state, input):
        """args:
            context: context vector at time step t
            state: decoder state s_t
            input: decoder input x_t
        Return:
            the probability to generate a word from the source sentence
        """
        p_gen = F.sigmoid(
            torch.matmul(context, self.v_c.unsqueeze(1))
            +
            torch.matmul(state, self.v_s.unsqueeze(1))
            +
            torch.matmul(input, self.v_i.unsqueeze(1))
            +
            self.bias.unsqueeze(0)
        )

        return p_gen


class CopySeq2SeqSum(Seq2SeqSum):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, n_layer=1, bi_enc=True, dropout=0.0):

        super().__init__(vocab_size, emb_dim,
                         n_hidden, n_layer, bi_enc, dropout)
        #copy mechanism
        self._copy = CopyLinear(self.enc_out_dim, n_hidden, emb_dim)
        #a different docoder
        self.decoder = CopyLSTMDecoder(
            self._copy, self.embedding, n_hidden,
            vocab_size, self.enc_out_dim, n_layer,
            dropout=dropout
        )

    def forward(self, src, src_lengths, tgt, extend_src, extend_vsize):
        enc_outs, init_dec_states = self.encode(src, src_lengths)
        attn_mask = len_mask(src_lengths).to(src.device)

        logit = self.decoder(tgt, init_dec_states,
                             enc_outs, attn_mask,
                             extend_src, extend_vsize)
        return logit


class CopyLSTMDecoder(nn.Module):
    def __init__(self, copy_linear, embedding, n_hidden,
                 vocab_size, enc_out_dim, n_layers=1, dropout=0.1):

        super(CopyLSTMDecoder, self).__init__()
        self.copy_linear = copy_linear
        self.embedding = embedding
        self.n_layers = n_layers

        emb_size = embedding.weight.size(1)
        self.decoder_cell = nn.LSTMCell(emb_size, n_hidden)
        self.attn = nn.Linear(enc_out_dim, n_hidden)
        self.concat = nn.Linear(enc_out_dim+n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, vocab_size)

    def forward(self, tgt, init_states, enc_outs, attn_mask,
                extend_src, extend_vsize):

        max_len = tgt.size(1)
        states = init_states
        logits = []
        for i in range(max_len):
            #the i step target: [batch_size, 1]
            target_i = tgt[:, i:i+1]
            #one step decoding, use teacher forcing
            #import pdb;pdb.set_trace()
            logit, states = self._step(target_i, states,
                                       enc_outs, attn_mask,
                                       extend_src, extend_vsize)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)

        return logits

    def _step(self, inp, last_hidden, enc_outs, attn_mask,
              extend_src, extend_vsize):
        embed = self.embedding(inp).squeeze(1)

        h_t, c_t = self.decoder_cell(embed, last_hidden)
        attn_scores = self.get_attn(h_t, enc_outs, attn_mask)
        context = attn_scores.matmul(enc_outs.transpose(0, 1))

        #context : [batch_size, 1, enc_out_dim]
        context = context.squeeze(1)
        # Luong eq.5.
        concat_out = torch.tanh(self.concat(
            torch.cat([context, h_t], dim=1)
        ))

        logit = self.out(concat_out)

        bsize, vsize = logit.size()
        #add oov
        if extend_vsize > vsize:
            oov_logit = torch.Tensor(bsize, extend_vsize-vsize).fill_(1e-6)
            extend_logit = torch.cat(
                [logit, oov_logit.to(logit.device)],
                dim=1
            )
        else:
            extend_logit = logit
        extend_logit = F.softmax(extend_logit)

        #calculate probability of copy
        p_copy = self.copy_linear(context, h_t, embed)
        #obtain probability distribution over the extended vocabulary
        p_w = torch.log(
            ((1 - p_copy) * extend_logit).scatter_add(
                dim=1,
                index=extend_src,
                source=attn_scores.squeeze(1)*p_copy
            ) + 1e-8)

        return p_w, (h_t, c_t)

    def get_attn(self, dec_out, enc_outs, attn_mask):
        #implement attention mechanism
        keys = values = enc_outs
        query = dec_out.unsqueeze(0)

        #query: [1, batch_size, hidden_size]
        #enc_outs: [max_len, batch_size, hidden_size]
        #weights: [max_len, batch_size]
        weights = torch.sum(query * self.attn(keys), dim=2)
        weights = weights.transpose(0, 1)
        weights = weights.masked_fill(attn_mask==0, -1e18)
        weights = weights.unsqueeze(1)
        return F.softmax(weights, dim=2)























#
