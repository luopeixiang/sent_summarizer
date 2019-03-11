
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Seq2SeqSum(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, n_layer=1, bi_enc=True, dropout=0.0):
        super(Seq2SeqSum, self).__init__()

        self.n_layer = n_layer
        self.bi_enc = bi_enc  # whether encoder is bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional = bi_enc,
            dropout = 0 if n_layer==1 else dropout
        )

        #initial encoder hidden states as learnable parameters
        states_size0 = n_layer * (2 if bi_enc else 1)
        self.enc_init_h = nn.Parameter(
            torch.Tensor(states_size0, n_hidden)
        )
        self.enc_init_c = nn.Parameter(
            torch.Tensor(states_size0, n_hidden)
        )
        init.uniform_(self.enc_init_h, -1e-2, 1e-2)
        init.uniform_(self.enc_init_c, -1e-2, 1e-2)

        #reduce encoder states to decoder initial states
        self.enc_out_dim = n_hidden * (2 if bi_enc else 1)
        self._dec_h = nn.Linear(self.enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(self.enc_out_dim, n_hidden, bias=False)

        self.decoder = AttnDecoder(
            self.embedding, n_hidden, vocab_size,
            self.enc_out_dim, n_layer,
            dropout=dropout
            )

    def forward(self, src, src_lengths, tgt):
        """args:
            src: [batch_size, max_len]
            src_lengths: [batch_size]
            tgt: [batch_size, max_len]
        """
        enc_outs, init_dec_states = self.encode(src, src_lengths)
        attn_mask = len_mask(src_lengths).to(src.device)
        assert attn_mask.device == src.device
        logit = self.decoder(tgt, init_dec_states, enc_outs, attn_mask)
        #return logit: [batch_size, max_len, voc_size]
        return logit

    def encode(self, src, src_lengths):
        """run encoding"""

        #expand init encoder states in batch size dim
        size = (
            self.enc_init_c.size(0),
            len(src_lengths),
            self.enc_init_c.size(1)
        )
        init_hidden = (
            self.enc_init_h.unsqueeze(1).expand(*size).contiguous(),
            self.enc_init_c.unsqueeze(1).expand(*size).contiguous()
        )

        embed = self.embedding(src.transpose(0, 1))
        padded_seq = pack_padded_sequence(embed, src_lengths)
        enc_out, hidden = self.encoder(padded_seq, init_hidden)
        outputs, _ = pad_packed_sequence(enc_out)

        #only consider n_layers = 1
        #init dec_input and hidden
        if self.bi_enc:
            h, c = hidden
            h, c = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_dec_states = (self._dec_h(h).squeeze(0),
                            self._dec_c(c).squeeze(0))
        return outputs, init_dec_states

    def greedy_decode(self, inp, inp_len, word2id, max_len):
        """greedy decoing(support batch decode)
        args:
            inp: [batch_size, max_len] batch size sentences
            inp_len: [batch_size] the lengths of sentence
            word2id: a dictionary convert word to id
            max_len: max decoding length
        return:
            results:[batch_size] a target sentences list
        """
        #encoding
        enc_outs, states = self.encode(inp, inp_len)
        attn_mask = len_mask(inp_len).to(inp.device)

        SOS, END, UNK = word2id["<start>"], word2id["<end>"], word2id['<unk>']
        batch_size = inp.size(0)
        # initial the first step of decoder's input
        dec_inp = torch.ones([batch_size, 1]).long() * SOS
        # store resutls
        results = torch.ones([batch_size, max_len]).long() * END

        for i in range(max_len):
            logit, states = self.decoder._step(
                            dec_inp, states, enc_outs, attn_mask)

            max_word_inds = torch.max(logit, dim=1, keepdim=True)[1] #[batch_size, 1]
            results[:, i:i+1] = max_word_inds

        # convert word id in results to sentence
        id2word = dict((id_, word) for word, id_ in word2id.items())
        dec_outs = []
        for result in results:
            dec_sent = []
            for id_ in result:
                if id_ == END:
                    break
                dec_sent.append(id2word.get(id_, UNK))
            dec_outs.append(" ".join(dec_sent))
        return dec_outs


    def bs_decode(self, inp, word2id, bsize=4):
        """beam search decoding(not support batch)
        args:
            inp: [1, max_len] represent a source sentence
            word2id: a dictionary convert word to id
            bsize: beam size to generate sentence summary
        return:
            dec_out: [dec_len] represent target sentence

        TODO: BATCH BEAM DECODE
        """
        inp_len = torch.LongTensor([inp.size(1)])
        attn_mask = torch.ones_like(inp).long()
        SOS, END = word2id["<start>"], word2id["<end>"]
        #store top k sequence score, init it as zero
        top_k_scores = torch.zeros(bsize)
        #store top k squence
        top_k_words = torch.ones([bsize, 1]).long() * SOS

        #store completed seqs and their scores
        completed_seqs = []
        completed_seqs_score = []

        prev_words = top_k_words
        vocab_size = len(word2id)
        step = 1
        k = bsize
        #encoding
        enc_outs, (h, c) = self.encode(inp, inp_len)
        h = h.expand(bsize, h.size(1))
        c = c.expand(bsize, h.size(1))
        while True:

            #decoding
            dec_out, (h, c) = self.decoder._step(
                prev_words, (h, c), enc_outs, attn_mask)
            logit = F.log_softmax(dec_out, dim=1) #[k, vocab_size]

            logit = top_k_scores.unsqueeze(1).expand_as(logit) + logit
            #current time step topk
            if step == 1:
                top_k_scores, ctop_k_words = logit[0].topk(k, dim=0)
            else:
                top_k_scores, ctop_k_words = logit.view(-1).topk(k, dim=0)

            #prev words sequence index in top_k_words
            pw_inds_tk = ctop_k_words / vocab_size
            #next word index in vocab
            next_word_inds = ctop_k_words % vocab_size
            #add new words to sequences
            top_k_words = torch.cat([top_k_words[pw_inds_tk],
                                    next_word_inds.unsqueeze(1)],
                                    dim=1)

            #check if exist word sequence reach end token
            incomplete_word_ind = [i for i, word_ind in enumerate(next_word_inds)
                                    if word_ind != word2id['<end>']]
            complete_word_ind = [ind for ind in range(len(next_word_inds))
                                    if ind not in incomplete_word_ind]

            if len(complete_word_ind):
                completed_seqs.extend(top_k_words[complete_word_ind])
                completed_seqs_score.extend(top_k_scores[complete_word_ind])
            k -= len(complete_word_ind)
            if k == 0:
                break

            #prepare for next time step
            top_k_words = top_k_words[incomplete_word_ind]
            top_k_scores = top_k_scores[incomplete_word_ind]
            h = h[pw_inds_tk[incomplete_word_ind]]
            c = c[pw_inds_tk[incomplete_word_ind]]
            prev_words = top_k_words[:, -1:]
            step += 1

        assert len(completed_seqs) == bsize
        max_score_index = completed_seqs_score.index(max(completed_seqs_score))
        max_score_seqs = completed_seqs[max_score_index]

        id2word = dict((id_, word) for word, id_ in word2id.items())
        words = [id2word[id_.item()] for id_ in max_score_seqs
                    if id_ not in [SOS, END]]
        sent = " ".join(words)

        return sent

class AttnDecoder(nn.Module):
    def __init__(self, embedding, hidden_size,
                 output_size, enc_out_dim, n_layers=1, dropout=0.1):
        super(AttnDecoder, self).__init__()

        self.embedding = embedding
        self.n_layers = n_layers

        emb_size = embedding.weight.size(1)
        self.decoder_cell = nn.LSTMCell(emb_size, hidden_size)
        self.attn = nn.Linear(enc_out_dim, hidden_size)
        self.concat = nn.Linear(enc_out_dim+hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, target, init_states, enc_outs, attn_mask):
        max_len = target.size(1)
        states = init_states
        logits = []
        for i in range(max_len):
            #the i step target: [batch_size, 1]
            target_i = target[:, i:i+1]
            #one step decoding, use teacher forcing
            #import pdb;pdb.set_trace()
            logit, states = self._step(target_i, states, enc_outs, attn_mask)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)

        return logits

    def _step(self, inp, last_hidden, enc_outs, attn_mask):
        embed = self.embedding(inp).squeeze(1)
        # run one step decoding
        h_t, c_t = self.decoder_cell(embed, last_hidden)
        attn_scores = self.get_attn(h_t, enc_outs, attn_mask)
        context = attn_scores.matmul(enc_outs.transpose(0, 1))

        #context : [batch_size, 1, enc_out_dim]
        context = context.squeeze(1)
        # Luong eq.5.
        concat_out = torch.tanh(self.concat(
            torch.cat([context, h_t], dim=1)
        ))

        logit = F.log_softmax(self.out(concat_out), dim=-1)
        return logit, (h_t, c_t)


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


        #另一种实现
        # values = enc_outs.transpose(0, 1) #[batch_size, max_len, hsize]
        # keys = self.attn(values).transpose(1, 2) #batch_size, hsize, max_len
        # query = dec_out.unsqueeze(1) #[batch_size, 1, hsize]
        # attn_scores = query.matmul(keys) #batch_size, 1, max_len
        # attn_scores = attn_scores.masked_fill(attn_mask==0, -1e18)
        # context = query.matmul(values) # batch_size, 1, hidden_size

        # return [batch_size, 1 max_len]
        return F.softmax(weights, dim=2)

#helper function
def len_mask(lens):
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask


if __name__ == "__main__":
    #test
    model = Seq2SeqSum(300, 64, 128)
    src = torch.randint(299, (32, 15)).long()
    src_lengths =torch.randint(2, 14, (32,)).long()
    lens = torch.LongTensor(list(reversed(sorted(src_lengths.tolist()))))
    tgt = torch.randint(299, (32, 10)).long()
    out = model(src, lens, tgt)
    print(out)





















    #
