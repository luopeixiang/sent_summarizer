import pickle as pkl

PAD, UNK, START, END = 0, 1, 2, 3
def make_word2id(wc_pkl, vocab_size):
    word2id = {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END

    with open(wc_pkl, 'rb') as f:
        wc = pkl.load(f)
        for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
            word2id[w] = i

    return word2id

def load_sents(data_path):
    with open(data_path, 'r') as f:
        sents = [line.strip('\n') for line in f.readlines()]
        sents = [sent for sent in sents if sent != ''] # remove empty sent
    return sents
