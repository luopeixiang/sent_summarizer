import collections
import pickle as pkl

def get_tokens(path):
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip("\n").split()
            tokens = [t.strip() for t in tokens] #strip
            tokens = [t for t in tokens if t != ""] #remove empty
            yield tokens

def make_vocab(src_path, tgt_path, vocab_path):
    vocab_counter = collections.Counter()

    print("Reading sentences...")
    for tokens in get_tokens(src_path):
        vocab_counter.update(tokens)
    for tokens in get_tokens(tgt_path):
        vocab_counter.update(tokens)

    #saving
    print("Writing vocab file...")
    with open(vocab_path, 'wb') as f:
        pkl.dump(vocab_counter, f)
    print("Finished writing vacab file")

if __name__ == "__main__":

    SRC_PATH = "/home/luo/DATA/Gigaword/sumdata/Giga/input.txt"
    TARGET_PATH = "/home/luo/DATA/Gigaword/sumdata/Giga/task1_ref0.txt"
    VOCAB_PATH = './vocab.pkl'
    make_vocab(SRC_PATH, TARGET_PATH, VOCAB_PATH)
