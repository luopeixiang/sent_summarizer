import torch

from Seq2Seq import Seq2SeqSum
from utils import make_word2id

# load model and use beam search
# to generate sentence summary
def beam_search(model_path, word2id, beam_size=4):
    #load model
    model = Seq2SeqSum(len(word2id), 128, 256, 1)
    ckpt = torch.load(model_path)['state_dict']
    model.load_state_dict(ckpt)

    #iter test data loader
    test_src = """villarreal coach manuel pellegrini was understandably
    disappointed at dropping two points in the title race as malaga
    scored a ##th minute equaliser to snatch a #-# draw on sunday ."""
    test_src = "ministers from ## african nations and the united states are to hold annual talks this week aimed at devising strategies for a vibrant private sector in africa and stepping up trade ."
    test_src = "days of rioting between christians and muslims in eastern pakistan following allegations that a quran was defiled escalated saturday , leaving six christians dead , including a child , authorities said ."

    src = torch.LongTensor([[word2id.get(word, word2id['<unk>'])
                            for word in test_src.split()[:30]]])

    sent = model.bs_decode(src, word2id, beam_size)
    print("OUTPUT: ", sent)


if __name__ == "__main__":
    word2id = make_word2id("./vocab.pkl", 5000)
    model_path = "./ckpt/ckpt-8.378940-28e-0s"
    beam_search(model_path, word2id)
