class Config():
    def __init__(self):
        # data
        self.filename_trimmed_train = 'data/128,32/train.pt'
        self.filename_trimmed_valid = 'data/128,32/valid.pt'
        self.filename_trimmed_test = 'data/128,32/test.pt'
        self.filename_idx2word = 'data/128,32/tgt_index2word.pkl'
        self.filename_word2idx = 'data/128,32/tgt_word2index.pkl'

        # self.filename_trimmed_train = 'data/pointer/train.pt'
        # self.filename_trimmed_valid = 'data/pointer/valid.pt'
        # self.filename_trimmed_test = 'data/pointer/test.pt'
        #
        # self.filename_idx2word = 'data/pointer/tgt_index2word.pkl'
        # self.filename_word2idx = 'data/pointer/tgt_word2index.pkl'

        self.filename_data = 'result/data/'
        self.filename_model = 'result/model/'
        self.filename_rouge = 'result/data/ROUGE.txt'
        self.src_vocab_size = 523566
        self.tgt_vocab_size = 8250
        self.pad = 0
        self.bos = 2

        self.t_len = 128 # word inputs length
        self.s_len = 32 # summary length
        self.c_len = 150 # character inputs length
        # self.c_vocab_size = 8250

        self.share_vocab = False
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.beam_size = 10

        self.batch_size = 64
        self.model_size = 512
        self.d_ff = 2048
        self.dropout = 0.1
        self.n_head = 8
        self.n_layer = 4
        self.lr = 2
        self.ls = 0.1

        self.warmup_steps = 8000
        self.accumulation_steps = 4