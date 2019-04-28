class Config():
    def __init__(self):
        # data
        self.bert = False
        self.fine_tune = True
        if self.bert:
            self.filename_trimmed_train = 'data/bert/train.pt'
            self.filename_trimmed_valid = 'data/bert/valid.pt'
            self.filename_trimmed_test = 'data/bert/test.pt'
            self.src_vocab_size = 523566
            self.tgt_vocab_size = 21128
            self.filename_data = 'result/data/bert/'
            self.filename_model = 'result/model/bert/'
            self.filename_rouge = 'result/data/bert/ROUGE.txt'
            self.pad = 0
            self.unk = 100
            self.cls = 101
            self.sep = 102
            self.mask_id = 103
            self.bos = 102  # sep
            self.eos = 105
            self.accumulation_steps = 8
        else:
            self.filename_trimmed_train = 'data/train.pt'
            self.filename_trimmed_valid = 'data/valid.pt'
            self.filename_trimmed_test = 'data/test.pt'
            self.filename_idx2word = 'data/tgt_index2word.pkl'
            self.filename_word2idx = 'data/tgt_word2index.pkl'
            self.filename_data = 'result/data/'
            self.filename_model = 'result/model/'
            self.filename_rouge = 'result/data/ROUGE.txt'
            self.src_vocab_size = 523566
            self.tgt_vocab_size = 8250
            self.pad = 0
            self.bos = 2
            self.accumulation_steps = 2

        self.t_len = 150
        self.s_len = 50

        self.share_vocab = False
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.beam_size = 10

        self.batch_size = 64
        self.model_size = 512
        self.d_ff = 2048
        self.dropout = 0.1
        self.n_head = 8
        self.n_layer = 6
        self.lr = 0.2
        self.ls = 0.1

        self.warmup_steps = 8000