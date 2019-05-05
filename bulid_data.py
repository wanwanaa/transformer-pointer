import torch
import pickle
import numpy as np
import torch.utils.data as data_util


def get_dataset(filename):
    datasets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = list(line)
            datasets.append(line)
    return datasets


def get_trimmed_datasets(datasets, word2idx, max_length):
    data = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        sen = np.zeros(max_length, dtype=np.int32)
        for i in range(max_length):
            if i == len(line):
                sen[i] = word2idx['<eos>']
                break
            else:
                flag = word2idx.get(line[i])
                if flag is None:
                    sen[i] = word2idx['<unk>']
                else:
                    sen[i] = word2idx[line[i]]
        data[k] = sen
        k += 1
    data = torch.from_numpy(data).type(torch.LongTensor)
    return data


def save_data(data, filename, filename_trimmed):
    t = torch.load(filename)
    src = t[:][0]
    tgt = t[:][1]
    data = data_util.TensorDataset(src, data, tgt)
    print('data save at ', filename_trimmed)
    torch.save(data, filename_trimmed)


def main():
    filename_train = 'data/raw_data/train.source'
    filename_valid = 'data/raw_data/valid.source'
    filename_test = 'data/raw_data/test.source'

    filename_train_word = 'data/128,32/train.pt'
    filename_valid_word = 'data/128,32/valid.pt'
    filename_test_word = 'data/128,32/test.pt'

    filename_trimmed_train = 'data/char/train.pt'
    filename_trimmed_valid = 'data/char/valid.pt'
    filename_trimmed_test = 'data/char/test.pt'

    filename_word2idx = 'data/128,32/tgt_word2index.pkl'

    f = open(filename_word2idx, 'rb')
    word2idx = pickle.load(f)

    # train
    datasets = get_dataset(filename_train)
    data = get_trimmed_datasets(datasets, word2idx, 150)
    save_data(data, filename_train_word, filename_trimmed_train)

    # # valid
    # datasets = get_dataset(filename_valid)
    # data = get_trimmed_datasets(datasets, word2idx, 150)
    # save_data(data, filename_valid_word, filename_trimmed_valid)

    # # test
    # datasets = get_dataset(filename_test)
    # data = get_trimmed_datasets(datasets, word2idx, 150)
    # save_data(data, filename_test_word, filename_trimmed_test)


# convert idx to words, if idx <bos> is stop, return sentence
def index2sentence(index, idx2word):
    sen = []
    for i in range(len(index)):
        if idx2word[index[i]] == '<eos>':
            break
        if idx2word[index[i]] == '<bos>':
            continue
        else:
            sen.append(idx2word[index[i]])
    if len(sen) == 0:
        sen.append('<unk>')
    return sen


def test():
    filename_trimmed_test = 'data/char/test.pt'
    test = torch.load(filename_trimmed_test)
    filename_word2idx = 'data/128,32/tgt_index2word.pkl'
    f = open(filename_word2idx, 'rb')
    word2idx = pickle.load(f)
    sen = index2sentence(np.array(test[0][1]), word2idx)
    print(sen)

    filename_word2idx = 'data/128,32/src_index2word.pkl'
    f = open(filename_word2idx, 'rb')
    word2idx = pickle.load(f)
    sen = index2sentence(np.array(test[0][0]), word2idx)
    print(sen)

    filename_word2idx = 'data/128,32/tgt_index2word.pkl'
    f = open(filename_word2idx, 'rb')
    word2idx = pickle.load(f)
    sen = index2sentence(np.array(test[0][2]), word2idx)
    print(sen)


if __name__ == '__main__':
    main()
    # test()