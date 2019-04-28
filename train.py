import argparse
import torch
import pickle
from tqdm import tqdm
from models.transformer import Transformer
from models.autoencoder import AE
from models.model import save_model, load_model
from models.loss import LabelSmothingLoss
from models.optimizer import Optim
from utils.data import data_load, index2sentence
from utils.config import Config
from utils.rouge import rouge_score, write_rouge
from pytorch_pretrained_bert import BertTokenizer


def valid(epoch, config, model, loss_func):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # data
    valid_loader = data_load(config.filename_trimmed_valid, config.batch_size, False)
    all_loss = 0
    num = 0
    for step, batch in enumerate(tqdm(valid_loader)):
        x, y = batch
        word = y.ne(config.pad).sum().item()
        num += word
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            if config.bert:
                output, final_output, _ = model.sample(x)
                loss = loss_func(output, y) + loss_func(final_output, y)
            else:
                output, _ = model.sample(x)
                loss = loss_func(output, y)
        all_loss += loss.item()
    print('epoch:', epoch, '|valid_loss: %.4f' % (all_loss / num))
    return all_loss / num


def test(epoch, config, model, loss_func, tokenizer):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # data
    test_loader = data_load(config.filename_trimmed_test, config.batch_size, False)
    all_loss = 0
    num = 0

    r = []
    for step, batch in enumerate(tqdm(test_loader)):
        x, y = batch
        word = y.ne(config.pad).sum().item()
        num += word
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            if config.bert:
                output, final_output, out = model.sample(x)
                loss = loss_func(output, y) + loss_func(final_output, y)
            else:
                output, out = model.sample(x)
                loss = loss_func(output, y)
        all_loss += loss.item()

        # idx2word
        if config.bert:
            out = out.cpu().numpy()
            for i in range(out.shape[0]):
                t = []
                for c in list(out[i]):
                    if c == 102:
                        break
                    t.append(c)
                if len(t) == 0:
                    sen = []
                    sen.append('[UNK]')
                else:
                    if len(t) == 1:
                        t.append(config.unk)
                    sen = tokenizer.convert_ids_to_tokens(t)
                r.append(' '.join(sen))
        else:
            for i in range(out.shape[0]):
                sen = index2sentence(list(out[i]), tokenizer)
                r.append(' '.join(sen))

    print('epoch:', epoch, '|test_loss: %.4f' % (all_loss / num))
    # write result
    filename_data = config.filename_data + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # rouge
    score = rouge_score(config.filename_gold, filename_data)

    # write rouge
    write_rouge(config.filename_rouge, score, epoch)

    # print rouge
    print('epoch:', epoch, '|ROUGE-1 f: %.4f' % score['rouge-1']['f'],
          ' p: %.4f' % score['rouge-1']['p'],
          ' r: %.4f' % score['rouge-1']['r'])
    print('epoch:', epoch, '|ROUGE-2 f: %.4f' % score['rouge-2']['f'],
          ' p: %.4f' % score['rouge-2']['p'],
          ' r: %.4f' % score['rouge-2']['r'])
    print('epoch:', epoch, '|ROUGE-L f: %.4f' % score['rouge-l']['f'],
          ' p: %.4f' % score['rouge-l']['p'],
          ' r: %.4f' % score['rouge-l']['r'])

    return score['rouge-2']['f']


def train(args, config, model):
    if config.bert:
        tokenizer = BertTokenizer.from_pretrained('data/bert/vocab.txt')
    else:
        tokenizer = pickle.load(open(config.filename_idx2word, 'rb'))
    max_sorce = 0.0
    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-9)
    optim = Optim(optimizer, config)
    # KLDivLoss
    loss_func = LabelSmothingLoss(config)

    # data
    train_loader = data_load(config.filename_trimmed_train, config.batch_size, True)

    # # display the result
    # f = open('data/clean/data_char/src_index2word.pkl', 'rb')
    # idx2word = pickle.load(f)

    for e in range(args.checkpoint, args.epoch):
        model.train()
        all_loss = 0
        num = 0
        loss1 = 0
        loss2 = 0
        for step, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            word = y.ne(config.pad).sum().item()
            num += word
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            if config.bert:
                output, final_out = model(x, y)
                loss1 = loss_func(output, y)
                loss2 = loss_func(final_out, y)
                loss = loss1 + loss2
            else:
                out = model(x, y)
                loss = loss_func(out, y)
            all_loss += loss.item()
            if step % 200 == 0:
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % (loss.item()/word))
                # print('epoch:', e, '|step:', step, '|loss_1: %.4f' % (loss1.item()/word))
                # print('epoch:', e, '|step:', step, '|loss_2: %.4f' % (loss2.item()/word))

                # output = torch.nn.functional.softmax(output[-1], dim=-1)
                # output = torch.argmax(output, dim=-1)
                # final_out = torch.nn.functional.softmax(final_out[-1], dim=-1)
                # final_out = torch.argmax(final_out, dim=-1)
                # if torch.cuda.is_available():
                #     output = output.cpu().numpy()
                #     final_out = final_out.cpu().numpy()
                #     y = y[-1].cpu().numpy()
                # else:
                #     output = output.numpy()
                #     final_out = final_out.numpy()
                #     y = y[-1].numpy()
                # output = tokenizer.convert_ids_to_tokens(list(output))
                # final_out = tokenizer.convert_ids_to_tokens(list(final_out))
                # y = tokenizer.convert_ids_to_tokens(list(y))
                # print(''.join(output))
                # print(''.join(final_out))
                # print(''.join(y))

            # loss regularization
            loss = loss / config.accumulation_steps
            loss.backward()
            if ((step+1) % config.accumulation_steps) == 0:
                optim.updata()
                optim.zero_grad()

            # ###########################
            # if step == 2:
            #     break
            # ###########################

            # if step % 500 == 0:
            #     test(e, config, model, loss_func)

            if step != 0 and step % 5000 == 0:
                filename = config.filename_model + 'model_' + str(step) + '.pkl'
                save_model(model, filename)
                # test(e, config, model, loss_func)
        # train loss
        loss = all_loss / num
        print('epoch:', e, '|train_loss: %.4f' % loss)

        # test
        sorce = test(e, config, model, loss_func, tokenizer)
        if sorce > max_sorce:
            max_sorce = sorce
            filename = config.filename_model + 'model.pkl'
            save_model(model, filename)


def main():
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size for train')
    parser.add_argument('--epoch', '-e', type=int, default=50, help='number of training epochs')
    parser.add_argument('--n_layer', '-n', type=int, default=6, help='number of encoder layers')
    parser.add_argument('-seed', '-s', type=int, default=123, help="Random seed")
    parser.add_argument('--save_model', '-m', action='store_true', default=False, help="whether to save model")
    parser.add_argument('--checkpoint', '-c', type=int, default=0, help="load model")
    args = parser.parse_args()

    ########test##########
    # args.batch_size = 1
    ########test##########

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.n_layer:
        config.n_layer = args.n_layer

    # seed
    torch.manual_seed(args.seed)

    # rouge initalization
    open(config.filename_rouge, 'w')

    if config.bert:
        model = AE(config)
    else:
        model = Transformer(config)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    train(args, config, model)


if __name__ == '__main__':
    main()