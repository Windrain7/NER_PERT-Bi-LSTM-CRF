import datetime
import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import CWS
from dataloader import Sentence

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--cuda', action='store_true', default=True)
    return parser.parse_args()


def set_logger():
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = os.path.join('save', time + '-log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    kind, start, end = '', -1, -1
    for j in range(len(x)):
        tag = id2tag[y[j]]
        tag_kind = tag[2:len(tag)]
        tag = tag[0]
        if tag == 'B':
            kind = tag_kind
            start = cur + j
        elif tag == 'I' and start != -1 and kind == tag_kind:
            continue
        elif tag == 'E' and start != -1 and kind == tag_kind:
            end = cur + j
            entities.add((tag_kind, start, end))
            kind, start, end = '', -1, -1
        elif tag == 'S':
            entities.add((tag_kind, cur + j, cur + j))
            kind, start, end = '', -1, -1
        else:
            kind, start, end = '', -1, -1


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()

    with open('data/ner_datasave.pkl', 'rb') as inp:
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    model = CWS(tag2id, id2tag, args.hidden_dim)
    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s, device=%s' % (name, str(param.shape), str(param.requires_grad), str(param.device)))

    optimizer = AdamW(model.parameters(), lr=args.lr)

    x_train = x_train[0:1000]
    y_train = y_train[0:1000]
    x_test = x_test[0:100]
    y_test = y_test[0:100]

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=2
    )

    test_data = DataLoader(
        dataset=Sentence(x_test, y_test),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=2
    )

    best_f1, f1score = 0, 0
    for epoch in range(0, args.max_epoch):
        step = 0
        log = []
        for sentence, label, mask, length in train_data:
            if use_cuda:
                sentence = sentence.cuda()
                label = label.cuda()
                mask = mask.cuda()

            # forward
            loss = model(sentence, label, mask, length)
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        entity_predict = set()
        entity_label = set()
        with torch.no_grad():
            model.eval()
            cur = 0
            for inputs, label, mask, length in test_data:
                if use_cuda:
                    inputs = inputs.cuda()
                    label = label.cuda()
                    mask = mask.cuda()
                predict = model.infer(inputs, mask, length)

                # 构造划分的词区间
                for i in range(len(length)):
                    entity_split(inputs[i, :length[i]], predict[i], id2tag, entity_predict, cur)
                    entity_split(inputs[i, :length[i]], label[i, :length[i]], id2tag, entity_label, cur)
                    cur += length[i]

            right_predict = [i for i in entity_predict if i in entity_label]
            if len(right_predict) != 0:
                precision = float(len(right_predict)) / len(entity_predict)
                recall = float(len(right_predict)) / len(entity_label)
                f1score = (2 * precision * recall) / (precision + recall)
                logging.info("precision: %f" % precision)
                logging.info("recall: %f" % recall)
                logging.info("fscore: %f" % f1score)
            else:
                logging.info("precision: 0")
                logging.info("recall: 0")
                logging.info("fscore: 0")
            model.train()

            if f1score > best_f1:
                best_f1 = f1score
                save_path = os.path.join('save', str(epoch) + '.pkl')
                torch.save(model, save_path)
                logging.info("model has been saved in %s" % save_path)

if __name__ == '__main__':
    set_logger()
    main(get_param())
