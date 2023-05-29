import pickle
from transformers import AutoTokenizer

INPUT_DATA = "train.txt"
TRAIN_DATA = "ner_train.txt"
VALID_DATA = "ner_valid.txt"
SAVE_PATH = "./ner_datasave.pkl"

# create id2tag
unique = set()
with open('ner_train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            unique.update([line.strip('\n').split(' ')[1]])
        except:
            pass
id2tag = list(unique)
print(id2tag)
tag2id = {}
for i, label in enumerate(id2tag):
    tag2id[label] = i

tokenizer = AutoTokenizer.from_pretrained('../PERT')
# tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-pert-base')

def handle_data():
    '''
    处理数据，并保存至savepath
    :return:
    '''
    outp = open(SAVE_PATH, 'wb')

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    with open(TRAIN_DATA, 'r', encoding="utf-8") as ifp:
        line_x = []
        line_y = []
        for line in ifp:
            line = line.strip()
            if not line:
                # bert 最长512
                line_x = line_x[0:min(512 - 2, len(line_x))]
                line_y = line_y[0:min(512 - 2, len(line_y))]
                # <CLS> 和 <SEP> 的ID与label
                line_x = [tokenizer.cls_token_id] + line_x + [tokenizer.sep_token_id]
                line_y = [tag2id['O']] + line_y + [tag2id['O']]
                x_train.append(line_x)
                y_train.append(line_y)

                line_x = []
                line_y = []
                continue
            line = line.split(' ')
            line_x.append(tokenizer.convert_tokens_to_ids(line[0]))
            line_y.append(tag2id[line[1]])

    with open(VALID_DATA, 'r', encoding="utf-8") as ifp:
        line_x = []
        line_y = []
        for line in ifp:
            line = line.strip()
            if not line:
                # bert 最长512
                line_x = line_x[0:min(512 - 2, len(line_x))]
                line_y = line_y[0:min(512 - 2, len(line_y))]
                # <CLS> 和 <SEP> 的ID与label
                line_x = [tokenizer.cls_token_id] + line_x + [tokenizer.sep_token_id]
                line_y = [tag2id['O']] + line_y + [tag2id['O']]
                x_valid.append(line_x)
                y_valid.append(line_y)

                line_x = []
                line_y = []
                continue
            line = line.split(' ')
            line_x.append(tokenizer.convert_tokens_to_ids(line[0]))
            line_y.append(tag2id[line[1]])



    print(x_train[0])
    print([tokenizer.convert_ids_to_tokens(i) for i in x_train[0]])
    print(y_train[0])
    print([id2tag[i] for i in y_train[0]])

    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    pickle.dump(x_train, outp)
    pickle.dump(y_train, outp)
    pickle.dump(x_valid, outp)
    pickle.dump(y_valid, outp)

    outp.close()


if __name__ == "__main__":
    handle_data()
