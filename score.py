test_path = 'data/ner_test.txt'
predict_path = 'ner_result.txt'
entity_test = set()
entity_predict = set()

def entity_spilt(path, entities):
    with open(path, 'r', encoding='utf-8') as f:
        kind, start = '', -1
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            tag = line.split()[1]
            tag_kind = tag[2:len(tag)]
            tag = tag[0]
            if tag == 'B':
                start = i
                kind = tag_kind
            elif tag == 'I' and start != -1 and kind == tag_kind:
                continue
            elif tag == 'E' and start != -1 and kind == tag_kind:
                entities.add((tag_kind, start, i))
                kind, start = '', -1
            elif tag == 'S':
                entities.add((tag_kind, i, i))
                kind, start = '', -1
            else:
                kind, start = '', -1


entity_spilt(test_path, entity_test)
entity_spilt(predict_path, entity_predict)
right_predict = [i for i in entity_predict if i in entity_test]

precision = float(len(right_predict)) / len(entity_predict)
recall = float(len(right_predict)) / len(entity_test)
f1score = 0 if len(right_predict) == 0 else (2 * precision * recall) / (precision + recall)
print('test:%s\tpredict:%s' % (test_path, predict_path))
print("precision: %f" % precision)
print("recall: %f" % recall)
print("fscore: %f" % f1score)
