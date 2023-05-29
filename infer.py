import torch
import pickle

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('save/6.pkl', map_location=device)
    output = open('ner_result.txt', 'w', encoding='utf-8')
    test = 'data/ner_test.txt'

    with open(test, 'r', encoding='utf-8') as f:
        line_test = []
        for test in f:
            test = test.strip()

            # 一句话读入完毕
            if not test:
                x = [model.tokenizer.cls_token_id]  # <CLS>的ID
                x.extend(model.tokenizer.convert_tokens_to_ids(line_test))
                x.append(model.tokenizer.sep_token_id)  # <SEP>的ID
                x = torch.LongTensor(x).to(device)
                mask = torch.ones_like(x, dtype=torch.uint8).to(device)
                length = [x.shape[0]]

                x = x.reshape(1, -1)
                mask = mask.reshape(1, -1)

                predict = model.infer(x, mask, length)[0]
                # 首尾去除，只考虑原文的字符
                for i in range(0, len(line_test)):
                    print(line_test[i], model.id2tag[predict[i + 1]], file=output)
                print(file=output)

                line_test.clear()
            
            else:
                test = test.split(' ')
                line_test.append(test[0])
    output.close()
