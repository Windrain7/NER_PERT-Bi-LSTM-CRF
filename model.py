import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModel

class CWS(nn.Module):

    def __init__(self, tag2id, id2tag, hidden_dim, embedding_dim=768, p=0.5):
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.tagset_size = len(tag2id)

        # self.word_embeds = AutoModel.from_pretrained('hfl/chinese-pert-base')
        # self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-pert-base')
        self.word_embeds = AutoModel.from_pretrained('PERT')
        self.tokenizer = AutoTokenizer.from_pretrained('PERT')

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=p)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, batch_first=True)

    # 双向网络通道为2
    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, sentence, mask, length):
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # idx->embedding, 输入是(batch_size, seq_len), return_dict=False返回向量
        embeds, _ = self.word_embeds(input_ids=sentence, attention_mask=mask, return_dict=False)
        embeds = pack_padded_sequence(embeds, length, batch_first=True)

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # dropout与全连接
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        emissions = self._get_lstm_features(sentence, mask, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask, length):
        emissions = self._get_lstm_features(sentence, mask, length)
        return self.crf.decode(emissions, mask)
