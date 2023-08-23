import math

import torch
from torch import nn
import torchcrf


class Embeddings(nn.Module):

    def __init__(self,
                 word_input_dim,
                 word_emb_dim,
                 word_emb_pretrained,
                 word_emb_dropout,
                 word_emb_froze,
                 use_char_emb,
                 char_input_dim,
                 char_emb_dim,
                 char_emb_pretrained,
                 char_emb_dropout,
                 char_emb_froze,
                 char_cnn_filter_num,
                 char_cnn_kernel_size,
                 char_cnn_dropout,
                 word_pad_idx,
                 char_pad_idx,
                 device
                 ):
        super().__init__()
        self.device = device
        self.word_pad_idx = word_pad_idx
        self.char_pad_idx = char_pad_idx
        # Word Embedding
        # initialize embedding with pretrained weights if given
        if word_emb_pretrained is not None:
            self.word_emb = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(word_emb_pretrained),
                padding_idx=self.word_pad_idx,
                freeze=word_emb_froze
            )
        else:
            self.word_emb = nn.Embedding(
                num_embeddings=word_input_dim,
                embedding_dim=word_emb_dim,
                padding_idx=self.word_pad_idx
            )
            self.word_emb.weight.data[self.word_pad_idx] = torch.zeros(word_emb_dim)
        self.word_emb_dropout = nn.Dropout(word_emb_dropout)
        self.output_dim = word_emb_dim
        # Char Embedding
        self.use_char_emb = use_char_emb
        if self.use_char_emb:
            self.char_emb_dim = char_emb_dim
            if char_emb_pretrained is not None:
                self.char_emb = nn.Embedding.from_pretrained(
                    embeddings=torch.as_tensor(word_emb_pretrained),
                    padding_idx=self.char_pad_idx,
                    freeze=char_emb_froze
                )
            else:
                self.char_emb = nn.Embedding(
                    num_embeddings=char_input_dim,
                    embedding_dim=char_emb_dim,
                    padding_idx=self.char_pad_idx
                )
            # initialize embedding for char padding as zero
            self.char_emb.weight.data[self.char_pad_idx] = torch.zeros(self.char_emb_dim)
            self.char_emb_dropout = nn.Dropout(char_emb_dropout)
            # Char CNN
            self.char_cnn = nn.Conv1d(
                in_channels=char_emb_dim,
                out_channels=char_emb_dim * char_cnn_filter_num,
                kernel_size=char_cnn_kernel_size,
                groups=char_emb_dim  # different 1d conv for each embedding dim
            )
            self.char_cnn_dropout = nn.Dropout(char_cnn_dropout)
            self.output_dim += char_emb_dim * char_cnn_filter_num

    def forward(self, words, chars):
        # words = [sentence length, batch size]
        # chars = [batch size, sentence length, word length)
        # tags = [sentence length, batch size]
        # embedding_out = [sentence length, batch size, embedding dim]
        embedding_out = self.word_emb_dropout(self.word_emb(words))
        if not self.use_char_emb: return embedding_out
        # character cnn layer forward
        # reference: https://github.com/achernodub/targer/blob/master/src/layers/layer_char_cnn.py
        # char_emb_out = [batch size, sentence length, word length, char emb dim]
        char_emb_out = self.char_emb_dropout(self.char_emb(chars))
        batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
        char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels, device=self.device)
        for sent_i in range(sent_len):
            # sent_char_emb = [batch size, word length, char emb dim]
            sent_char_emb = char_emb_out[:, sent_i, :, :]
            # sent_char_emb_p = [batch size, char emb dim, word length]
            sent_char_emb_p = sent_char_emb.permute(0, 2, 1)
            # char_cnn_sent_out = [batch size, out channels * char emb dim, word length - kernel size + 1]
            char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
            char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)
        char_cnn = self.char_cnn_dropout(char_cnn_max_out)
        # concat word and char embedding
        # char_cnn_p = [sentence length, batch size, char emb dim * num filter]
        char_cnn_p = char_cnn.permute(1, 0, 2)
        word_features = torch.cat((embedding_out, char_cnn_p), dim=2)
        return word_features


class LSTMAttn(nn.Module):

    def __init__(self,
                 input_dim,
                 lstm_hidden_dim,
                 lstm_layers,
                 lstm_dropout,
                 word_pad_idx,
                 attn_heads=None,
                 attn_dropout=None
                 ):
        super().__init__()
        self.word_pad_idx = word_pad_idx
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        self.attn_heads = attn_heads
        if self.attn_heads:
            self.attn = nn.MultiheadAttention(
                embed_dim=lstm_hidden_dim * 2,
                num_heads=attn_heads,
                dropout=attn_dropout
            )

    def forward(self, words, word_features):
        lstm_out, _ = self.lstm(word_features)
        if not self.attn_heads: return lstm_out
        # create masking for paddings
        key_padding_mask = torch.as_tensor(words == self.word_pad_idx).permute(1, 0)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        return attn_out


class CRF(nn.Module):

    def __init__(self,
                 input_dim,
                 fc_dropout,
                 word_pad_idx,
                 tag_names,
                 ):
        super().__init__()
        self.word_pad_idx = word_pad_idx
        # Fully-connected
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(input_dim, len(tag_names))
        # CRF
        self.crf = torchcrf.CRF(num_tags=len(tag_names))
        self.init_crf_transitions(tag_names)

    def forward(self, words, word_features, tags):
        # fc_out = [sentence length, batch size, output dim]
        fc_out = self.fc(self.fc_dropout(word_features))
        crf_mask = words != self.word_pad_idx
        crf_out = self.crf.decode(fc_out, mask=crf_mask)
        crf_loss = -self.crf(fc_out, tags=tags, mask=crf_mask) if tags is not None else None
        return crf_out, crf_loss

    def init_crf_transitions(self, tag_names, imp_value=-100):
        num_tags = len(tag_names)
        for i in range(num_tags):
            tag_name = tag_names[i]
            # I and L and <pad> impossible as a start
            if tag_name[0] in ("I", "L") or tag_name == "<pad>":
                torch.nn.init.constant_(self.crf.start_transitions[i], imp_value)
            # B and I impossible as an end
            if tag_name[0] in ("B", "I"):
                torch.nn.init.constant_(self.crf.end_transitions[i], imp_value)
        # init impossible transitions between positions
        tag_is = {}
        for tag_position in ("B", "I", "O", "U", "L"):
            tag_is[tag_position] = [i for i, tag in enumerate(tag_names) if tag[0] == tag_position]
        tag_is["P"] = [i for i, tag in enumerate(tag_names) if tag == "tag"]
        impossible_transitions_position = {
            "B": "BOUP",
            "I": "BOUP",
            "O": "IL",
            "U": "IL"
        }
        for from_tag, to_tag_list in impossible_transitions_position.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        torch.nn.init.constant_(
                            self.crf.transitions[from_tag_i, to_tag_i], imp_value
                        )
        # init impossible B and I transitions to different entity types
        impossible_transitions_tags = {
            "B": "IL",
            "I": "IL"
        }
        for from_tag, to_tag_list in impossible_transitions_tags.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        try:
                            if tag_names[from_tag_i].split("-")[1] != tag_names[to_tag_i].split("-")[1]:
                                torch.nn.init.constant_(
                                    self.crf.transitions[from_tag_i, to_tag_i], imp_value
                                )
                        except:
                            continue


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self,
                 input_dim,
                 attn_heads,
                 attn_dropout,
                 trf_layers,
                 fc_hidden,
                 word_pad_idx
                 ):
        super().__init__()
        self.word_pad_idx = word_pad_idx
        self.position_encoder = PositionalEncoding(
            d_model=input_dim
        )
        layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=attn_heads,
            activation="relu",
            dropout=attn_dropout
        )
        self.trf = nn.TransformerEncoder(
            encoder_layer=layers,
            num_layers=trf_layers
        )
        # 2-layers fully-connected with GELU activation in-between
        self.fc = nn.Linear(
            in_features=input_dim,
            out_features=fc_hidden
        )
        self.fc_gelu = nn.GELU()
        self.fc_norm = nn.LayerNorm(fc_hidden)
        self.output_dim = fc_hidden

    def forward(self, words, word_features):
        # Transformer
        key_padding_mask = torch.as_tensor(words == self.word_pad_idx).permute(1, 0)
        # pos_out = [sentence length, batch size, embedding dim + char emb dim * num filter]
        pos_out = self.position_encoder(word_features)
        # enc_out = [sentence length, batch size, embedding dim + char emb dim * num filter]
        trf_out = self.trf(pos_out, src_key_padding_mask=key_padding_mask)
        # fc_out = [sentence length, batch size, fc hidden]
        fc_out = self.fc_norm(self.fc_gelu(self.fc(trf_out)))
        return fc_out


class NERModel(nn.Module):

    def __init__(self,
                 word_input_dim,
                 word_pad_idx,
                 char_pad_idx,
                 tag_names,
                 device,
                 model_arch="bilstm",
                 word_emb_dim=50,
                 word_emb_pretrained=None,
                 word_emb_dropout=0.5,
                 word_emb_froze=False,
                 use_char_emb=False,
                 char_input_dim=None,
                 char_emb_dim=None,
                 char_emb_pretrained=None,
                 char_emb_dropout=None,
                 char_emb_froze=False,
                 char_cnn_filter_num=None,
                 char_cnn_kernel_size=None,
                 char_cnn_dropout=None,
                 lstm_hidden_dim=64,
                 lstm_layers=2,
                 lstm_dropout=0.1,
                 attn_heads=None,
                 attn_dropout=None,
                 trf_layers=None,
                 fc_hidden=None,
                 fc_dropout=0.25
                 ):
        super().__init__()
        # Embeddings
        self.embeddings = Embeddings(
            word_input_dim=word_input_dim,
            word_emb_dim=word_emb_dim,
            word_emb_pretrained=word_emb_pretrained,
            word_emb_dropout=word_emb_dropout,
            word_emb_froze=word_emb_froze,
            use_char_emb=use_char_emb,
            char_input_dim=char_input_dim,
            char_emb_dim=char_emb_dim,
            char_emb_pretrained=char_emb_pretrained,
            char_emb_dropout=char_emb_dropout,
            char_emb_froze=char_emb_froze,
            char_cnn_filter_num=char_cnn_filter_num,
            char_cnn_kernel_size=char_cnn_kernel_size,
            char_cnn_dropout=char_cnn_dropout,
            word_pad_idx=word_pad_idx,
            char_pad_idx=char_pad_idx,
            device=device
        )
        if model_arch.lower() == "bilstm":
            # LSTM-Attention
            self.encoder = LSTMAttn(
                 input_dim=self.embeddings.output_dim,
                 lstm_hidden_dim=lstm_hidden_dim,
                 lstm_layers=lstm_layers,
                 lstm_dropout=lstm_dropout,
                 word_pad_idx=word_pad_idx,
                 attn_heads=attn_heads,
                 attn_dropout=attn_dropout
            )
            encoder_output_dim = lstm_hidden_dim * 2
        elif model_arch.lower() == "transformer":
            # Transformer
            self.encoder = Transformer(
                input_dim=self.embeddings.output_dim,
                attn_heads=attn_heads,
                attn_dropout=attn_dropout,
                trf_layers=trf_layers,
                fc_hidden=fc_hidden,
                word_pad_idx=word_pad_idx
            )
            encoder_output_dim = self.encoder.output_dim
        else:
            raise ValueError("param `model_arch` must be either 'bilstm' or 'transformer'")
        # CRF
        self.crf = CRF(
            input_dim=encoder_output_dim,
            fc_dropout=fc_dropout,
            word_pad_idx=word_pad_idx,
            tag_names=tag_names
        )

    def forward(self, words, chars, tags=None):
        word_features = self.embeddings(words, chars)
        # lstm_out = [sentence length, batch size, hidden dim * 2]
        encoder_out = self.encoder(words, word_features)
        # fc_out = [sentence length, batch size, output dim]
        crf_out, crf_loss = self.crf(words, encoder_out, tags)
        return crf_out, crf_loss

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def load_state(self, path):
        self.load_state_dict(torch.load(path))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

