import torch
import transformers
import pickle
from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertPreTrainedModel, BertLayer, BertPooler
from torch import nn

bert_config = BertConfig()

class AnnoBERT(BertPreTrainedModel):
    def __init__(self, bert_config, **kwargs):  # num_classes, anno_emb_dir, anno_pool, feature_extract_num_layers
        super(AnnoBERT, self).__init__(bert_config)
        # need to assert kwargs are valid and have default values

        bert_config.classifier_dropout = bert_config.hidden_dropout_prob
        bert_config.update(kwargs)
        self.config = bert_config
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        if getattr(self.config, "anno_emb_dir", None) is not None:
            anno_tensor = torch.load(self.config.anno_emb_dir).to(torch.float32)
            self.config.anno_emb_dim = anno_tensor.shape[1]
            if self.config.anno_pool is not None:       # "mean", "max", "sum"
                self.anno_embeddingbag = nn.EmbeddingBag.from_pretrained(anno_tensor, freeze=self.config.anno_emb_freeze, mode=self.config.anno_pool, padding_idx=0)
                if self.config.anno_emb_dim > self.config.hidden_size:
                    self.anno_emb_reduce = nn.Linear(anno_tensor.shape[1], self.config.hidden_size)
                    self.config.anno_concat_dim = self.config.hidden_size
                else:
                    self.anno_emb_reduce = None
                    self.config.anno_concat_dim = self.config.anno_emb_dim
            else:
                self.anno_embedding = nn.Embedding.from_pretrained(anno_tensor, freeze=self.config.anno_emb_freeze, padding_idx=0)
                self.config.anno_concat_dim = self.config.anno_emb_dim * self.config.max_annotators
        else:  # not using pre-trained anno embs
            assert self.config.anno_emb_freeze is False
            if self.config.anno_pool is not None:  # "mean", "max", "sum"
                self.anno_embeddingbag = nn.EmbeddingBag(num_embeddings=self.config.max_anno_num + 1, embedding_dim=self.config.anno_emb_dim, mode=self.config.anno_pool, padding_idx=0)
                self.config.anno_concat_dim = self.config.anno_emb_dim
            else:
                self.anno_embedding = nn.Embedding(num_embeddings=self.config.max_anno_num + 1, embedding_dim=self.config.anno_emb_dim, padding_idx=0)
                self.config.anno_concat_dim = self.config.anno_emb_dim * self.config.max_annotators
            self.anno_emb_reduce = nn.Linear(self.anno_embedding.weight.shape[1], self.config.hidden_size) if self.config.anno_concat_dim > self.config.hidden_size else None
        self.reduce = nn.Linear(self.config.hidden_size + self.config.anno_concat_dim, self.config.hidden_size)
        self.feature_extract = nn.ModuleList([BertLayer(self.config) for _ in range(kwargs.get("feature_extract_num_layers", 6))])
        self.classifier = PoolerClassifier(self.config)

        self.init_weights()

    def forward(self, input_ids, anno_input,
                attention_mask=None, position_ids=None, token_type_ids=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids)
        text_intermediate = bert_outputs.last_hidden_state   # (batch_size, sequence_length, hidden_size)
        text_intermediate = self.dropout(text_intermediate)

        if self.config.anno_pool is not None:
            anno_emb = self.anno_embeddingbag(anno_input)  # (batch_size, anno_emb_dim)
        else:
            anno_emb_unflattend = self.anno_embedding(anno_input)  # (batch_size, max_anno_num, anno_emb_dim)
            anno_emb = torch.flatten(anno_emb_unflattend, start_dim=1)  # (batch_size, max_anno_num * anno_emb_dim)
        if self.anno_emb_reduce is not None:
            anno_emb = self.anno_emb_reduce(anno_emb)
        anno_emb_repeated = anno_emb.unsqueeze(1).repeat(1, text_intermediate.shape[1], 1)  # repeat along dimension 1 to be concatenated to every word in sequence
        intermediate = torch.cat([text_intermediate, anno_emb_repeated], dim=2)  # (batch_size, sequence_length, hidden_size+anno_emb_dim)
        intermediate = self.dropout(intermediate.to(torch.float32))

        intermediate_reduced = torch.tanh(self.reduce(intermediate))

        for i, l in enumerate(self.feature_extract):
            intermediate_reduced = l(intermediate_reduced)[0]

        final_features = self.dropout(intermediate_reduced)

        logits = self.classifier(final_features)   # use [cls]

        return {"logits": logits}


class PoolerClassifier(nn.Module):
    def __init__(self, bert_config):
        super(PoolerClassifier, self).__init__()
        self.pooler = BertPooler(bert_config)
        self.dropout = nn.Dropout(bert_config.classifier_dropout)
        self.linear = nn.Linear(bert_config.hidden_size, bert_config.num_labels)

    def forward(self, hidden_states):
        pooled = self.pooler(hidden_states)
        pooled = self.dropout(pooled)
        logits = self.linear(pooled)
        return logits

