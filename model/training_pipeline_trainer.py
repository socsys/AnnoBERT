import ast
import numpy as np
import pandas as pd
import torch
import transformers
from annobert import AnnoBERT
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import HfArgumentParser, TrainingArguments, Trainer, BertTokenizer
from typing import Optional

transformers.logging.set_verbosity_error()

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=75,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    text_data_dir: str = field(
        default=None,
        metadata={
            "help": "Location of training data: formatted; will be further divided into train/val sets. "
        }
    )

    crowd_data_dir: str = field(
        default=None,
        metadata={
            "help": "Location of testing data: formatted; will be used as test set. "
        }
    )

    label_dict: str = field(
        default="{0:_'nonmisogynistic',_1:_'misogynistic'}",
        metadata={
            "help": "dict that maps binary int labels to label text. replace space with underscore for hf parsing. "
        }
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    anno_emb_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Annotator embedding .pt file"
        }
    )
    anno_pool: str = field(
        default='mean',
        metadata={
            "help": "mean, max, sum, None"
        }
    )
    anno_emb_dim: int = field(
        default=768,
        metadata={
            "help": "Dimensionality of annotator embeddings, only needs to be specified when not using pre-trained embeddings"
        }
    )
    anno_emb_freeze: bool = field(
        default=False,
        metadata={
            "help": "whether to freeze annotator embeddings during training. does not apply to any other layer, including subsequent dimension-reduction linear"
        }
    )
    max_anno_num: int = field(
        default=6,
        metadata={
            "help": "Number of unique annotators in the dataset, only needs to be specified when not using pre-trained embeddings"
        }
    )
    feature_extract_num_layers: int = field(
        default=6,
        metadata={
            "help": "Number of feature extraction layers after concatenation of annotator embeddings and bert embeddings"
        }
    )
    num_classes: int = field(
        default=2,
        metadata={
            "help": "Number of classes"
        }
    )


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

class AnnoDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, inputs, masks, token_type_ids, anno_matrix, label_ids):
        'Initialization'
        self.input_ids = inputs
        self.attention_mask = masks
        self.token_type_ids = token_type_ids
        self.anno_input = anno_matrix
        self.label_ids = label_ids

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_ids)

  def __getitem__(self, index):
        'Generates one sample of data'
        input = self.input_ids[index]
        mask = self.attention_mask[index]
        token_type_ids = self.token_type_ids[index]
        anno_ids = self.anno_input[index]
        label_id = self.label_ids[index]
        return {
            'input_ids': torch.tensor(input, dtype=torch.long),
            'anno_input': torch.tensor(anno_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label_id': torch.tensor(label_id, dtype=torch.long)
        }


def compute_metrics(pp):
    # pp.predictions and pp.label_ids
    preds = pp.predictions
    preds = np.argmax(preds, axis=1)
    precisions = precision_score(pp.label_ids, preds, average=None)
    recalls = recall_score(pp.label_ids, preds, average=None)
    f1s = f1_score(pp.label_ids, preds, average=None)
    return {'macro_f1': f1_score(pp.label_ids, preds, average='macro'),
            'hate_f1': f1s[1],
            'nohate_f1': f1s[0],
            'hate_precision': precisions[1],
            'hate_recall':  recalls[1],
            'nohate_precision': precisions[0],
            'nohate_recall': recalls[0],
            'accuracy': accuracy_score(pp.label_ids, preds),
            'balanced_accuracy': balanced_accuracy_score(pp.label_ids, preds)}


class CustomTrainer(Trainer):
    def compute_loss(self, model, batch, return_outputs=False):
        loss_fn = CrossEntropyLoss()
        outputs = model(batch['input_ids'], batch['anno_input'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
        loss = loss_fn(outputs["logits"], batch.get('label_id'))
        return (loss, outputs) if return_outputs else loss


def gen_train_anno_matrix(data, crowddata):
    anno_matrix = []
    for i in range(len(data)):
        row = data.iloc[i]
        anno_matrix.append([])
        crowd_subset = crowddata[(crowddata['item_id']==row['item_id']) &
            (crowddata['label']==row['join_label'])]
        anno_matrix[i].extend(list(set(crowd_subset['worker_id'])))
        anno_matrix[i].extend([0]*(6 - len(anno_matrix[i])))
    return anno_matrix


def gen_anno_matrix(data, crowddata):
    anno_matrix = []
    for i in range(len(data)):
        row = data.iloc[i]
        anno_matrix.append([])
        crowd_subset = crowddata[ (crowddata['item_id']==row['item_id']) ]
        anno_matrix[i].extend(list(set(crowd_subset['worker_id'])))
        anno_matrix[i].extend([0]*(6 - len(anno_matrix[i])))
    return anno_matrix


def duplicate_rows(data):
    data['join_label'] = data['gold']
    dup_data = data.copy(deep=True)
    for i in range(len(data)):
        row = data.iloc[i]
        if row['join_label'] == 1:
            row['join_label'] = 0
        else:
            row['join_label'] = 1
        dup_data = dup_data.append(row, ignore_index=True)
    return dup_data

def bert_tokenization(text, labels=None):
    bert_tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for i, sent in enumerate(text):
        if labels != None:
            label = labels[i]
            bert_inp = bert_tokenizer.encode_plus(sent, label, add_special_tokens=True, max_length=data_args.max_seq_length,
                pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True, truncation=True)
        else:
            bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=data_args.max_seq_length,
                pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True, truncation=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])
        token_type_ids.append(bert_inp['token_type_ids'])

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    token_type_ids = torch.tensor(token_type_ids)
    return input_ids, attention_masks, token_type_ids

def get_datasets(textdata, crowddata):
    train_data = textdata[textdata['split'] == 'train']
    train_label_ids = textdata[textdata['split'] == 'train']['gold']
    train_data, val_data, train_label_ids, val_label_ids = train_test_split(
        train_data, train_label_ids, test_size=0.2, stratify=train_label_ids)

    train_data = duplicate_rows(train_data)
    train_label_ids = train_data['gold']

    test_data = textdata[textdata['split'] == 'test']
    test_label_ids = textdata[textdata['split'] == 'test']['gold']

    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_label_ids = val_label_ids.reset_index(drop=True)
    test_label_ids = test_label_ids.reset_index(drop=True)

    train_anno_matrix = gen_train_anno_matrix(train_data, crowddata)
    val_anno_matrix = gen_anno_matrix(val_data, crowddata)
    test_anno_matrix = gen_anno_matrix(test_data, crowddata)

    label_dict = ast.literal_eval(data_args.label_dict.replace('_', ' '))

    train_labels = [label_dict[label_id] for label_id in train_data['join_label']]

    train_inp, train_mask, train_token_type_ids = bert_tokenization(train_data['text'], labels=train_labels)
    val_inp, val_mask, val_token_type_ids = bert_tokenization(val_data['text'])
    test_inp, test_mask, test_token_type_ids = bert_tokenization(test_data['text'])

    train_dataset = AnnoDataset(train_inp, train_mask, train_token_type_ids, train_anno_matrix, train_label_ids)
    val_dataset = AnnoDataset(val_inp, val_mask, val_token_type_ids, val_anno_matrix, val_label_ids)
    test_dataset = AnnoDataset(test_inp, test_mask, test_token_type_ids, test_anno_matrix, test_label_ids)
    return train_dataset, val_dataset, test_dataset

textdata = pd.read_csv(data_args.text_data_dir)
crowddata = pd.read_csv(data_args.crowd_data_dir)

train_dataset, val_dataset, test_dataset = get_datasets(textdata, crowddata)

model = AnnoBERT.from_pretrained(model_args.model_name_or_path,
                                 anno_emb_dir=model_args.anno_emb_dir,
                                 anno_pool=model_args.anno_pool,
                                 anno_emb_freeze=model_args.anno_emb_freeze,
                                 max_anno_num=model_args.max_anno_num,
                                 anno_emb_dim=model_args.anno_emb_dim,
                                 feature_extract_num_layers=model_args.feature_extract_num_layers,
                                 num_class=model_args.num_classes)

trainer = CustomTrainer(
    model=model,
    tokenizer=None,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=val_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics
)
# train

if training_args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
# eval
if training_args.do_eval:
    metrics = trainer.evaluate(eval_dataset=val_dataset)

    metrics["eval_samples"] = len(val_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if training_args.do_predict:
    predict_dataset = test_dataset
    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    metrics["predict_samples"] = len(predict_dataset)

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    predictions = np.argmax(predictions, axis=1)
    pred_df = pd.DataFrame({'pred': predictions, 'gold': labels})
    pred_df.to_csv('{}/predictions.csv'.format(training_args.output_dir), index=False)
