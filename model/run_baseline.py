import re
import logging
import pandas as pd
import numpy as np
from itertools import product

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from transformers import BertTokenizer

from leam import LEAM
from crowdlayer import CrowdLayer, Pre_Crowdlayer, get_crowd_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class Config():
    def __init__(self,dataset,model):

        # for datasets
        if dataset == 'eacl':
            self.original_data_path = 'data/eacl/text_gold_eacl.csv'
            self.crowddata_path = 'data/eacl/annotation_eacl.csv'
            self.embedding_save_path = 'processed_data/emb.eacl.pkl'
            self.pred_save_path = 'result/eacl/eacl.'+str(model)
            self.num_classes = 2
        elif dataset == 'dynabench':
            self.original_data_path = 'data/dynabench/text_gold_dynabench.csv'
            self.crowddata_path = 'data/dynabench/annotation_dynabench.csv'
            self.embedding_save_path = 'processed_data/emb.dynabench.pkl'
            self.pred_save_path = 'result/dynabench/dynabench.'+str(model)
            self.num_classes = 2

        # for baselines
        if model == 'leam':
            self.embedding_path = 'embedding/Fasttext/cc.en.300.vec.txt'
            self.model_save_path = 'model/leam/' + dataset + '/'
            self.keep_prob = 0.5
            self.epochs = 15 #100 #50 #10
            self.vocab_size = 5000
            self.attention_size = 128
            self.hidden_dim = 128
            self.embedding_dim = 300
            self.filters = [3, 4, 5]
        else:
            self.epochs = 4

        # general config
        self.seq_length = 75
        self.batch_size = 32
        self.train_test_split_value = 0.2
        self.runs = 15


def text_preprocessing(text):
    
    reg_user = r'@[\w\d.·-]+'
    reg_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    text = text.lower()
    text = re.sub(reg_url, '<url>', text)
    text = re.sub(reg_user, '<user>', text)
    text = re.sub('[\s]?\d+\w+[\s]?|[\s]?\d+[\s]?|[\s]?\w+\d+[\s]?', ' ', text)  # remove numbers that are fully made of digits
    text = re.sub(r'([#:\"\'.,!?()_<>])', r' \1 ', text)  # pad punctuations
    text = re.sub('\s{2,}', ' ', text)  # remove extra spaces
    text = re.sub('_{2,}', '_', text)  # remove extra _

    text = text.strip()
    return text


def bert_tokenization(text):
    
    input_ids=[]
    attention_masks=[]
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for sent in text:
        bert_inp = bert_tokenizer.encode_plus(text_preprocessing(sent), 
                                              add_special_tokens=True, 
                                              max_length=75, 
                                              pad_to_max_length=True, 
                                              return_attention_mask=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    
    return input_ids, attention_masks


def load_word_vectors(vocab, fp):

    embeddings_index = {}
    with open(fp) as f:
        print("read embedding...")
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            if len(values[1:]) != 300:
                coefs = np.asarray(np.random.random(300) * -2 + 1, dtype=np.float32)
            else:
                coefs = np.asarray(values[1:], dtype=np.float32)

            embeddings_index[word] = coefs
        f.close()

    print("vovabulary size = " + str(len(vocab)))
    print("embedding size = " + str(len(embeddings_index)))
    embedding_matrix = np.zeros((len(vocab) + 1, 300))
    unk_dict = {}
    
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif word in unk_dict:
            embedding_matrix[i] = unk_dict[word]
        else:
            unk_embed = np.random.random(300) * -2 + 1
            unk_dict[word] = unk_embed
            embedding_matrix[i] = unk_dict[word]

    return embedding_matrix


def run_leam(train_text, test_text, train_y, test_y):

    print('Preprocessing ...')

    train_text_pre = train_text['text'].apply(text_preprocessing)
    test_text_pre = test_text['text'].apply(text_preprocessing)

    print('Process inputs ...')
    tokenizer = Tokenizer(num_words=config.vocab_size)
    tokenizer.fit_on_texts(train_text_pre)

    sequences = tokenizer.texts_to_sequences(train_text_pre)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=config.seq_length, dtype='int64')

    test_sequences = tokenizer.texts_to_sequences(test_text_pre)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=config.seq_length, dtype='int64')
    train_y = to_categorical(train_y)

    print('Train Data: {}'.format(sequences_matrix.shape))
    print('Train Label: {}'.format(np.array(train_y).shape))
    print('Test Data: {}'.format(test_sequences_matrix.shape))
    print('Test Label: {}'.format(np.array(test_y).shape))

    print('Generate embeddings ...')
    vocab = tokenizer.word_index
    embedding_matrix = load_word_vectors(vocab, config.embedding_path)

    runs = config.runs
    for i in range(runs):
        print('\n======== '+str(i+1)+' run ========\n')

        model = LEAM(num_classes=config.num_classes,vocab=vocab,emb_matrix=embedding_matrix)
        model.fit(sequences_matrix, train_y, 
                            batch_size=config.batch_size, 
                            epochs=config.epochs,
                            validation_split=config.train_test_split_value)
                            # callbacks=callbacks)

        save_path = config.pred_save_path+str(i+1)+'.tsv'
        evaluate(test_text['text'],test_sequences_matrix,test_y,model,save_path)


def run_crowdlayer(train_text, test_text, train_y, test_y, crowddata):

    # Preprocessing and tokenisation
    print('Preprocessing inputs...')
    train_inp, train_mask = bert_tokenization(train_text['text'])
    test_inp, test_mask = bert_tokenization(test_text['text'])
    test_y = np.array(test_y)

    # generate crowd matrix
    train_crowd_matrix = get_crowd_matrix(crowddata,train_text,train_y)

    print('Train Data: {},{}'.format(train_inp.shape, train_mask.shape))
    print('Train Label: {}'.format(np.array(train_y).shape))
    print('Train Crowd Matrix: {}'.format(np.array(train_crowd_matrix).shape))
    print('Test Data: {},{}'.format(test_inp.shape, test_mask.shape))
    print('Test Label: {}'.format(np.array(test_y).shape))

    runs = config.runs
    for i in range(runs):
        print('\n======== '+str(i+1)+' run ========\n')

        model = CrowdLayer(train_crowd_matrix)
        model.fit([train_inp, train_mask], train_crowd_matrix, 
                    batch_size=config.batch_size, 
                    epochs=config.epochs)

        pmodel = Pre_Crowdlayer(model)

        save_path = config.pred_save_path+str(i+1)+'.tsv'
        evaluate(test_text['text'],[test_inp, test_mask],test_y,pmodel,save_path)


def evaluate(test_text,test_x,test_y,model,save_path,save=True):

    probs = np.array(model.predict(test_x))
    preds = np.argmax(probs, axis=1) 

    print(confusion_matrix(test_y, preds))
    print(classification_report(test_y, preds,digits=5))

    if save:
        # write configs
        f = open(save_path,'w')
        f.write('EPOCH='+str(config.epochs)+'\n')
        f.write('BATCH_SIZE='+str(config.batch_size)+'\n')
        f.write('SEQUENCE_LENGTH='+str(config.seq_length)+'\n\n')
        f.close()

        # write classification report
        report = classification_report(test_y, preds,digits=5,output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(save_path,mode='a+',sep='\t')

        # write predictions
        f = open(save_path,'a+')
        f.write('\nText\tPrediction\tGround Truth\n')
        for i in range(len(test_text)):
            f.write(str(np.asarray(test_text)[i]))
            f.write('\t')
            f.write(str(preds[i]))
            f.write('\t')
            f.write(str(np.asarray(test_y)[i]))
            f.write('\n')
        f.close()
        

if __name__ == "__main__":

    datasets = ['eacl', 'dynabench']
    model_names = ['crowdlayer', 'leam']
    
    run_names = product(datasets, model_names)

    for run_name in run_names:
        dataset, model_name = run_name[0], run_name[1]
        config = Config(dataset,model_name)

        logging.info('--'*20)
        logging.info('Testing '+model_name+' baseline on '+dataset+' dataset ...')
        logging.info('Load Dataset ...')

        data = pd.read_csv(config.original_data_path)
        data = data.dropna(subset=['text','gold'])
        crowddata = pd.read_csv(config.crowddata_path) if model_name == 'crowdlayer' else None

        logging.info('Split Data ...')
        train_text, test_text, train_y, test_y = train_test_split(data[['text','item_id']], 
                                                                  data['gold'], 
                                                                  test_size=config.train_test_split_value,
                                                                  stratify=data['gold'], 
                                                                  random_state=42)

        logging.info('Start running ...')
        if model_name == 'leam':
            run_leam(train_text, test_text, train_y, test_y)
        elif model_name == 'crowdlayer':
            run_crowdlayer(train_text, test_text, train_y, test_y, crowddata)
