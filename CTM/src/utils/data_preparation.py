import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.sparse
import warnings
from contextualized_topic_models.datasets.dataset import CTMDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
import torch
from tqdm import tqdm
from tqdm.autonotebook import trange
from sentence_transformers.util import batch_to_device

def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(text_file, sbert_model_to_load, batch_size=200):
    """
    Creates SBERT Embeddings from an input file
    """
    model = SentenceTransformer(sbert_model_to_load)
    with open(text_file, encoding="utf-8") as filino:
        train_text = list(map(lambda x: x, filino.readlines()))

    return np.array(model.encode(train_text, show_progress_bar=True, batch_size=batch_size))

def bert_encode(model, tokenizer, sent):
    """
    mBERT base uncased embeddings
    """
    with torch.no_grad():
        input_ids = tokenizer.encode(sent, add_special_tokens=True, truncation=True)
        outputs = model(input_ids)
        return outputs[0] # extracts [CLS] token of output -> last hidden state


def encode_doc(tokenized_doc, model, max_len):
    padded = torch.tensor(np.array([tokenized_doc + [0]*(max_len-len(tokenized_doc))]))

    # attention mask
    attention_mask = np.where(padded != 0, 1, 0)

    # NP to tensor
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    # forward pass to derive embeddings
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    # get output corresponding to input CLS token, represents whole sentence
    doc_features = output[0][:,0,:][0]

    return doc_features

def mean_pooling(model_output, attention_mask):
    '''
    Mean Pooling - Take attention mask into account for correct averaging
    Token embeddings are averaged
    '''
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_bert(sentences, batch_size=128, device='cuda',
               show_progress_bar: bool = True,
               output_value: str = 'token_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               normalize_embeddings: bool = False):
    
    # Load model from HuggingFace Hub
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    mBERT_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    model = SentenceTransformer()

    mBERT_model.eval()
    mBERT_model.to(device)

    if convert_to_tensor:
        convert_to_numpy = False

    if output_value == 'token_embeddings':
        convert_to_tensor = False
        convert_to_numpy = False

    input_was_string = False
    if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
        sentences = [sentences]
        input_was_string = True

    if device is None:
        device = mBERT_model._target_device

    all_embeddings = []
    print('Sort Sentences ...')
    length_sorted_idx = np.argsort([-model._text_length(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    

    
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
    #mBERT_model = AutoModel.from_pretrained('bert-base-multilingual-uncased')

    '''
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = mBERT_model(**encoded_input)
    '''


    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
        sentences_batch = sentences_sorted[start_index:start_index+batch_size]
        features = tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt')
        features = batch_to_device(features, device)
        


        with torch.no_grad():
            out_features = mBERT_model(**features)
            #print(f'OUTFEATURES: {out_features}')
            '''
            if output_value == 'token_embeddings':
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                    last_mask_id = len(attention)-1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0:last_mask_id+1])
            else:   #Sentence embeddings
                embeddings = out_features[output_value]
                embeddings = embeddings.detach()


                if convert_to_numpy:
                    embeddings = embeddings.cpu()
            '''
            embeddings = out_features['pooler_output'] # CLS token output
            embeddings = embeddings.detach()


            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    if convert_to_tensor:
        all_embeddings = torch.stack(all_embeddings)
    elif convert_to_numpy:
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

    if input_was_string:
        all_embeddings = all_embeddings[0]

    return all_embeddings, features


def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=128): #TODO here 
    """
    Creates SBERT Embeddings from a list
    and create same functionality for mBERT base uncased embeddings
    """

    if sbert_model_to_load == 'bert-base-multilingual-uncased': #use only pooled output -> for sentence representation
        model_output, encoded_input = encode_bert(texts, batch_size=batch_size, output_value = 'token_embedding', convert_to_numpy=True)
        
        try:
            print(f'Model CLS outputs: {model_output.size}')
        except:
            print(f'Model CLS outputs: {model_output.shape}') #incase of numpy
        

        return model_output

        
        
    else:
        model = SentenceTransformer(sbert_model_to_load)
        return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


class TopicModelDataPreparation:

    def __init__(self, contextualized_model=None):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None
        self.label_encoder = None

    def load(self, contextualized_embeddings, bow_embeddings, id2token, labels=None):
        return CTMDataset(contextualized_embeddings, bow_embeddings, id2token, labels)

    def fit(self, text_for_contextual, text_for_bow, labels=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param labels: list of labels associated with each document (optional).

        """

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model, batch_size=128)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(train_contextualized_embeddings, train_bow_embeddings, self.id2token, encoded_labels)

    def transform(self, text_for_contextual, text_for_bow=None, labels=None):
        """
        This methods create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM
        """

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                          "are using ZeroShotTM in a cross-lingual setting")

            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 1)))
        test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)

        if labels:
            encoded_labels = self.label_encoder.transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(test_contextualized_embeddings, test_bow_embeddings, self.id2token, encoded_labels)
