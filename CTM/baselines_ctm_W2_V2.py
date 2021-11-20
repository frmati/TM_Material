#!/usr/bin/env python3
#   pip install contextualized-topic-models==2.0.1
import os
from re import match
import pickle
import logging
import csv
from contextualized_topic_models.models import ctm
import string
from skopt.space.space import Real, Categorical, Integer
from scipy import spatial
import numpy as np
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.evaluation.measures import Matches, KLDivergence, CentroidDistance, CoherenceNPMI
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis as vis
import matplotlib.pyplot as plt
import random
import torch

def evaluation2():
    ...

def evaluation_zeroshot(doc_distribution_original_language, dataset_path, topics): #must be test
    """
    doc_distribution_original_language: document-topic distribution (Matrix in form of)
    dataset_path:                        
    dataset_path:
    topics:
    """
    doc_distribution_unseen_language = evaluation(dataset_path)

    print(f'Len doc distribution original {len(doc_distribution_original_language)}')
    print(f'Type doc distribution original {type(doc_distribution_original_language)}')
    print(f'Len doc distribution unseen language {len(doc_distribution_unseen_language)}')
    print(f'Type doc distribution unseen language {type(doc_distribution_unseen_language)}')

    print(f'EVALUATION beteween Test EN and {dataset_path}')
    # Matches
    matches = Matches(doc_distribution_original_language, doc_distribution_unseen_language)
    print(f'Matches: {matches.score()}')
    logging.info(f'Dataset {dataset_path}: Matches: {matches.score()} ')

    # centroidsim
    cd = CentroidDistance(doc_distribution_original_language, doc_distribution_unseen_language, topics)
    print(f'Centroid Distance: {cd.score()}')
    logging.info(f'Dataset {dataset_path}: Centroid Distance: {cd.score()} ')

    # kl
    kl = KLDivergence(doc_distribution_original_language, doc_distribution_unseen_language)
    print(f'KL: {kl.score()}')
    logging.info(f'Dataset {dataset_path}: KL: {kl.score()} ')





#============================================================================================================
def prepare_bow_dataset(train_texts_path, store_preprocessed=True):
    # create list of articles for each document
    train_texts = [line.strip() for line in open(train_texts_path, encoding="utf-8").readlines()]

    # Preprocess TRAIN documents
    sp = WhiteSpacePreprocessing(train_texts, stopwords_language='english', vocabulary_size=2000) 
    en_preprocessed_corpus, en_unpreprocessed_corpus, en_vocab = sp.preprocess() # filters infrequent tokens and stopwords
    en_preprocessed_documents = []
    en_unpreprocessed_documents = []

    stps = set(stopwords.words("english"))

    preprocessed_docs_tmp = en_preprocessed_corpus
    preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
    preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
    preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 2 and w not in stps]) for doc in preprocessed_docs_tmp]
    
    # get vocab
    vectorizer = CountVectorizer(max_features=2000, token_pattern=r'\b[a-zA-Z]{2,}\b')
    vectorizer.fit_transform(preprocessed_docs_tmp)
    vocabulary = set(vectorizer.get_feature_names())
    preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary]) for doc in preprocessed_docs_tmp]

    en_preprocessed_documents = []
    for i, doc in enumerate(preprocessed_docs_tmp):
        if len(doc) > 0:
            en_preprocessed_documents.append(doc)
            en_unpreprocessed_documents.append(en_unpreprocessed_corpus[i]) # need to to the same as in notebook

    # save to file for eval purposes:
    logging.critical(f'LEN train_docs_preprocessed.txt must be == {len(en_preprocessed_documents)}')
    if store_preprocessed:
        with open(f'{dataset_folder}train_docs_preprocessed.txt', 'w', encoding='utf-8') as f:
            for line in en_preprocessed_documents:
                f.write(line)
                f.write('\n')

    return en_unpreprocessed_documents, en_preprocessed_documents

def plot_loss(train_losses, folder):
    
    x = [i for i in range(1, len(train_losses)+1)]

    plt.title(f'CTM Model loss during training on W2, k=25 topics')
    plt.xlabel("Epoch")
    #plt.xticks(np.arange(min(tags), max(tags) + 1, 1.0))
    plt.ylabel("Loss")
    plt.plot(x,train_losses, color='blue')
    plt.legend()
    plt.savefig(f'{folder}train_loss.png')
    #plt.show() 
    plt.close()

def train_ctm(nr_topics, num_epochs, train_texts, lang_model, nr_of_words_per_topic, dropout, batch_size, lr, momentum, viz_freq):
    #TRAIN SET PREPARATIONS and PREPROCESSING
    en_unpreprocessed_documents, en_preprocessed_documents = prepare_bow_dataset(train_texts_path=train_texts, store_preprocessed=True)
    

    # Train Zero Shot TM on unpreprocessed English TRAIN Dataset
    tp = TopicModelDataPreparation(lang_model) # multilingual version of LM 
    training_dataset_en = tp.fit(text_for_contextual=en_unpreprocessed_documents, text_for_bow=en_preprocessed_documents)

    ctm_model = ZeroShotTM(tp=tp, dataset_folder=dataset_folder, bow_size=len(tp.vocab), contextual_size=768, n_components=nr_topics, num_epochs=num_epochs, hidden_sizes=(100, 100), activation='softplus', dropout=dropout, learn_priors=True, batch_size=batch_size, lr=lr, momentum=momentum, solver='adam', reduce_on_plateau=False)
    train_losses  = ctm_model.fit(training_dataset_en, viz_freq=viz_freq)

    #plot train losses
    plot_loss(train_losses, dataset_folder)




    topics_predictions_all = ctm_model.get_thetas(training_dataset_en, n_samples=5) # get all the topic predictions, n_sample ++ is better
    
    topics = ctm_model.get_topic_lists(nr_of_words_per_topic)
    with open(f'{dataset_folder}top_words_per_topic.txt', 'w', encoding='utf-8') as f:
        for t in topics:
            f.write(str(t))
            f.write('\n')
    
    logging.info(f'Most important words per topic: {topics}')


    # Evaluation of Topic Coherence: (transform text of unpreprocessed corpus to list of articles, containing list of tokens):
    preprocessed_list = [doc.split() for doc in en_preprocessed_documents]
    
    npmi = CoherenceNPMI(topics=topics, texts=preprocessed_list) # they used preprocessed docs in the documentation.
    print(f'NPMI SCORE: {npmi.score()}')
    logging.info(f'NPMI: {npmi.score()} ')
    
    return ctm_model, topics, topics_predictions_all, training_dataset_en, tp


def evaluation(model, test_dataset, test_en_docs, n_samples):
    # n_sample how many times to sample the distribution (see the documentation)
    document_topic_distribution = model.get_thetas(test_dataset, n_samples) # get all the topic predictions ( document-topic distribution for a dataset of topics)

    #get most dominant topic per document:
    most_dominant_topic = []
    for i in range(len(test_en_docs)):
        most_dominant_topic.append(np.argmax(document_topic_distribution[i]))

    return document_topic_distribution, most_dominant_topic

def test_ctm(model, test_language, test_texts, tp):
    test_most_dominant_topic_path = f"{dataset_folder}test_{test_language}_most_dominant_topic.txt"
    test_doc_topic_distribution_path = f"{dataset_folder}test_{test_language}_document_topic_distribution.txt"

    # Evaluation on TEST Dataset
    #TEST SET PREPARATIONS
    test_docs = [line.strip() for line in open(test_texts, encoding="utf-8").readlines()]
    test_dataset = tp.transform(test_docs)

    doc_distribution_test, most_dominant_topic_test = evaluation(model, test_dataset, test_docs, n_samples=5)
    
    # save distribution and predictions
    np.savetxt(test_doc_topic_distribution_path, doc_distribution_test, delimiter=" ", newline='\n')
    np.savetxt(test_most_dominant_topic_path, most_dominant_topic_test, delimiter=" ", newline='\n') #prediction on test set
    logging.critical(f'LEN test_document_topic_distribution must be == {len(doc_distribution_test)}')
    logging.critical(f'LEN test_most_dominant_topic.csv must be == {len(most_dominant_topic_test)}')
    
    logging.info(f'Doc Distribution of test.txt {doc_distribution_test.shape}')


def get_accuracy(truth_labels, predictions):
    # calculate matches # NOT SUITABLE FOR CTM MODEL!
    
    matches = len([i for i, j in zip(truth_labels, predictions) if i == j])
    accuracy = matches/len(truth_labels)

    return accuracy, matches


def calculate_accuracy(truth_label_file, prediction_file):
    #Evaluate ACCURACY on test set in comparison to ground truth labels (MAKES NO SENSE FOR CTM)

    if truth_label_file == None:
        print(f'Accuracy: None')
        logging.info(f'Accuracy: None')

        return None
    else:
        # convert truth label file to list
        with open(truth_label_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)

        labels = [int(float(item)) for sublist in data for item in sublist]


        # convert prediction file to list
        with open(prediction_file, encoding='utf-8') as f: #newline=''
            reader = csv.reader(f)
            data = list(reader)
        
        predictions = [int(float(item)) for sublist in data for item in sublist]

        accuracy, matches = get_accuracy(labels, predictions)

        print(f'Matches: {matches}')
        print(f'Accuracy: {accuracy}')
        logging.info(f'Matches: {matches}')
        logging.info(f'Accuracy: {accuracy}')

        return accuracy


def main():
    global dataset_folder

    # SETTINGS:
    dataset_folder = 'W2_Dataset/'  
    nr_topics = 25
    nr_of_words_per_topic = 10 # 10 default
    nr_epochs = 20 
    viz_freq = 5
    lang_model = 'paraphrase-multilingual-mpnet-base-v2'# fill mask 'bert-base-multilingual-uncased'  sent 'distiluse-base-multilingual-cased-v2'
    dropout = 0.2 # 0.2 default
    batch_size = 64 # 64 default
    lr = 0.002 # 0.002 default
    momentum = 0.99 # 0.99 default
    log_file_name = f"{dataset_folder}baselines_ctm_nrTopic_{nr_topics}_{lang_model}_lr{lr}_batchSize{batch_size}.log"
    train_loc = f"{dataset_folder}train_texts.txt"
    #test_labels = f"{dataset_folder}test_labels.txt" # in case they exist, path
    test_labels = False # in case test dataset was not annotated

    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    nltk.download('stopwords')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'

    ctm_model, topics, topics_predictions_all, training_dataset, tp = train_ctm(nr_topics, nr_epochs, train_texts=train_loc, lang_model=lang_model, nr_of_words_per_topic=nr_of_words_per_topic,  dropout=dropout, batch_size=batch_size, lr=lr, momentum=momentum, viz_freq=viz_freq) # in paper: "paraphrase-multilingual-mpnet-base-v2" bert-base-multilingual-uncased

    # visualization of the trained model
    lda_vis_data = ctm_model.get_ldavis_data_format(tp.vocab, training_dataset, n_samples=5)
    visualization_tm = vis.prepare(**lda_vis_data)
    vis.save_html(visualization_tm, f'{dataset_folder}CTM_topic_overview_{dataset_folder[:-1]}_n{nr_topics}_epochs{nr_epochs}.html')
    #vis.display(visualization_tm)

    #get more infos about the topics derived during training:
    word_prop_per_topic = []
    for i in range(nr_topics):
        word_prop_per_topic.append(ctm_model.get_word_distribution_by_topic_id(i))

    with open(f'{dataset_folder}word_prop_per_topic.data', 'wb') as f: 
        pickle.dump(word_prop_per_topic, f) 

    # save model:
    ctm_model.save(models_dir=dataset_folder)

    test_ctm(ctm_model, test_language='en', test_texts= f"{dataset_folder}test_en.txt", tp=tp)
    test_ctm(ctm_model, test_language='en_translated_from_ger', test_texts= f"{dataset_folder}translated_test_ger_to_en.txt", tp=tp)
    test_ctm(ctm_model, test_language='ger', test_texts= f"{dataset_folder}test_ger.txt", tp=tp)
    test_ctm(ctm_model, test_language='fr', test_texts= f"{dataset_folder}test_fr.txt", tp=tp)
    test_ctm(ctm_model, test_language='it', test_texts= f"{dataset_folder}test_it.txt", tp=tp)
    test_ctm(ctm_model, test_language='port', test_texts= f"{dataset_folder}test_port.txt", tp=tp)

if __name__ == "__main__":
    main()
