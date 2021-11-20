#!/usr/bin/env python3

from itertools import count
import logging
from contextualized_topic_models.evaluation.measures import Matches, KLDivergence, CentroidDistance, CoherenceNPMI
from numpy import dtype
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import nltk
from nltk.corpus import stopwords
from scipy.sparse import data
import csv
import string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
from scipy import stats
import re
import matplotlib.pyplot as plt
import pickle

def remove_whitespaces(list):
    preprocessed_list = []
    for doc in list:
        new_doc =  []
        for word in doc:
            w = word.replace(' ', '')
            new_doc.append(w)
        preprocessed_list.append(new_doc)

    return preprocessed_list

def filtering_lowercasing(en_preprocessed_corpus):
    #nltk.download('stopwords')
    stps = set(stopwords.words("english"))
    preprocessed_docs_tmp = en_preprocessed_corpus
    preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
    preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
    preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 2 and w not in stps]) for doc in preprocessed_docs_tmp]
    return preprocessed_docs_tmp

def prepare_bow_dataset(train_texts_path, cat_names_loc ,store_preprocessed=True):
    # create list of articles for each document
    train_texts = [line.strip() for line in open(train_texts_path, encoding="utf-8").readlines()]
    cat_names = [line.strip() for line in open(cat_names_loc, encoding="utf-8").readlines()]
    print(f'ORIGINAL CAT NAMES: {cat_names[0]}')
    # Preprocess TRAIN documents
    sp = WhiteSpacePreprocessing(train_texts, stopwords_language='english', vocabulary_size=2000) 
    en_preprocessed_corpus, en_unpreprocessed_corpus, en_vocab = sp.preprocess() # filters infrequent tokens and stopwords
    en_preprocessed_documents = []
    en_unpreprocessed_documents = []
    
    preprocessed_docs_tmp = filtering_lowercasing(en_preprocessed_corpus)
    preprocessed_cats = filtering_lowercasing(cat_names)
    
    # get vocab of training text again:
    vectorizer = CountVectorizer(max_features=2000, token_pattern=r'\b[a-zA-Z]{2,}\b')
    vectorizer.fit_transform(preprocessed_docs_tmp)
    vocabulary = set(vectorizer.get_feature_names())
    preprocessed_cats = [' '.join([w for w in doc.split() if w in vocabulary]) for doc in preprocessed_cats]
    
    print(f'PROCESSED CAT NAMES: {preprocessed_cats[0]}')
    # save to file for eval purposes:
    if store_preprocessed:
        with open(f'{dataset_folder}cat_names_preprocessed.txt', 'w', encoding='utf-8') as f:
            for line in preprocessed_cats:
                f.write(line)
                f.write('\n')

    return preprocessed_cats

def get_data(dataset_folder, method, nr_of_words_per_topic, labels_available):
    if method == 'ctm':
        # get preprocessed trainset 
        train_texts_loc= f'{dataset_folder}train_docs_preprocessed.txt'
        train_texts_preprocessed = [line.strip() for line in open(train_texts_loc, encoding="utf-8").readlines()]
        tmp_list = [doc.split() for doc in train_texts_preprocessed] #transform to format needed
        
        preprocessed_list = remove_whitespaces(tmp_list)
        print(preprocessed_list[0])

        # get k topic words per topic from txt file
        topics_loc= f'{dataset_folder}top_words_per_topic.txt'
        topics = [line[1:-2].replace("'", "").replace(",","").split() for line in open(topics_loc, encoding="utf-8").readlines()]


        # get truth labels
        if labels_available:
            truth_label_file = f"{dataset_folder}test_labels.txt"
            with open(truth_label_file, encoding='utf-8') as f:
                reader = csv.reader(f)
                data = list(reader)
            labels = [int(float(item)) for sublist in data for item in sublist]
        else:
            labels = None

    elif method == 'lotc':
        # get preprocessed trainset 
        train_texts_loc= f'{dataset_folder}train_docs_preprocessed.txt' # use of preprocessed won't work since cat words are not preprocessed
        train_texts_preprocessed = [line.strip() for line in open(train_texts_loc, encoding="utf-8").readlines()]
        tmp_list = [doc.split() for doc in train_texts_preprocessed] #transform to format needed
        
        preprocessed_list = remove_whitespaces(tmp_list)
        print(preprocessed_list[0])

        # get k topic words per topic 
        topics_loc= f'{dataset_folder}cat_vocabulary.txt'
        # apply same preprocessing as during training of CTM:
        topics = prepare_bow_dataset(train_texts_loc, topics_loc, store_preprocessed=True)
        topics = [el.split() for el in topics]

        topics = remove_whitespaces(topics)

        print(f'TOPICS: {topics}')
        
        # get truth labels
        if labels_available:
            truth_label_file = f"{dataset_folder}test_labels.txt"
            with open(truth_label_file, encoding='utf-8') as f:
                reader = csv.reader(f)
                data = list(reader)

            labels = [int(float(item)) for sublist in data for item in sublist]
            print(labels[:7])
        else:
            labels = None


    # keep only first k words defined in 'nr_of_words_per_topic'
    topics = [cat[:nr_of_words_per_topic] for cat in topics]
    
    return preprocessed_list, topics, labels
    

def get_testdoc_topic_dist(dataset_folder, method, language, nr_topics):
    if method == 'ctm':
        # get testdocument-topic-matrix (distributions of topics per doc. on testset): 
        test_doc_distribution_path = f"{dataset_folder}test_{language}_document_topic_distribution.txt"
        t_d_list_prov = [line for line in open(test_doc_distribution_path, encoding="utf-8").readlines()]

        print(t_d_list_prov[0])
        
        t_d_list = []
        for doc in t_d_list_prov:
            mat_row = doc.split()
            mat_row = [float(x) for x in mat_row]
            t_d_list.append(mat_row)
               
        # convert list of list of floats to np array:
        t_d_array = np.array(t_d_list, dtype=np.float32)
        print(t_d_array.shape)

        # get predictions on the test set
        # convert prediction file to list
        prediction_file = f"{dataset_folder}test_{language}_most_dominant_topic.txt"
        with open(prediction_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
        predictions = [int(float(item)) for sublist in data for item in sublist]
        print(f'PREDICTIONS: {predictions[:7]}')


    elif method == 'lotc':        
        t_d_list = [] 

        # get test document topic distributions:
        test_doc_distribution_path = f"{dataset_folder}{language}_test_topic_predictions.txt"
        t_d_list_prov = [line for line in open(test_doc_distribution_path, encoding="utf-8").readlines()]

        print(t_d_list_prov[0])
        len_topic = math.ceil(nr_topics/6)
        print(f'LEN TOPIC: {len_topic}')
        t_d_list = []
        for doc in t_d_list_prov:
            mat_row = doc.split()
            mat_row = [float(x) for x in mat_row]
            t_d_list.append(mat_row)
        
        # convert list of list of floats to np array:
        t_d_array = np.array(t_d_list, dtype=np.float32)
        print(t_d_array.shape)
        

        # get predictions on the test set
        # convert prediction file to list
        prediction_file = f"{dataset_folder}{language}_out.txt"
        predictions = [int(line) for line in open(prediction_file, encoding="utf-8").readlines()]
        
        #int(float(line.split())) 
        print(predictions[:7])

    return t_d_array, predictions


# on trainset calculate NPMI Coherence:
def calc_npmi(topics, topk, preprocessed_list):
    print(f'------------------------EVALUATION OF DATASET: {dataset_folder}------------------------------------')
    logging.info(f'------------------------EVALUATION OF DATASET: {dataset_folder}------------------------------------')
    
    
    npmi = CoherenceNPMI(topics=topics, texts=preprocessed_list) # they used preprocessed docs in the documentation.
    npmi_score = npmi.score(topk=topk)
    print(f'NPMI SCORE: {npmi_score} (using {topk} words per topic)')
    logging.info(f'NPMI: {npmi_score} (using {topk} words per topic)')

    return npmi_score
# on test set calculate:
def calc_accuracy(truth_labels, predictions):
    # calculate matches # NOT SUITABLE FOR CTM MODEL!
    
    matches = len([i for i, j in zip(truth_labels, predictions) if i == j])
    accuracy = matches/len(truth_labels)

    print(f'Matches: {matches} of {len(truth_labels)}')
    print(f'Accuracy: {accuracy}')
    logging.info(f'Matches: {matches}')
    logging.info(f'Accuracy: {accuracy}')

def calc_centroid_sim(t_d_dist_lang1, t_d_dist_lang2, topics):
    # centroidsim
    cd = CentroidDistance(t_d_dist_lang1, t_d_dist_lang2, topics)
    cd_score = cd.score()
    print(f'Centroid Distance: {cd_score}')
    logging.info(f'Centroid Distance: {cd_score}')
    
    return cd_score

def calc_kl_divergence(lang, t_d_dist_lang1, t_d_dist_lang2):
    kl = KLDivergence(t_d_dist_lang1, t_d_dist_lang2)
    kl_score, kl_list = kl.score()
    print(f'Average KL Divergence: {kl_score}')
    logging.info(f'Average KL Divergence: {kl_score}')

    # store kl_list as file:
    with open(f'{dataset_folder}kl_list_{lang}.data', 'wb') as f:
        pickle.dump(kl_list, f)

def calc_test_correl(list1, list2): #not useful
    correlation, p_value = stats.pearsonr(list1, list2)
    logging.info(f'CORRELATION: {correlation}, with p-Value of: {p_value}')
    print(f'CORRELATION: {correlation}, with p-Value of: {p_value}')

def calc_matching_predictions(predictions1, predictions2):
    matches = len([i for i, j in zip(predictions1, predictions2) if i == j])
    
    print(f'MATCHES: {matches} of {len(predictions1)}. In %: {matches/len(predictions1)}')
    logging.info(f'MATCHES: {matches} of {len(predictions1)}')

def transfer_eval(t_d_arrays, predictions, topics, translations_available):
    # Transfer metrics, multilingual evaluation
    if translations_available:
        print(f'\n Comparing EN with EN translation from GER:')
        logging.info(f'Comparing EN with EN translation from GER:')
        calc_centroid_sim(t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['en_translated_from_ger'], topics=topics)
        calc_kl_divergence(lang='EN_ENtrans', t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['en_translated_from_ger'])
        calc_matching_predictions(predictions1=predictions['en'], predictions2=predictions['en_translated_from_ger'])

        print(f'\n Comparing GER with EN translation from GER:')
        logging.info(f'Comparing GER with EN translation from GER:')
        calc_centroid_sim(t_d_dist_lang1=t_d_arrays['ger'] , t_d_dist_lang2=t_d_arrays['en_translated_from_ger'], topics=topics)
        calc_kl_divergence(lang='GER_ENtrans', t_d_dist_lang1=t_d_arrays['ger'] , t_d_dist_lang2=t_d_arrays['en_translated_from_ger'])
        calc_matching_predictions(predictions1=predictions['ger'], predictions2=predictions['en_translated_from_ger'])
    
    print(f'\n Comparing EN with PORT:')
    logging.info(f'Comparing EN with PORT:')
    calc_centroid_sim(t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['port'], topics=topics)
    calc_kl_divergence(lang='EN_PORT', t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['port'])
    calc_matching_predictions(predictions1=predictions['en'], predictions2=predictions['port'])

    print(f'\n Comparing EN with GER:')
    logging.info(f'Comparing EN with GER:')
    calc_centroid_sim(t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['ger'], topics=topics)
    calc_kl_divergence(lang='EN_GER', t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['ger'])
    calc_matching_predictions(predictions1=predictions['en'], predictions2=predictions['ger'])

    print(f'\n Comparing EN with IT:')
    logging.info(f'Comparing EN with IT:')
    calc_centroid_sim(t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['it'], topics=topics)
    calc_kl_divergence(lang='EN_IT', t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['it'])
    calc_matching_predictions(predictions1=predictions['en'], predictions2=predictions['it'])

    print(f'\n Comparing EN with FR:')
    logging.info(f'Comparing EN with FR:')
    calc_centroid_sim(t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['fr'], topics=topics)
    calc_kl_divergence(lang='EN_FR', t_d_dist_lang1=t_d_arrays['en'] , t_d_dist_lang2=t_d_arrays['fr'])
    calc_matching_predictions(predictions1=predictions['en'], predictions2=predictions['fr'])

def plot_loss(log_path, folder):
    # working directory is parent folder of archiv
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    losses = []
    losses = re.findall(r"Train Loss: (\d*.\d*)", content)
    losses = [float(x) for x in losses]

    print(losses)
    print(len(losses))

    x = [i for i in range(1, len(losses)+1)]

    plt.title(f'CTM Model loss during training on W2, k=25 topics')
    plt.xlabel("Epoch")
    #plt.xticks(np.arange(min(tags), max(tags) + 1, 1.0))
    plt.ylabel("Loss")
    plt.plot(x,losses, color='blue')
    plt.legend()
    plt.savefig(f'{folder}train_loss.png')
    #plt.show() 
    plt.close()

def eval_npmi(preprocessed_train_doc_list, topics, labels, eval_mode=False):
    if eval_mode:
        npmi_results = {}
        for i in range(2, 21):
            npmi_res = calc_npmi(topics=topics, topk=i , preprocessed_list=preprocessed_train_doc_list) # on the test set
            npmi_results[i] =  npmi_res

        print(npmi_results)
        logging.info(f'NPMI RESULTS: {npmi_results}')

        # plot npmi scores
        x = list(npmi_results.keys())
        y = list(npmi_results.values())

        plt.title(f'NPMI scores on train set, CTM Model, n=25')
        plt.xlabel("top-k words per topic")
        #plt.xticks(np.arange(min(tags), max(tags) + 1, 1.0))
        plt.ylabel("NPMI score")
        plt.plot(x,y, color='blue')
        plt.legend()
        plt.savefig(f'{dataset_folder}npmi_scores.png')
        #plt.show() 
        plt.close()
    else:
        npmi_res = calc_npmi(topics=topics, topk=10 , preprocessed_list=preprocessed_train_doc_list) # as in paper implementation


def main():   
    global dataset_folder

    # SETTINGS:
    dataset_folder = 'archiv/LOTClass_W2_n25_1label_mBertBaseCased/'
    logging.basicConfig(filename=f'{dataset_folder}evaluate_files.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S') 
    #logfile = "out_CTM_W2_V2_n25_ep20.txt" #console outfile needed for plotting of train loss of LOTClass
    logfile = False #if no log file available
    nr_topics = 25
    labels_available = False
    method = 'lotc'
    lingu_transfer = True
    translations_available = True
    nr_of_words_per_topic = 20





    # get data from train set
    preprocessed_train_doc_list, topics, labels = get_data(dataset_folder, method, nr_of_words_per_topic=nr_of_words_per_topic, labels_available=labels_available)

    # calculate monolingual evaluation measures for various numbers of top words per topic (topk) if eval_mode == True
    eval_npmi(preprocessed_train_doc_list, topics, labels, eval_mode=False)

    # Multilingual Evaluation:
    predictions = {}
    t_d_arrays = {}
    if lingu_transfer:
        if translations_available:
            # get predictions on all test sets
            languages = ['en', 'ger', 'it', 'fr', 'port', 'en_translated_from_ger']
        else:
            languages = ['en', 'ger', 'it', 'fr', 'port']
        
        for lang in languages:
            t_d_arrays[lang], predictions[lang]  = get_testdoc_topic_dist(dataset_folder, method, language=lang, nr_topics=nr_topics)
    else:
        t_d_arrays['en'], predictions['en']  = get_testdoc_topic_dist(dataset_folder, method, language='en', nr_topics=nr_topics)


    # monolingual evaluation
    if labels_available:
        if method == 'lotc':
            calc_accuracy(truth_labels=labels, predictions=predictions['en'])
        elif method == 'ctm':
            calc_test_correl(list1=labels, list2=predictions['en']) 

    # multilingual evaluation
    if lingu_transfer:
        transfer_eval(t_d_arrays, predictions, topics, translations_available)

    if logfile:
        plot_loss(f'{dataset_folder}{logfile}', dataset_folder)




if __name__ == "__main__":
    main()