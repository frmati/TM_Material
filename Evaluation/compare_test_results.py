#!/usr/bin/env python3

import pandas as pd
import csv
from contextualized_topic_models.evaluation.measures import KLDivergence, kl_div
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from bokeh.plotting import figure, output_file, save
from bokeh.models import Label, ColumnDataSource, Circle, ColorBar, HoverTool
from bokeh.io import output_notebook, show
import bokeh.palettes as bp
from bokeh.transform import linear_cmap

def read_prediction_file(path):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
    predictions = [int(float(item)) for sublist in data for item in sublist]
    return predictions

def create_match_list(l1, l2):
    matches = []
    for i, j in zip(l1,l2):
        if i == j:
            matches.append(True)
        else: 
            matches.append(False)
    return matches

def decode_label_id(id2label_mapping, ids):
    labels = []
    for i in ids:
        labels.append(id2label_mapping[i])

    return labels

# decode category id
def mapping_def():
    keys = [i for i in range(nr_topics)]
    id2label_mapping=  {key: None for key in keys}
    id2label_mapping[0] = 'geographic, landscapes'
    id2label_mapping[1] = 'distance relations, measures, directions'
    id2label_mapping[2] = 'buildings, history'
    id2label_mapping[3] = 'american university, college'
    id2label_mapping[4] = 'olympics, sports'
    id2label_mapping[5] = 'movies, cinema'
    id2label_mapping[6] = 'military'
    id2label_mapping[7] = 'sports, cricket, etc.'
    id2label_mapping[8] = 'animals, nature, tropics'
    id2label_mapping[9] = 'arts, creativity'
    id2label_mapping[10] = 'USA'
    id2label_mapping[11] = 'digital, IT'
    id2label_mapping[12] = 'News, TV'
    id2label_mapping[13] = 'car racing, tennis, championships'
    id2label_mapping[14] = 'politics'
    id2label_mapping[15] = 'colours'
    id2label_mapping[16] = 'academics, research, medical'
    id2label_mapping[17] = 'historic figures, USA'
    id2label_mapping[18] = 'team championships, tournaments'
    id2label_mapping[19] = 'TV, books'
    id2label_mapping[20] = 'music, bands'
    id2label_mapping[21] = 'medieval, aristocracy'
    id2label_mapping[22] = 'politivs, law'
    id2label_mapping[23] = 'railways'
    id2label_mapping[24] = 'villages, province'

    return id2label_mapping

def t_d_matrix(language):
    # get testdocument-topic-matrix (distributions of topics per doc. on testset): 
    test_doc_distribution_path = f"{ctm_dataset_folder}test_{language}_document_topic_distribution.txt"
    t_d_list_prov = [line for line in open(test_doc_distribution_path, encoding="utf-8").readlines()]

    print(t_d_list_prov[0])
    
    t_d_list = []
    for doc in t_d_list_prov:
        mat_row = doc.split()
        mat_row = [float(x) for x in mat_row]
        t_d_list.append(mat_row)
            
    # convert list of list of floats to np array:
    t_d_array = np.array(t_d_list, dtype=np.float32)

    return t_d_array

def diff_of_top2_predictions(t_d_dist):
    print(t_d_dist.shape) # docs x topics
    print(len(t_d_dist))
    
    top_topic_ids = []
    top2_topic_id = []
    top1_weight = []
    top2_weight = [] # not returned

    for i in range(len(t_d_dist)): # for each document (row):
        indices = (-t_d_dist[i]).argsort()[:2] #gets indices of the two largest values
        top_topic_ids.append(indices)
        top1_weight.append(t_d_dist[i][indices[0]])
        top2_weight.append(t_d_dist[i][indices[1]])


    
    print(f'TOP TOPIC IDs 0{top_topic_ids[0]}')
    print(top_topic_ids[0][0])
    print(top_topic_ids[0][1])
    print(top1_weight[0])
    print(top2_weight[0])


    # Calculate abs. diff between the weights of the top2 topic predictions on one test article
    weight_diffs = []
    
    for i in range(len(t_d_dist)):
        weight_diffs.append(top1_weight[i] - top2_weight[i])    
    
    return top1_weight, weight_diffs


def cluster_test_docs(doc_topic_distribution, predicted_topic, testtexts, lang='en', out_path=None):
    # visualize test documents and their topics:
    topic_weights = doc_topic_distribution

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_red = tsne_model.fit_transform(topic_weights)

    # Plot the Topic Clusters using Bokeh
    if out_path == None:
        output_file(filename=f"{ctm_dataset_folder}test_clusters_{lang}.html", title=f"Cluster Results on {lang} Testset")
    else:
        output_file(filename=f"{out_path}test_clusters_{lang}.html", title=f"Cluster Results on {lang} Testset")
    source = ColumnDataSource(data = {"x" : tsne_red[:,0], "y" : tsne_red[:,1], "label" : predicted_topic, "short":  [i[:300] for i in testtexts]})
    mapper = linear_cmap("label", low=min(predicted_topic), high=max(predicted_topic), palette=bp.turbo(len(set(predicted_topic))))

    p = figure(plot_width=1000, plot_height=800, title="t-SNE Clustering of W2 Test-Topics")
    p.scatter(y="y", x="x", marker="circle", size=5, fill_color=mapper, source=source, alpha=.8)

    hover = HoverTool(tooltips=[("Label", "@label"), ("short", "@short")])
    p.add_tools(hover)
    color_bar = ColorBar(title="Category", color_mapper=mapper["transform"], label_standoff=12, location=(0,0))
    p.add_layout(color_bar, "right")
    save(p)



def cluster_lotclass_predictions(lotclass_dataset_folder, language):
    # get testdocument-topic-matrix (distributions of topics per doc. on testset): 
    test_doc_distribution_path = f"{lotclass_dataset_folder}{language}_test_topic_predictions.txt"
    t_d_list_prov = [line for line in open(test_doc_distribution_path, encoding="utf-8").readlines()]
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
    prediction_file = f"{lotclass_dataset_folder}{language}_out.txt"
    predictions = [int(line) for line in open(prediction_file, encoding="utf-8").readlines()]

    # get testtexts
    testtext= [line.strip() for line in open(f'{lotclass_dataset_folder}test_{language}.txt', encoding="utf-8").readlines()]
    
    #cluster test documents 
    cluster_test_docs(doc_topic_distribution=t_d_array, predicted_topic=predictions, testtexts=testtext, lang=language, out_path=lotclass_dataset_folder)





def main():
    global ctm_dataset_folder
    global lotclass_dataset_folder
    global lang
        


    #SETTINGS:
    ctm_dataset_folder = 'archiv/CTM_W2_20e_lr0.0001_paraBERT/' #for the CTM model
    lotclass_dataset_folder = 'archiv/LOTClass_W2_n25_mBERT(bert-base-multilingual-uncased)_smallBatches_ALL_LANGUAGES/' # fot lotclass
    lang = 'EN_GER' # pair of languages to compare
    language1 = 'en' # 1st language of pair
    language2 = 'ger'# 2nd language of pair
    nr_topics = 25
    lotclass_cluster_only = False # true to visualize only the LOTClass test predictions and skip rest.








    if lotclass_cluster_only:
        #cluster_lotclass_predictions(lotclass_dataset_folder, language='ger')
        cluster_lotclass_predictions(lotclass_dataset_folder, language='en')

    else:
        # word -> weight in the topic
        with open(f'{ctm_dataset_folder}word_prop_per_topic.data', 'rb') as f: 
            word_prop_per_t = pickle.load(f)

        top20_word_prob = [t[:20] for t in word_prop_per_t]

        probas = []
        for t in top20_word_prob:
            t_probas = []
            for el in t:
                t_probas.append(el[1])
            probas.append(t_probas)

        top20_word_prob = probas


        # evaluate topics 
        ctm_topics = [line.strip() for line in open(f'{ctm_dataset_folder}top_words_per_topic.txt', encoding="utf-8").readlines()]
        lotc_topics = [line.strip() for line in open(f'{lotclass_dataset_folder}cat_vocabulary.txt', encoding="utf-8").readlines()]
        assert len(ctm_topics) == len(lotc_topics),"Nr. of topics do not match."
        topic_df = pd.DataFrame(list(zip(ctm_topics, top20_word_prob, lotc_topics)), columns =['CTM most important words', 'CTM_word_proba','LOTClass most important words']) # index used as topic id
        topic_df.to_csv(f'{ctm_dataset_folder}topics_eval.csv', encoding='utf-8')  

        # get doc texts
        testtext_en = [line.strip() for line in open(f'{ctm_dataset_folder}test_en.txt', encoding="utf-8").readlines()]
        testtext_fr = [line.strip() for line in open(f'{ctm_dataset_folder}test_fr.txt', encoding="utf-8").readlines()]
        testtext_ger= [line.strip() for line in open(f'{ctm_dataset_folder}test_ger.txt', encoding="utf-8").readlines()]
        #testtext_port = [line.strip() for line in open('analysis/test_docs/test_port.txt', encoding="utf-8").readlines()]

        # get predictions of CTM
        predicted_topic = read_prediction_file(f'{ctm_dataset_folder}test_en_most_dominant_topic.txt')
        predicted_ger = read_prediction_file(f'{ctm_dataset_folder}test_ger_most_dominant_topic.txt')
        predicted_fr = read_prediction_file(f'{ctm_dataset_folder}test_fr_most_dominant_topic.txt')

        # get testdoc_topic distributions:
        t_d_1 = t_d_matrix(language=language1)
        t_d_2 = t_d_matrix(language=language2)

        # matching predictions
        matches = create_match_list(predicted_topic, predicted_ger)

        # get similarities of doc topic distribution on document level
        with open(f'{ctm_dataset_folder}kl_list_{lang}.data', 'rb') as f:
            # read the data as binary data stream
            kl_list = pickle.load(f)
            print(f'KL SIMILARITIES: {kl_list[:3]}...')

        # evaluate whether there is a clear dominant topic:
        predictions_weights1, weight_diffs1 = diff_of_top2_predictions(t_d_1)
        predictions_weights2, weight_diffs2 = diff_of_top2_predictions(t_d_2)

        #Cluster Test Docs 
        cluster_test_docs(doc_topic_distribution=t_d_1, predicted_topic=predicted_topic, testtexts=testtext_en, lang='en')
        cluster_test_docs(doc_topic_distribution=t_d_2, predicted_topic=predicted_ger, testtexts=testtext_ger, lang='ger')

        # Topic ID -> Topic Name
        mapping = mapping_def()
        labels_en = decode_label_id(mapping, predicted_topic)
        labels_ger = decode_label_id(mapping, predicted_ger)

        print(len(testtext_en))
        print(len(testtext_ger))

        print(len(predicted_topic))
        print(len(labels_en))
        print(len(predicted_ger))
        print(len(labels_ger))
        print(len(matches))

        # create df
        #df = pd.DataFrame(list(zip(testtext_en, testtext_ger, predicted_en, labels_en, predicted_ger, labels_ger, matches)), columns =['TestDocsEN', 'TestDocsGER', 'PredictedTopicEN', 'TopicLabelEN' 'PredictedTopicGER' 'TopicLabelGER', 'MatchingPredictions'])
        df = pd.DataFrame({
            f'TestDocs_{language1}': testtext_en, f'TestDocs_{language2}': testtext_ger, f'PredictedTopic_{language1}': predicted_topic, 
            f'TopicLabel_{language1}': labels_en, f'Topic Weight_{language1}': predictions_weights1, 
            f'Abs. weight difference bet. top 2 pred_{language1}': weight_diffs1, f'PredictedTopic_{language2}': predicted_ger, 
            f'TopicLabel_{language2}': labels_ger, f'Topic Weight_{language2}': predictions_weights2, 
            f'Abs. weight difference bet. top 2 pred_{language2}': weight_diffs1, 'MatchingPredictions': matches, 
            'Doc_Topic_KL_Div': kl_list})

        df.to_csv(f'{ctm_dataset_folder}comparison_overview{lang}.csv', encoding='utf-8') 

if __name__ == "__main__":
    main()