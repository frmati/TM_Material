from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import numpy as np
import pandas as pd

train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

pprint(list(train.target_names))



train_art_list = [d.replace("\n", " ") for d in train['data']]
train_label_list = [l for l in train['target']]
train_partition_list = ['train' for el in range(len(train_art_list))]



test_art_list = [d.replace("\n", " ") for d in test['data']]
test_label_list = [l for l in test['target']]
test_partition_list = ['test' for el in range(len(test_art_list))]

# delete additional characters:
delete_chars = ['\t', '\n', '\r']
train_art_list_cleaned = []
for art in train_art_list:
    for ch in delete_chars:
        if ch in art:
            art = art.replace(ch, ' ')
    train_art_list_cleaned.append(art)


test_art_list_cleaned = []
for art in test_art_list:
    for ch in delete_chars:
        if ch in art:
            art = art.replace(ch, ' ')
    test_art_list_cleaned.append(art)

print(len(train_art_list_cleaned))
print(len(train_label_list))
print(len(train_partition_list))
print(len(test_art_list_cleaned))
print(len(test_label_list))
print(len(test_partition_list))

with open('all_texts.txt', 'w', encoding='utf-8') as c:  
    for el in train_art_list_cleaned:
        c.write(el)
        c.write('\n')
    for el in test_art_list_cleaned:
        c.write(el)
        c.write('\n')

with open('train_texts.txt', 'w', encoding='utf-8') as c:  
    for el in train_art_list_cleaned:
        c.write(el)
        c.write('\n')

with open('test_texts.txt', 'w', encoding='utf-8') as c:  
    for el in test_art_list_cleaned:
        c.write(el)
        c.write('\n')

with open('all_labels.txt', 'w', encoding='utf-8') as l:  
    for el in train_label_list:
        l.write(str(el.item()))
        l.write('\n')
    for el in test_label_list:
        l.write(str(el.item()))
        l.write('\n')

with open('train_labels.txt', 'w', encoding='utf-8') as l:  
    for el in train_label_list:
        l.write(str(el.item()))
        l.write('\n')
    

with open('test_labels.txt', 'w', encoding='utf-8') as l: 
    for el in test_label_list:
        l.write(str(el.item()))
        l.write('\n')

with open('partitions.txt', 'w', encoding='utf-8') as p:  
    for el in train_partition_list:
        p.write(el)
        p.write('\n')
    for el in test_partition_list:
        p.write(el)
        p.write('\n')


# generate corpus.tsv document


dict = {'article': train_art_list_cleaned + test_art_list_cleaned, 'partition': train_partition_list + test_partition_list, 'label': train_label_list + test_label_list}
df = pd.DataFrame(dict)
# only strings
df = df.astype(str)
df.to_csv('~/Py/TM_Baselines/datasets/20NewsGroup_original/corpus.tsv', index=False, sep='\t', header=None, encoding='utf-8', na_rep=' ') 
