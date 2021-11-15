# document, the partitition, and the label

paste train.txt  train_partition.txt train_labels.txt > corpus_1.tsv
paste test.txt  test_partition.txt test_labels.txt > corpus_2.tsv

# Concatenate corpus 1 and 2
cat corpus_1.tsv corpus_2.tsv > corpus.tsv

# create file containing all text -> for vocab.
cut -f1 corpus.tsv > all_texts.txt  # If field separator is tab 
