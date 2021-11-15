#!/usr/bin/env python 
from collections import Counter
from string import punctuation
import string

def create_vocabulary(filename, output_filename, size, character_level=False, min_count=1):
    vocab = Counter()
    punctuation = []
    with open(filename) as input_file, \
         open(output_filename, 'w') as output_file:
        for line in input_file:
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.strip() if character_level else line.split()

            for w in line:
                vocab[w] += 1

        if min_count > 1:
            vocab = {w: c for (w, c) in vocab.items() if c >= min_count}

        vocab = {w: c for (w, c) in vocab.items()}
        vocab_list = sorted(vocab, key=lambda w: (-vocab[w], w))
        if 0 < size < len(vocab_list):
            #vocab_list = vocab_list[:size]
            ...

        output_file.writelines(w.lower() + '\n' for w in vocab_list)

    return dict(map(reversed, enumerate(vocab_list))) 




def main():
    create_vocabulary('all_texts.txt', 'vocab.txt', 100000, character_level=False, min_count=1)


if __name__ == "__main__": 

    main() 

 

