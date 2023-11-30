import torch
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
import pickle
import utils
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

if __name__=='__main__':
    with open('data/train.tone', 'r', encoding='utf-8') as f:
        train_data = f.readlines()

    corpus = utils.tokenize(train_data)

    ngram = 2
    lm2 = KneserNeyInterpolated(ngram)
    train_data, padded_data = padded_everygram_pipeline(ngram, corpus)

    print('training 2-gram....')
    lm2.fit(train_data, padded_data)

    print('saving 2-gram model....')
    with open('out/trained_models/2gram-model.pkl', 'wb') as f:
        pickle.dump(lm2, f)

    ngram = 3
    lm3 = KneserNeyInterpolated(ngram)
    train_data, padded_data = padded_everygram_pipeline(ngram, corpus)

    print('training 3-gram....')
    lm3.fit(train_data, padded_data)

    print('saving 3-gram model....')
    with open('out/trained_models/3gram-model.pkl', 'wb') as f:
        pickle.dump(lm3, f)

