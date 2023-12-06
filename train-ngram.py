import torch
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
import pickle
import utils
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', dest='train', type=str)
    parser.add_argument('-save', dest='save', type=str)
    parser.add_argument('-ngram', dest='ngram', type=int)
    args = parser.parse_args()

    if args.train:
        with open(args.train, 'r', encoding='utf-8') as f:
            train_data = f.readlines()

        corpus = utils.tokenize(train_data)

        ngram = args.ngram
        lm = KneserNeyInterpolated(ngram)
        train_data, padded_data = padded_everygram_pipeline(ngram, corpus)

        print(f'training {ngram}-gram....')
        lm.fit(train_data, padded_data)

        if args.save:
            print(f'saving {ngram}-gram model....')
            with open(args.save, 'wb') as f:
                pickle.dump(lm, f)


