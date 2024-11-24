import torch
import pickle
import utils
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse
import os
import multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

parser = argparse.ArgumentParser()
parser.add_argument('-load', dest='load', type=str)
parser.add_argument('-ngram', dest='ngram', type=int)
parser.add_argument('-npool', dest='npool', type=int)
parser.add_argument('-kbeam', dest='kbeam', type=int)
parser.add_argument('-outdir', dest='outdir', type=str)
parser.add_argument('-test_notone', dest='test_notone', type=str)
parser.add_argument('-test_tone', dest='test_tone', type=str)
args = parser.parse_args()

lm = None
with open(args.load, 'rb') as f:
    lm = pickle.load(f)

detokenize = TreebankWordDetokenizer().detokenize

# greedy search
def greedy_search(words, model):
    sequences = []
    for idx, word in enumerate(words):
        if idx == 0:
            sequences = [([x], 0.0) for x in utils.get_all_tones(word)]
        else:
            all_sequences = []
            for seq in sequences:
                for next_word in utils.get_all_tones(word):
                    current_word = seq[0][-1]
                    score = model.logscore(next_word, [current_word])
                    new_seq = seq[0].copy()
                    new_seq.append(next_word)
                    all_sequences.append((new_seq, seq[1] + score))
            # sort and keep only the top-scoring sequence
            all_sequences = sorted(all_sequences, key=lambda x: x[1], reverse=True)
            sequences = [all_sequences[0]]
    return sequences[0] 

# beam search
def beam_search(words, model, k=3):
    sequences = []
    for idx, word in enumerate(words):
        if idx == 0:
            sequences = [([x], 0.0) for x in utils.get_all_tones(word)]
        else:
            all_sequences = []
            for seq in sequences:
                for next_word in utils.get_all_tones(word):
                    current_word = seq[0][-1]
                    score = model.logscore(next_word, [current_word])
                    new_seq = seq[0].copy()
                    new_seq.append(next_word)
                    all_sequences.append((new_seq, seq[1] + score))
            # sort and keep only the top-k sequence
            all_sequences = sorted(all_sequences,key=lambda x: x[1], reverse=True)
            sequences = all_sequences[:k]
    return sequences

def restore_beam(sentence, lm, k=5):
    sentence = sentence.replace('\n', '')
    result = beam_search(sentence.lower().split(), lm, k)
    return detokenize(result[0][0])

def restore_greedy(sentence, lm):
    sentence = sentence.replace('\n', '')
    result = greedy_search(sentence.lower().split(), lm)
    return detokenize(result[0])

def restore_greedy_pool(words):
    return restore_greedy(words, lm)

def restore_beam_pool(words):
    kbeam = args.kbeam
    return restore_beam(words, lm, kbeam)

if __name__=='__main__':
    if args.load and args.ngram and args.npool:
        ngram = args.ngram
        npool = args.npool
        lm = None
        kbeam = 0

        if args.test_notone and args.test_tone:
            
            with open(args.test_notone, 'r', encoding='utf-8') as f:
                test_data = f.readlines()
            
            with open(args.test_tone, 'r', encoding='utf-8') as f:
                test_data_tone = f.readlines()

            '''if not args.kbeam:
                print(f'testing {args.ngram}-gram model with greedy search....')
                with multiprocessing.Pool(npool) as p:
                    results = list(tqdm(p.imap(restore_greedy_pool, test_data), total=len(test_data))) 

                total = 0
                total_correct = 0
                for idx, sentence in enumerate(results):
                    correct, n = utils.evaluate(sentence, test_data_tone[idx])
                    total += n
                    total_correct += correct
                print(f'test_acc={total_correct/total}')

                if args.outdir:
                    out_file = os.path.join(args.outdir, f'{ngram}gram-greedy.out.txt')
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(results))'''

            if args.kbeam:
                print(f'testing {args.ngram}-gram model with beam search....') 
                print(f'k={args.kbeam}')

                with multiprocessing.Pool(npool) as p:
                    results = list(tqdm(p.imap(restore_beam_pool, test_data), total=len(test_data)))

                total = 0
                total_correct = 0
            
                for idx, sentence in enumerate(results):
                    correct, n = utils.evaluate(sentence, test_data_tone[idx])
                    total += n
                    total_correct += correct
                    
                print(f'test_acc={total_correct/total}')

                if args.outdir:
                    out_file = os.path.join(args.outdir, f'{ngram}gram-beam-{args.kbeam}.out.txt')
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(results))