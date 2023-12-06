import torch
import pickle
import utils
from nltk.tokenize.treebank import TreebankWordDetokenizer
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable
    
# beam search
def beam_search(words, model, k=5):
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
            all_sequences = sorted(all_sequences,key=lambda x: x[1], reverse=True)
            sequences = all_sequences[:k]
    return sequences

def restore(sentence, lm, detokenize, k=5):
    sentence = sentence.replace('\n', '')
    result = beam_search(sentence.lower().split(), lm, k)
    return detokenize(result[0][0])

if __name__=='__main__':
    with open('data/test.notone', 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    
    with open('data/test.tone', 'r', encoding='utf-8') as f:
        test_data_tone = f.readlines()

    detokenize = TreebankWordDetokenizer().detokenize

    # load 2gram model
    print('loading 2-gram model....')
    with open('out/models/2gram-model.pkl', 'rb') as f:
        lm2 = pickle.load(f)

    # load 3gram model
    print('loading 3-gram model....')
    with open('out/models/3gram-model.pkl', 'rb') as f:
        lm3 = pickle.load(f)

    test_sentence = 'minh dang la sinh vien nam cuoi nganh khoa hoc may tinh o truong dai hoc Notre Dame'
    print(f'input: {test_sentence}')
    print("expected output: mình đang là sinh viên năm cuối ngành khoa học máy tính ở trường đại học Notre Dame")
    start = time.perf_counter()
    result = restore(test_sentence, lm2, detokenize)
    end = time.perf_counter()
    print(f'2-gram result: {result} ({end-start:0.4f}s)')
    start = time.perf_counter()
    result = restore(test_sentence, lm3, detokenize)
    end = time.perf_counter()
    print(f'3-gram result: {result} ({end-start:0.4f}s)')

    