import re
import torch
from nltk.tokenize import word_tokenize
import numpy as np
import dataloader
import torch.utils

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable
    
def tokenize(data):
    print('tokenizing...')
    m = torch.load('src/tokenizer_standarized.h5')
    vnword = m['tone'].word_index
    corpus = []
    for line in tqdm(data):
        tokens = word_tokenize(line)
        for idx, token in enumerate(tokens):
            if token not in vnword:
                tokens[idx] = '<unk>'
            
        corpus.append(tokens)
    return corpus

# get vietnamese dictionary
def get_vn_dict():
    tokenizer = torch.load('src/tokenizer_standarized.h5')
    words = list(tokenizer['tone'].word_index.keys())
    vn_dict = {}
    for word in words:
        no_tone = remove_tones(word)
        if not no_tone in vn_dict.keys():
            vn_dict.setdefault(no_tone, [word])
        else:
            vn_dict[no_tone].append(word)
    return vn_dict

def create_dataset(src_file, trg_file):
    tokenizer = torch.load('src/tokenizer_standarized.h5')
    src_tokenizer = tokenizer['notone']
    trg_tokenizer = tokenizer['tone']

    dataset = dataloader.Dataset(src_tokenizer, trg_tokenizer, src_file, trg_file)

    return dataset


# extract words and numbers in sentences
def extract_words(sentence):
    pattern = '[AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊ'+ \
            'OÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬ'+ \
            'ĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴ'+ \
            'AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢ'+ \
            'UƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊ'+ \
            'OÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐ'+ \
            'EÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴ'+ \
            'AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢ'+ \
            'UƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴA-Z' + '0-9' + ']+'
    indicies = []
    words = []
    for m in re.finditer(pattern, sentence, re.IGNORECASE):
        words.append(m.group(0))
        indicies.append((m.start(0), m.end(0)))
    
    return words, indicies

# get all possible diacritics of a given word
def get_all_tones(word):
    '''word = remove_tones(word.lower())
    words_with_tones = {word}
    for w in open('vn_syllables.txt').read().splitlines():
        no_tones = remove_tones(w.lower())
        if no_tones == word:
            words_with_tones.add(w)
    
    return words_with_tones'''
    vn_dict = get_vn_dict()
    word = remove_tones(word.lower())
    if word in vn_dict:
        return vn_dict[word]
    else:
        return [word]

# remove diacritics
def remove_tones(utf8_str):
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    intab = [ch for ch in str(intab_l+intab_u)]

    outtab_l = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"
    outtab_u = "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"
    outtab = outtab_l + outtab_u

    r = re.compile("|".join(intab))
    replaces_dict = dict(zip(intab, outtab))

    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)

# normalize diacritics positions
def normalize_tone(utf8_str):
    intab_l = "áàảãạâấầẩẫậăắằẳẵặđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    intab_u = "ÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
    intab = [ch for ch in str(intab_l + intab_u)]

    outtab_l = [
        "a1", "a2", "a3", "a4", "a5",
        "a6", "a61", "a62", "a63", "a64", "a65",
        "a8", "a81", "a82", "a83", "a84", "a85",
        "d9",
        "e1", "e2", "e3", "e4", "e5",
        "e6", "e61", "e62", "e63", "e64", "e65",
        "i1", "i2", "i3", "i4", "i5",
        "o1", "o2", "o3", "o4", "o5",
        "o6", "a61", "o62", "o63", "o64", "o65",
        "o7", "o71", "o72", "o73", "o74", "o75",
        "u1", "u2", "u3", "u4", "u5",
        "u7", "u71", "u72", "u73", "u74", "u75",
        "y1", "y2", "y3", "y4", "y5",
    ]

    outtab_u = [
        "A1", "A2", "A3", "A4", "A5",
        "A6", "A61", "A62", "A63", "A64", "A65",
        "A8", "A81", "A82", "A83", "A84", "A85",
        "D9",
        "E1", "E2", "E3", "E4", "E5",
        "E6", "E61", "E62", "E63", "E64", "E65",
        "I1", "I2", "I3", "I4", "I5",
        "O1", "O2", "O3", "O4", "O5",
        "O6", "O61", "O62", "O63", "O64", "O65",
        "O7", "O71", "O72", "O73", "O74", "O75",
        "U1", "U2", "U3", "U4", "U5",
        "U7", "U71", "U72", "U73", "U74", "U75",
        "Y1", "Y2", "Y3", "Y4", "Y5",
    ]

    r = re.compile("|".join(intab))
    replaces_dict = dict(zip(intab, outtab_l + outtab_u))

    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)

# nornamalize diacritics
def normalize_tones(utf8_str):
    intab_l = "áàảãạâấầẩẫậăắằẳẵặđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    intab_u = "ÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
    intab = [ch for ch in str(intab_l + intab_u)]

    outtab_l = [
        "a1", "a2", "a3", "a4", "a5",
        "a6", "a61", "a62", "a63", "a64", "a65",
        "a8", "a81", "a82", "a83", "a84", "a85",
        "d9",
        "e1", "e2", "e3", "e4", "e5",
        "e6", "e61", "e62", "e63", "e64", "e65",
        "i1", "i2", "i3", "i4", "i5",
        "o1", "o2", "o3", "o4", "o5",
        "o6", "a61", "o62", "o63", "o64", "o65",
        "o7", "o71", "o72", "o73", "o74", "o75",
        "u1", "u2", "u3", "u4", "u5",
        "u7", "u71", "u72", "u73", "u74", "u75",
        "y1", "y2", "y3", "y4", "y5",
    ]

    outtab_u = [
        "A1", "A2", "A3", "A4", "A5",
        "A6", "A61", "A62", "A63", "A64", "A65",
        "A8", "A81", "A82", "A83", "A84", "A85",
        "D9",
        "E1", "E2", "E3", "E4", "E5",
        "E6", "E61", "E62", "E63", "E64", "E65",
        "I1", "I2", "I3", "I4", "I5",
        "O1", "O2", "O3", "O4", "O5",
        "O6", "O61", "O62", "O63", "O64", "O65",
        "O7", "O71", "O72", "O73", "O74", "O75",
        "U1", "U2", "U3", "U4", "U5",
        "U7", "U71", "U72", "U73", "U74", "U75",
        "Y1", "Y2", "Y3", "Y4", "Y5",
    ]

    r = re.compile("|".join(intab))
    replaces_dict = dict(zip(intab, outtab_l + outtab_u))

    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)

def simplify(word):
    """
    normalize and simplify a vni word:
    * move tone digit to the end
    * return only digits
    * return 0 if there is no digit
    """
    if word.isalpha(): 
        return '0'
    ret = ''
    tone = ''
    for letter in word:
        if '1' <= letter <= '9':
            if '1' <= letter <= '5':
                # assert len(tone) == 0, '{}, {}'.format(tone, word)
                if tone != '':
                    return '#'  # ignore this word
                tone = letter
            else:
                ret += letter
    return ret + tone

def process_line(line):
    """
    process a line
    :param line:
    :return: no_tone_line, no_tone_words, simplified_words
    """
    # utf8_line = line.encode('utf-8')

    utf8_line = line.strip('\n')

    no_tone_line_pre = remove_tones(utf8_line)
    normalized_line_pre = normalize_tones(utf8_line)

    no_tone_words, _ = extract_words(no_tone_line_pre)
    normalized_words, _ = extract_words(normalized_line_pre)

    assert len(no_tone_words) == len(normalized_words)

    filtered_no_tone_words = []
    simplified_words = []
    for i, word in enumerate(no_tone_words):
        if not word.isalpha():
            continue
        simplified_word = simplify(normalized_words[i])
        filtered_no_tone_words.append(word)
        simplified_words.append(simplified_word)

    return filtered_no_tone_words, simplified_words

def evaluate(pred, label):
    _, pred_punc = process_line(pred)
    _, label_punc = process_line(label)

    pred_punc = np.array(pred_punc)
    label_punc = np.array(label_punc)

    true_values = np.sum(pred_punc==label_punc)
    n_values = len(pred_punc)

    return true_values, n_values