import utils

with open('data/test.tone', 'r', encoding='utf-8') as f:
    test_data_tone = f.readlines()

with open('out/outfiles/3gram-beam-5.out.txt', 'r', encoding='utf-8') as f:
    out_data = f.readlines()

total = 0
total_correct = 0
for idx, sentence in enumerate(out_data):
    correct, n = utils.evaluate(sentence, test_data_tone[idx])
    total += n
    total_correct += correct
print(f'test_acc={total_correct/total}')