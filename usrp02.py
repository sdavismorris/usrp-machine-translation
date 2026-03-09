from datasets import load_dataset
import csv
import pandas as pd
ds = load_dataset("nikitam/ACES", "ACES")

metric_data = {'source': [], 'translation': [], 'reference': [], 'bleu-score': [], 'chrf-score':[], 'ter-score': [], 'id': [], 'qual': [], 'error-type': []}
phenomena = {}
for x in ds['train']['phenomena']:
    if x not in phenomena:
        phenomena[x] = 0
    phenomena[x] += 1

print(phenomena)
error_types = {'addition', 'omission', 'untranslated', 'mistranslation', 'do-not-translate', 'undertranslation', 'overtranslation', 'real-world-knowledge', 'wrong-language', 'punctuation'}
error_data = {error : {key : [] for key in metric_data} for error in error_types}

i = 0
with (open("metric_data.csv", encoding="utf8") as csv_file):
    reader = csv.reader(csv_file)
    # skip header
    next(reader)
    for row in reader:
        p = ds['train']['phenomena'][i//2]
        # assign specific phenomenon to general error type
        if 'hallucination' in p or 'lexical' in p or 'modal' in p \
                or 'overly-literal' in p or 'ambiguous-' in p or 'anaphoric' in p \
                or 'pleonastic' in p or 'mismatch' in p \
                or 'nonsense' in p or 'xnli' in p or 'coreference' in p:
            error_type = 'mistranslation'
        elif p == 'addition':
            error_type = 'addition'
        elif 'do-not' in p:
            error_type = 'do-not-translate'
        elif p == 'omission':
            error_type = 'omission'
        elif 'real-world' in p or 'antonym-replacement' == p or 'ref-ambiguous' in p:
            error_type = 'real-world-knowledge'
        elif 'hypernym-replacement' in p:
            error_type = 'undertranslation'
        elif 'hyponym-replacement' in p:
            error_type = 'overtranslation'
        elif 'similar' in p:
            error_type = 'wrong-language'
        elif 'untranslated' in p or 'copy' in p:
            error_type = 'untranslated'
        elif 'punctuation' in p:
            error_type = 'punctuation'
        else:
            print(p)
        error_data[error_type]['source'].append(row[1])
        error_data[error_type]['translation'].append(row[2])
        error_data[error_type]['reference'].append(row[3])
        error_data[error_type]['bleu-score'].append(row[4])
        error_data[error_type]['chrf-score'].append(row[5])
        error_data[error_type]['ter-score'].append(row[6])
        # sets of good/bad translation given same id
        error_data[error_type]['id'].append(i // 2)
        if i % 2 == 0:
            error_data[error_type]['qual'].append('good')
        else:
            error_data[error_type]['qual'].append('bad')
        error_data[error_type]['error-type'].append(error_type)
        i += 1
        print(row)

for error in error_data:
    print(error)
    print(len(error_data[error]['source']))

# df = pd.DataFrame(error_data['addition'])
# df.to_csv('addition.csv')
#
for error in error_data:
    df = pd.DataFrame(error_data[error])
    df.to_csv(f'{error}_data.csv')
