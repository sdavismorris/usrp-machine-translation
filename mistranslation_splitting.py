from datasets import load_dataset
import csv
import pandas as pd
ds = load_dataset("nikitam/ACES", "ACES")

metric_data = {'source': [], 'translation': [], 'reference': [], 'bleu-score': [], 'chrf-score':[], 'ter-score': [], 'bert-score': [], 'bleurt-score': [], 'comet-score': [],'id': [], 'label': [], 'error-type': []}
# phenomena = {}
# for x in ds['train']['phenomena']:
#     if x not in phenomena:
#         phenomena[x] = 0
#     phenomena[x] += 1
#
# print(phenomena)


mistranslation_types = {'ambiguous', 'overly-literal', 'hallucination', 'anaphoric', 'pleonastic', 'nonsense', 'xnli', 'coreference', 'other', 'abc'}
mistranslation_data = {error : {key : [] for key in metric_data} for error in mistranslation_types}

i = 0
with (open("metric_data_2_with_labels.csv", encoding="utf8") as csv_file):
    reader = csv.reader(csv_file)
    # skip header
    row = next(reader)
    for row in reader:
        p = ds['train']['phenomena'][i//2]
        # assign specific phenomenon to general error type
        if 'hallucination' in p:
            error_type = 'hallucination'
        elif 'lexical' in p or 'modal' in p or 'mismatch' in p:
            error_type = 'other'
        elif 'overly-literal' in p:
            error_type = 'overly-literal'
        elif 'ambiguous-' in p:
            error_type = 'ambiguous'
        elif 'anaphoric' in p:
            error_type = 'anaphoric'
        elif 'pleonastic' in p:
            error_type = 'pleonastic'
        elif 'xnli' in p:
            error_type = 'xnli'
        elif 'nonsense' in p:
            error_type = 'nonsense'
        elif 'coreference' in p:
            error_type = 'coreference'
        else:
            error_type = 'abc'
        mistranslation_data[error_type]['source'].append(row[1])
        mistranslation_data[error_type]['translation'].append(row[2])
        mistranslation_data[error_type]['reference'].append(row[3])
        mistranslation_data[error_type]['bleu-score'].append(row[4])
        mistranslation_data[error_type]['chrf-score'].append(row[5])
        mistranslation_data[error_type]['ter-score'].append(row[6])
        mistranslation_data[error_type]['bert-score'].append(row[7])
        mistranslation_data[error_type]['bleurt-score'].append(row[8])
        mistranslation_data[error_type]['comet-score'].append(row[9])
        # sets of good/bad translation given same id
        mistranslation_data[error_type]['id'].append(i // 2)
        mistranslation_data[error_type]['label'].append(row[10])
        mistranslation_data[error_type]['error-type'].append(error_type)
        i += 1
        print(row)

# for error in mistranslation_data:
#     print(error)
#     print(len(mistranslation_data[error]['source']))
#
# # df = pd.DataFrame(error_data['addition'])
# # df.to_csv('addition.csv')
# #
for error in mistranslation_data:
    if error != 'abc':
        df = pd.DataFrame(mistranslation_data[error])
        df.to_csv(f'mistranslation-{error}_data.csv')
