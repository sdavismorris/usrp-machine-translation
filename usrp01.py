from datasets import load_dataset
import sacrebleu
import pandas as pd
ds = load_dataset("nikitam/ACES", "ACES")
print(ds['train'][0])

bleu_scores = {'source': [], 'translation': [], 'score': []}

for row in ds['train']:
    #bleu
    source = row['source']
    reference = row['reference']
    hypothesis = row['good-translation']
    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
    bleu_score = bleu.score
    bleu_scores['source'].append(source)
    bleu_scores['translation'].append(hypothesis)
    bleu_scores['score'].append(bleu_score)

    hypothesis = row['incorrect-translation']
    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
    bleu_score = bleu.score
    bleu_scores['source'].append(source)
    bleu_scores['translation'].append(hypothesis)
    bleu_scores['score'].append(bleu_score)
#
# # print(bleu_scores['source'][0])
# # print(bleu_scores['translation'][0])
# # print(bleu_scores['score'][0])
# #
# # print(bleu_scores['source'][1])
# # print(bleu_scores['translation'][1])
# # print(bleu_scores['score'][1])
#
chrf_scores = {'source': [], 'translation': [], 'score': []}
for row in ds['train']:
    # chrF
    source = row['source']
    reference = row['reference']
    hypothesis = row['good-translation']
    chrf = sacrebleu.corpus_chrf([hypothesis], [[reference]])
    chrf_score = chrf.score
    chrf_scores['source'].append(source)
    chrf_scores['translation'].append(hypothesis)
    chrf_scores['score'].append(chrf_score)

    hypothesis = row['incorrect-translation']
    chrf = sacrebleu.corpus_chrf([hypothesis], [[reference]])
    chrf_score = chrf.score
    chrf_scores['source'].append(source)
    chrf_scores['translation'].append(hypothesis)
    chrf_scores['score'].append(chrf_score)

# print(chrf_scores['source'][0])
# print(chrf_scores['translation'][0])
# print(chrf_scores['score'][0])
#
print(chrf_scores['source'][1])
print(chrf_scores['translation'][1])
print(chrf_scores['score'][1])

ter_scores = {'source': [], 'translation': [], 'score': []}
for row in ds['train']:
    # ter
    source = row['source']
    reference = row['reference']
    hypothesis = row['good-translation']
    ter = sacrebleu.corpus_ter([hypothesis], [[reference]])
    ter_score = ter.score
    ter_scores['source'].append(source)
    ter_scores['translation'].append(hypothesis)
    ter_scores['score'].append(ter_score)

    hypothesis = row['incorrect-translation']
    ter = sacrebleu.corpus_ter([hypothesis], [[reference]])
    ter_score = ter.score
    ter_scores['source'].append(source)
    ter_scores['translation'].append(hypothesis)
    ter_scores['score'].append(ter_score)

print(ter_scores['source'][2])
print(ter_scores['translation'][2])
print(ter_scores['score'][2])
#
# print(ter_scores['source'][3])
# print(ter_scores['translation'][3])
# print(ter_scores['score'][3])

metric_data = {'source': [], 'translation': [], 'reference': [], 'bleu-score': [], 'chrf-score':[], 'ter-score': []}
for i in range(len(bleu_scores['source'])):
    metric_data['source'].append(bleu_scores['source'][i])
    metric_data['translation'].append(bleu_scores['translation'][i])
    metric_data['reference'].append(ds['train'][i//2]['reference'])
    metric_data['bleu-score'].append(bleu_scores['score'][i])
    metric_data['chrf-score'].append(chrf_scores['score'][i])
    metric_data['ter-score'].append(ter_scores['score'][i])

for key in metric_data:
    print(len(metric_data[key]))
df = pd.DataFrame(metric_data)
df.to_csv('metric_data.csv')



