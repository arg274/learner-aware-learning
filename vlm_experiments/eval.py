import evaluate
from glob import glob
import pandas as pd

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

scores = []

results = glob('vlm_results/*.json')
with open('Fintech/Lecture1_clip-2.json', 'r', encoding='utf-8') as reference_file:
    reference = reference_file.read()
    
for result_file in results:
    with open(result_file, 'r', encoding='utf-8') as f:
        prediction = f.read()
    
    model, input_type = result_file.split('/')[-1].replace('.json','').split('_')
    
    bleu_score = bleu.compute(predictions=[prediction], references=[reference])
    rouge_score = rouge.compute(predictions=[prediction], references=[reference])
    bert_score = bertscore.compute(predictions=[prediction], references=[reference], lang='en')
    
    scores.append({
        'model': model,
        'input_type': input_type,
        'bleu': bleu_score['bleu'],
        'rouge1': rouge_score['rouge1'],
        'rouge2': rouge_score['rouge2'],
        'rougeL': rouge_score['rougeL'],
        'rougeLSum': rouge_score['rougeLsum'],
        'bert_precision': bert_score['precision'][0],
        'bert_recall': bert_score['recall'][0],
        'bert_f1': bert_score['f1'][0],
    })
    
scores_df = pd.DataFrame(scores)
scores_df.to_excel('vlm_results/vlm_evaluation_scores.xlsx', index=False)
