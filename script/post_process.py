import json

from codebleu import calc_codebleu
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import Levenshtein

def get_evaluation_metrics(ref, pred):
    """
    Calculate various scores between reference and prediction
    Args:
        ref: reference resolution
        pred: predicted resolution
    Returns:
        various scores between reference and prediction
    """
    codebleu = calc_codebleu([ref], [pred], lang="java")['codebleu']
    bleu = sentence_bleu([ref.split()], pred.split())
    my_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = my_rouge_scorer.score(ref, pred)['rougeL'].fmeasure
    levenshtein = Levenshtein.distance(ref, pred)

    return {
        "ref": ref,
        "pred": pred,
        "codebleu": codebleu,
        "bleu": bleu,
        "rouge": rouge,
        "levenshtein": levenshtein
    }

with open("results/evaluation_results_baseline.json", "r") as f:
    baseline_results = json.load(f)

for result in baseline_results:
    result.update(get_evaluation_metrics(result['ref'][3:], result['pred']))

with open("evaluation_results_adjusted.json", "w") as f:
    json.dump(baseline_results, f, indent=4)

