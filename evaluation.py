from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calculate_bleu(reference, generated):
    smoothie = SmoothingFunction().method4

    return sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=smoothie
    )


def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    return scorer.score(reference, generated)
