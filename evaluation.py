from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calculate_bleu(reference, generated):

    smoothie = SmoothingFunction().method4

    score = sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=smoothie
    )

    return score


def calculate_rouge(reference, generated):

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    return scores