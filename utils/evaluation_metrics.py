import evaluate

def compute_bleu(predictions, references):
    bleu = evaluate.load("google_bleu")
    return bleu.compute(predictions=predictions, references=references)
    