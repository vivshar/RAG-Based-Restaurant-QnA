!pip install nltk
!pip install rouge

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from rouge import Rouge
import pandas as pd

nltk.download('punkt')

# Load your DataFrame
df_test = pd.read_csv('/content/QnA_gpt4_without_context.csv')

def bleu1(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)


def bleu4(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)

rouge = Rouge()

ground_truth = df_test["answer"].values.tolist()
generated_ans = df_test["predicted_answer"].values.tolist()

Bleu1 = []
Bleu4 = []
rouge_1_score = []
rouge_2_score = []
rouge_L_score = []

for i in range(len(ground_truth)):
    grndAns = ground_truth[i]
    gen_ans = generated_ans[i]

    # BLEU SCORES
    gen_ans_ = word_tokenize(gen_ans.lower())
    grndAns_ = word_tokenize(grndAns.lower())
    bleu1s = bleu1([grndAns_], gen_ans_)
    bleu4s = bleu4([grndAns_], gen_ans_)
    Bleu1.append(bleu1s)
    Bleu4.append(bleu4s)

    # Rouge
    scores = rouge.get_scores(gen_ans, grndAns)
    r1s = scores[0]['rouge-1']['f']  # f1 score
    r2s = scores[0]['rouge-2']['f']
    rLs = scores[0]['rouge-l']['f']
    rouge_1_score.append(r1s)
    rouge_2_score.append(r2s)
    rouge_L_score.append(rLs)

print("BLEU 1 Gram: ", np.mean(Bleu1))
print("BLEU 4 Gram: ", np.mean(Bleu4))
print("ROUGE 1 Gram:", np.mean(rouge_1_score))
print("ROUGE 2 Gram:", np.mean(rouge_2_score))
print("ROUGE L Gram:", np.mean(rouge_L_score))
