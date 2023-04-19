import pickle
import os
from transformers import pipeline
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

subs = [
    "Cornell", 
    "Indiana",
    "Lincoln",
    "MSU",
    "Northwestern",
    "OSU",
    "Penn State",
    "Purdue",
    "Rutgers",
    "UIowa",
    "UIUC",
    "UMD",
    "UofM",
    "UofMN",
    "UWMadison"
]

X_test = [
    ["CAS vs Human Ecology, Which one is easier to get into?", "Cornell"],
    ["Hi there! I'm interested in getting a bachelor's degree in business and history.", "Indiana"],
    ["I wanted to minor in game development but is there like an official way to go about that?", "msu"],
    ["I was thinking of going to a steakhouse this weekend and was wondering what the best steakhouse is.", "Penn State"]
]

BRUH  = [
    "CAS vs Human Ecology, Which one is easier to get into?",
    "Hi there! I'm interested in getting a bachelor's degree in business and history.",
    "I wanted to minor in game development but is there like an official way to go about that?",
    "I was thinking of going to a steakhouse this weekend and was wondering what the best steakhouse is.",
]
tfidf, vect, clf = pickle.load(open('bigtenLinearSVC.pickle', 'rb'))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

for s, x in zip(clf.predict(vect.transform(BRUH)), X_test):
    print("Post: " + x[0])
    print("Predicted: " + s)
    print("Actual: " + x[1], '\n')

# pred_probability = []
# for eachArr in clf.decision_function(vect.transform(X_test)):
#     pred_probability.append(softmax(eachArr))

# for l, p in zip(pred_probability[0], subs):
#     if len(p) > 8:
#         print(p + '\t' + str(l))
#     else:
#         print(p + '\t\t' + str(l))




# OUTPUT_DIR = "./bigepoch"
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token = "<bos>", eos_token ="<eos>", truncation_side='right')
# generator = pipeline('text-generation', model=OUTPUT_DIR, tokenizer=tokenizer)
# output_sequences = generator(X_test, max_length=200, num_return_sequences=5)
# for sentence in output_sequences:
#     print(sentence["generated_text"].strip())