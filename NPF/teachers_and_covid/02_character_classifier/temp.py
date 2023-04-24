# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u"We congratulated them by say congratulations")

for token in doc:
    print(token, "becomes",  token.lemma_)

doc2 = nlp(u"We were watching a movie. It was insightful.")

for token in doc2:
    print(token, "becomes", token.lemma_)
