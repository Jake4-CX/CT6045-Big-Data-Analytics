import pandas as pd
data = pd.read_csv('../datasets/Finance_data.csv')

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(
  stop_words='english',
  ngram_range=(1, 1),
  tokenizer=token.tokenize
)
text_counts = cv.fit_transform(data['Sentence'])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
  text_counts,
  data['Sentiment'],
  test_size=0.25,
  random_state=5
)

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)

from sklearn import metrics
predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, Y_test)
print("Accuracy: ", accuracy_score)