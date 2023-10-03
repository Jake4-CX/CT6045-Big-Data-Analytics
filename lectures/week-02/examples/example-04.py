import nltk
import pandas as pd
from textblob import Word
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

data = pd.read_csv('../datasets/Finance_data.csv')

def create_sentiment(Sentiment):
  if Sentiment=='negative':
    return -1
  elif Sentiment=='positive':
    return 1
  else: # neutral sentiment
    return 0

data['SentimentNo'] = data['Sentiment'].apply(create_sentiment)

def cleaning(df, stop_words):
  df['Sentence'] = df['Sentence'].apply(lambda x: " ".join(x.lower() for x in x.split()))

  # remove digits/numbers
  df['Sentence'] = df['Sentence'].str.replace('d','')

  # remove stopwords
  df['Sentence'] = df['Sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

  # Lemmatization
  df['Sentence'] = df['Sentence'].apply(lambda x: " ".join([Word(x).lemmatize() for x in x.split()]))

  return df

stop_words = stopwords.words('english')
data_cleaned = cleaning(data, stop_words)

tokenizer = Tokenizer(num_words=500, split=" ")
tokenizer.fit_on_texts(data_cleaned['Sentence'].values)

X = tokenizer.texts_to_sequences(data_cleaned['Sentence'].values)
X = pad_sequences(X)

model = Sequential()

top_words = 500
embedding_vector_length = 120
max_review_length = X.shape[1]

model.add(Embedding(500, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(704, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(352, activation='LeakyReLU'))
model.add(Dense(3, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data_cleaned['SentimentNo'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=5)
model.fit(X_train, Y_train, epochs = 20, batch_size=32, verbose =1)

#Model Testing
model.evaluate(X_test,Y_test)