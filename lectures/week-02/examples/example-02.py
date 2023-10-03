from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()

text_1 = "The book was a perfect balance between writing style and plot."
text_2 = "The pizza tasts terrible."

sent_1 = sentiment.polarity_scores(text_1)
sent_2 = sentiment.polarity_scores(text_2)

print("Sentiment of text_1: ", sent_1)
print("Sentiment of text_2: ", sent_2)