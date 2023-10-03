from textblob import TextBlob

text_1 = "The movie was so awesome"
text_2 = "The food here tastes terrible."

# Determine Polarity
p_1 = TextBlob(text_1).sentiment.polarity
p_2 = TextBlob(text_2).sentiment.polarity

# Determine Subjectivity
s_1 = TextBlob(text_1).sentiment.subjectivity
s_2 = TextBlob(text_2).sentiment.subjectivity

print("Polarity of text_1: ", p_1)
print("Subjectivity of text_1: ", s_1)

print("Polarity of text_2: ", p_2)
print("Subjectivity of text_2: ", s_2)