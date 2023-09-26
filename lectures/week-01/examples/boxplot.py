import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./datasets/iris.csv')

sns.boxplot(x='species', y='sepal_length', data=df)

ax = sns.boxplot(x='species', y='sepal_length', data=df, color="grey")

plt.show()