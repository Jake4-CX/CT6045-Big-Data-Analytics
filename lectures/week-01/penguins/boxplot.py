import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./datasets/penguins.csv')

sns.boxplot(x='species', y='bill_length_mm', data=df)

ax = sns.boxplot(x='species', y='bill_length_mm', data=df, color="grey")

plt.show()