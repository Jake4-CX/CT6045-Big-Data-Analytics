import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./datasets/penguins.csv')

sns.histplot(x='bill_length_mm', data=df)

plt.show()