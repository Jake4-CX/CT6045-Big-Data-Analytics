import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./datasets/iris.csv')

sns.histplot(x='sepal_length', data=df)

plt.show()