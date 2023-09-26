import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./datasets/iris.csv')

plt.scatter(df["sepal_length"], df["sepal_width"], alpha=0.2, s=100*df["petal_width"], c=df["species_number"], cmap="viridis")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.show()