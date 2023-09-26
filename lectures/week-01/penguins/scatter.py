import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./datasets/penguins.csv')

plt.scatter(df["bill_length_mm"], df["bill_depth_mm"])

plt.xlabel("bill_length_mm")
plt.ylabel("bill_depth_mm")

plt.show()