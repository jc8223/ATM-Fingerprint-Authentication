import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('results.csv')

sns.scatterplot(data=data, x="threshold", y="score", hue="match", palette=["cyan", "orange"])
plt.show()
