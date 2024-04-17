# run garbage code here
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.random.normal(loc=0, scale=1, size=1000)  # Example data
bins = 20
range = (-3, 3)
alpha = 0.7
color = 'blue'

plt.hist(data, bins=bins, range=range, alpha=alpha, color=color, label='Data Counts')
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Histogram of Data (Counts)')
plt.legend()
plt.show()
data = np.random.normal(loc=0, scale=1, size=1000)  # Example data
bins = 20
range = (-3, 3)
alpha = 0.7
color = 'blue'

sns.histplot(data, bins=bins, kde=True, stat='density', color=color, alpha=alpha, label='Data Density')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram of Data (Density)')
plt.legend()
plt.show()