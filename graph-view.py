import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set Seaborn style
sns.set(style="whitegrid", context="talk", palette="Set2")

data = {
    'Model': ['DenseNet-121', 'ResNet-18', 'EfficientNet-b0'],
    'AUC': [0.8968, 0.8836, 0.9359],
    'ACC': [0.9375, 0.9203, 0.9516],
    'F1': [0.6875, 0.6222, 0.7207],
}

df = pd.DataFrame(data)
df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6), dpi=150)
ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', edgecolor='black')

for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Score')
plt.xlabel('')

plt.legend(title='Metric')
plt.tight_layout()

plt.show()
