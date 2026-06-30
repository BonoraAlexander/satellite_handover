import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('agent_log.csv')
print(df)
df['learning_step'] = np.arange(len(df))
df["learning_step"] = df["learning_step"]//20

plt.figure()
sns.lineplot(df, x='learning_step', y='reward')
plt.ylim(0,1)
plt.grid()
plt.savefig('reward.png', dpi=300)