import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('agent_log.csv')
print(df)
df["learning_step"] = df["learning_step"]//100

plt.figure()
sns.lineplot(df, x='learning_step', y='reward')
plt.savefig('reward.png', dpi=300)