import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and display data
df = pd.read_csv("data_raw/Admission_Predict.csv")
print(df.head())
print(df.info())

'''# Plot variable distributions & their correlation
sns.set(style = "ticks")
sns.pairplot(df,kind="scatter")
plt.show()

# Plot correlation matrix
cor_df = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(cor_df,cmap='coolwarm',annot=True)
plt.title("Correlation matrix")
plt.show()

# Plot scatter plot of GRE & Chance of Admit
sns.scatterplot(x=df['GRE Score'],y=df['Chance of Admit '])
plt.show()'''

# Plot linear regression plot of Chance of Admit & GRE Score
sns.regplot(x='GRE Score',y='Chance of Admit ',data = df,line_kws={'color':'orange'})
plt.show()