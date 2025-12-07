import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/Cloudyum/E-Commerce-Customer-Churn-Prediction-Data-Mining/refs/heads/main/output/cleaned_data.csv'
df = pd.read_csv('https://raw.githubusercontent.com/Cloudyum/E-Commerce-Customer-Churn-Prediction-Data-Mining/refs/heads/main/output/cleaned_data.csv')


#Barplot Frequency
features = ["Tenure", "SatisfactionScore", "OrderCount"]

for col in features:
    plt.figure(figsize=(6,4))
    df.boxplot(column=col, by="Churn")
    plt.title(f"{col} vs Churn")
    plt.suptitle("")  
    plt.xlabel("Churn")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


#Barplot Col Vs Churn
categorical_features = ["CityTier", "Complain", "Tenure", "SatisfactionScore", "OrderCount"]

for col in categorical_features:
    plt.figure(figsize=(6,4))
    cross = df.groupby([col, "Churn"]).size().unstack(fill_value=0)
    cross.plot(kind="bar")
    plt.title(f"{col} vs Churn")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


#HeatMap
numeric_df = df.select_dtypes(include=[np.number])

corr = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)


for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        value = corr.iloc[i, j]
        plt.text(
            j, i, 
            f"{value:.3f}",    
            ha='center', 
            va='center', 
            fontsize=7
        )

plt.title("Correlation Heatmap with Values")
plt.tight_layout()
plt.show()
