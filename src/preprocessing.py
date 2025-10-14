import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = "data/ecommerce_churn.xlsx"

xls = pd.ExcelFile(file_path)
print("Sheet names:", xls.sheet_names)

df = pd.read_excel(file_path, sheet_name="Ecommerce_Data")

print("\nData preview:")
print(df.head())
print("\nInfo:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\n--- Data Cleaning ---")

num_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
            'OrderAmountHikeFromlastYear', 'CouponUsed', 
            'OrderCount', 'DaySinceLastOrder']

for col in num_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

print("\nMissing values setelah imputasi:")
print(df[num_cols].isnull().sum())

cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
             'PreferedOrderCat', 'MaritalStatus']

for col in cat_cols:
    df[col] = df[col].astype('category')

print("\nTipe data setelah ubah kategori:")
print(df.dtypes)

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nUkuran data sebelum encoding:", df.shape)
print("Ukuran data setelah encoding:", df_encoded.shape)

output_path = "output/cleaned_data.csv"
df_encoded.to_csv(output_path, index=False)
print(f"\nDataset clean disimpan di: {output_path}")


num_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
            'OrderAmountHikeFromlastYear', 'CouponUsed', 
            'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']

def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = len(df)
        print(f"{col}: removed {before - after} outliers")
    return df

df_no_outliers = remove_outliers_iqr(df_encoded, num_cols)
print("\nJumlah data setelah buang outlier:", df_no_outliers.shape)

scaler = MinMaxScaler()
df_no_outliers[num_cols] = scaler.fit_transform(df_no_outliers[num_cols])

print("\nContoh hasil normalisasi:")
print(df_no_outliers[num_cols].head())

corr = df_no_outliers.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr[['Churn']].sort_values(by='Churn', ascending=False), annot=True, cmap='coolwarm')
plt.title("Korelasi Fitur terhadap Churn")
plt.show()

X = df_no_outliers.drop(columns=['Churn', 'CustomerID'])
y = df_no_outliers['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nUkuran data train:", X_train.shape)
print("Ukuran data test:", X_test.shape)

X_train.to_csv("output/X_train.csv", index=False)
X_test.to_csv("output/X_test.csv", index=False)
y_train.to_csv("output/y_train.csv", index=False)
y_test.to_csv("output/y_test.csv", index=False)
