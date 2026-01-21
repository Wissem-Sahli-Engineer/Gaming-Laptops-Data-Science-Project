import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("gaming_laptops2_cleaned.csv")

def get_name(name):
    if "msi" in name.lower():
        return "MSI"
    elif "asus" in name.lower():
        return "ASUS"
    elif "lenovo" in name.lower():
        return "LENOVO"
    elif "acer" in name.lower():
        return "ACER"
    elif "hp" in name.lower():
        return "HP"
    elif "dell" in name.lower():
        return "DELL"
    elif "gigabyte" in name.lower():
        return "GIGABYTE"
    elif "razer" in name.lower():
        return "RAZER"
    elif " Cooler Master" in name.lower():
        return "COOLER MASTER"
    else:
        return "OTHER"

# converting Ram  and storage values to numeric values 
df['RAM_GB'] = pd.to_numeric(df['RAM'].str.extract(r'(\d+)')[0], errors='coerce').fillna(8).astype(int)
df['Storage_GB'] = pd.to_numeric(df['Storage'].str.extract(r'(\d+)')[0], errors='coerce').fillna(512).astype(int)

# getting brand names from name column  
df['Brand'] = df['Name'].apply(get_name)

# creating a new dataframe with only the columns we need
df_model = df[['Brand', 'CPU', 'GPU', 'RAM_GB', 'Storage_GB', 'Price', 'Garentie', 'Availability']].copy()

# converting availability to binary values
df_model['Availability'] = df_model['Availability'].apply(lambda x: 1 if x == 'En stock' else 0)

# converting categorical variables to dummy variables
# drop_first=True is used to avoid multicollinearity

df_dummy = pd.get_dummies(df_model, columns=['Brand', 'CPU', 'GPU'],drop_first=True)

# train test split 

X= df_dummy.drop('Price', axis=1)
y= df_dummy['Price']
X_train, X_test, y_train, y_test = train_test_split (X , y , test_size = 0.20 , random_state = 42 ) 

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
lr_preds = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, lr_preds)
r2 = r2_score(y_test, lr_preds)

print(f"Average Error - Mean Absolute Error (MAE): {mae}")
print(f"Model Accuracy (R2 Score): {r2}")

# Random Forest is usually much more accurate for tech specs
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
print(f"Random Forest MAE: {mean_absolute_error(y_test, rf_preds):.2f} DT")

'''
# See which specs (GPU, RAM, etc.) impact the price the most
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Factors Driving Laptop Prices")
plt.show()
'''

'''
df.boxplot(column='Price', by='Brand')
plt.show()
'''

# 1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)

# 2. Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# 3. Define the Bounds
# Anything below Lower or above Upper is an outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 4. Filter the data
df_cleaned = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

print(f"Original rows: {len(df)}")
print(f"Rows after removing outliers: {len(df_cleaned)}")


