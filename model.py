import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("C:/Users/ADMIN/Downloads/archive (5)/loan_approval_dataset.csv")

c = [" education"," self_employed"," loan_status"]

labelencoder = LabelEncoder()
for i in c:
    df[i] = labelencoder.fit_transform(df[i])
df.head(20)


# Assuming 'numerical_columns' is a list of column names you want to standardize
# numerical_columns = [" income_annum", " loan_amount", " loan_term", " cibil_score", " residential_assets_value", " commercial_assets_value", " luxury_assets_value", " bank_asset_value"]

# # Initialize the StandardScaler
# scaler = StandardScaler()

# # Standardize the selected columns
# df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# # Display the first few rows of the DataFrame
# df.head()
df.drop(columns = ['loan_id'], inplace = True)
y = df[' loan_status']
x = df.drop(columns = [' loan_status'])

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)


learning_rates = [1,0.4,0.5]
for i in learning_rates:
  cb = CatBoostClassifier(n_estimators=20,learning_rate=i)
  cb.fit(x_train, y_train)
  print("Learning rate: ", i)
  print("Accuracy score(training) : {0:3f}".format(cb.score(x_train, y_train)))
  print("Accuracy score(validation) : {0:3f}".format(cb.score(x_test, y_test)))
  print()
model_path = "catboost_model.cbm"
cb.save_model(model_path)
print("Model saved successfully at:", model_path)