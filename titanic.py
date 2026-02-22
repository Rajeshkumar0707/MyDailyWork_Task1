import pandas as pd
# Load the dataset
df = pd.read_csv('titanic.csv')

#remove duplicates and handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

#print dataset information
print(df.info())

#print descriptive statistics
print(df.describe())

#sum of null values in each column
print(df.isnull().sum())

#sex and embarked columns to numeric values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print(df["Sex"].unique())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
print(df["Embarked"].unique())

#drop unnecessary columns and fill missing values with median for numeric columns and 0 for categorical columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
print(df.columns)
print(df.dtypes)
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)
x = df.drop("Survived", axis=1)
y = df["Survived"] 
print(x.head())
print(y.head())

#split the data into training and testing sets
from sklearn.model_selection import train_test_split
print(x.isnull().sum())
print("X columns",x.columns)
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
print("Training  size:", X_train.shape)
print("Test  size:", X_test.shape)

#train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000,solver='liblinear')
model.fit(X_train, y_train)
print("Model trained successfully.")

#evaluate the model and print the accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

#show the shape of the dataset
print(df.shape)