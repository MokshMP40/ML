1.    Load and explore dataset of excel file and csv file using pandas



Load and Explore CSV File: 
import pandas as pd 
# Load CSV file 
csv_data = pd.read_csv('train.csv')   
# Display the first few rows of the CSV file 
print("First few rows of CSV file:") 
print(csv_data.head()) 
# Summary statistics 
print("\nSummary statistics of CSV file:") 
print(csv_data.describe()) 
# Information about columns 
print("\nInformation about columns in CSV file:") 
print(csv_data.info()) 

Load and Explore Excel File: 
import pandas as pd 
# Load Excel file 
excel_data = pd.read_excel('Sample - Superstore.xlsx', sheet_name='Orders')   
# Display the first few rows of the Excel file 
print("First few rows of Excel file:") 
print(excel_data.head()) 
# Summary statistics 
print("\nSummary statistics of Excel file:") 
print(excel_data.describe()) 
# Information about columns 
print("\nInformation about columns in Excel file:") 
print(excel_data.info())




2.    Load and explore dataset of csv file using pandas


Load and Explore CSV File: 
import pandas as pd 
# Load CSV file 
csv_data = pd.read_csv('train.csv')   
# Display the first few rows of the CSV file 
print("First few rows of CSV file:") 
print(csv_data.head()) 
# Summary statistics 
print("\nSummary statistics of CSV file:") 
print(csv_data.describe()) 
# Information about columns 
print("\nInformation about columns in CSV file:") 
print(csv_data.info())



3.	Visualize dataset using matplotlib by plotting scatter plots and bar charts

import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_iris 
import pandas as pd 
# Load the Iris dataset 
iris = load_iris() 
# Convert the dataset to a pandas DataFrame 
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
iris_df['target'] = iris.target 
# Scatter plot using Matplotlib 
plt.figure(figsize=(8, 6)) 
plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], 
c=iris_df['target'], cmap='viridis', s=80, alpha=0.7) 
plt.xlabel('Sepal Length (cm)') 
plt.ylabel('Sepal Width (cm)') 
plt.title('Scatter Plot of Sepal Length vs Sepal Width') 
plt.colorbar(label='Species') 
plt.show() 
# Bar chart using Seaborn 
plt.figure(figsize=(8, 6)) 
sns.countplot(x='target', data=iris_df, palette='viridis') 
plt.xlabel('Species') 
plt.ylabel('Count') 
plt.title('Bar Chart: Count of Each Species') 
plt.show()


4.	Program to handle missing data. 
	 Mean/Median/mode imputations. 
	 Using forward fill method. 
	 Using backward fill method. 


import pandas as pd
from sklearn.impute import SimpleImputer
data = {'Age': [25, 20, None, 35, 40],'Salary': [50000, None, 65000, 70000, 80000]  # Fixed '8000077' to '80000'}
df = pd.DataFrame(data)
print("Data Frame before imputing missing values using mean:\n", df)
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])
print("\nData Frame after imputing missing values using mean:\n", df)



import pandas as pd
from sklearn.impute import SimpleImputer
data = {'Age': [25, 30, None, 35, 40],'Salary': [50000, None, 65000, 70000, 80000]}
df = pd.DataFrame(data)
print("Data Frame before forward fill method:\n", df)
df.fillna(method='ffill', inplace=True)
print("\nData Frame after forward fill method:\n", df)



import pandas as pd  
data = {'Age': [25, None, 35, 40],'Salary': [50000, None, 65000, 70000]}  
df = pd.DataFrame(data)  
print("Data Frame before backward fill method:\n", df)  
df.fillna(method='bfill', inplace=True)  
print("\nData Frame after backward fill method:\n", df)




5.	Program to handle missing data using encoded categorical variables. 
	 One hot encoding. 
	 Using label encoding. 



import pandas as pd  
from sklearn.preprocessing import OneHotEncoder  
data = {'City': ['Mandya', 'Bengaluru', 'India', 'Durga', 'Jalandhar', 'Tumakuru']}  
df = pd.DataFrame(data)  
print("Data Frame before One Hot Encoding:\n", df)  
encoder = OneHotEncoder(sparse_output=False)  
encoded = encoder.fit_transform(df[['City']])  
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['City']))  
print("\nData Frame after One Hot Encoding:\n", encoded_df)




import pandas as pd  
from sklearn.preprocessing import LabelEncoder  
df = pd.DataFrame({'Gender': ['Male', 'Female', 'Female', 'Male', 'Female']})  
print("Data Frame before Label Encoding:\n", df)  
encoder = LabelEncoder()  
df['Gender'] = encoder.fit_transform(df['Gender'])  
print("\nData Frame after Label Encoding:\n", df)




6.	Perform feature scaling.  
	 Using standardization. 
	 Min-Max scaling. 

import pandas as pd  
from sklearn.preprocessing import StandardScaler  
data = {'Age': [25, 30, 35, 40, 45],'Salary': [50000, 60000, 65000, 70000, 36000]}  
df = pd.DataFrame(data)  
print("Feature Scaling DF before standardization:\n", df)  
scaler = StandardScaler()  
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  
print("\nFeature Scaling DF after standardization:\n", df_scaled)




import pandas as pd  
from sklearn.preprocessing import MinMaxScaler  
data = {'Age': [25, 30, 35, 40, 45],'Salary': [50000, 60000, 65000, 70000, 36000]}  
df = pd.DataFrame(data)  
print("Feature Scaling DF before Min-Max Scaling:\n", df)  
scaler = MinMaxScaler()  
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  
print("\nFeature Scaling DF after Min-Max Scaling:\n", df_scaled)





7.	Implementing KNN classifier using scikit-learn.



from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 
 
# Load the Iris dataset (or any other dataset you want to use) 
iris = load_iris() 
X = iris.data 
y = iris.target 
 
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=42) 
 
# Initialize the k-NN classifier 
k = 3  # Set the number of neighbors 
knn_classifier = KNeighborsClassifier(n_neighbors=k) 
 
# Train the classifier on the training data 
knn_classifier.fit(X_train, y_train) 
 
# Make predictions on the testing data 
predictions = knn_classifier.predict(X_test) 
 
# Evaluate the performance of the classifier 
accuracy = metrics.accuracy_score(y_test, predictions) 
print(f"Accuracy: {accuracy}") 
 
# You can also print other evaluation metrics if needed 
# For example, classification report and confusion matrix 
print("Classification Report:") 
print(metrics.classification_report(y_test, predictions)) 
print("Confusion Matrix:") 
print(metrics.confusion_matrix(y_test, predictions))





8.	Implementing a linear regression model for regression tasks. 


import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 
 
# Generate a synthetic dataset 
X,y = make_regression(n_samples=1000, n_features=1, noise=20, 
random_state=42) 
 
# Split the dataset into training and testing sets 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, 
random_state=42) 
 
# Initialize the Linear Regression model 
linear_reg = LinearRegression() 
 
# Train the model on the training data 
linear_reg.fit(X_train, y_train) 
 
# Make predictions on the testing data 
predictions = linear_reg.predict(X_test) 
 
# Evaluate the model's performance 
mse = mean_squared_error(y_test, predictions) 
r2 = r2_score(y_test, predictions) 
print("Mean Squared Error (MSE):\n",mse) 
print("R-squared:\n",r2) 
# Plotting the regression line (optional) 
plt.scatter(X_test, y_test, color='blue') 
plt.plot(X_test, predictions, color='red', linewidth=3) 
plt.xlabel('X') 
plt.ylabel('y') 
plt.title('Linear Regression') 
plt.show() 




9.	Implementing decision-tree classifier using scikit-learn. 


from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
import matplotlib.pyplot as plt 
 
# Load the Iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
class_names = [str(name) for name in iris.target_names]   
 
# Initialize the Decision Tree Classifier 
decision_tree = DecisionTreeClassifier() 
 
# Train the classifier on the entire dataset 
decision_tree.fit(X, y) 
 
# Visualize the Decision Tree 
plt.figure(figsize=(12, 8)) 
plot_tree(decision_tree, feature_names=iris.feature_names, 
class_names=class_names, filled=True, rounded=True) 
plt.title("Decision Tree Visualization") 
plt.show() 


10.	Implementing K-means clustering and visualize clusters.


import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans 
# Generating synthetic data 
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, 
random_state=42) 
# Initialize K-Means with the number of clusters 
kmeans = KMeans(n_clusters=4) 
# Fit the K-Means model to the data 
kmeans.fit(X) 
# Predict cluster labels 
cluster_labels = kmeans.predict(X) 
# Visualize the clusters 
plt.figure(figsize=(7,5)) 
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
marker='o', s=200, color='red', label='Centroids') 
plt.title('K-Means Clustering') 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.legend() 
plt.show()





