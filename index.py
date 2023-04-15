# Import necessary libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle

# Load the dataset from a CSV file
data = pd.read_csv("student-mat.csv", sep=(";"))

# Select the columns of interest for the model
data = data[["G1","G2", 'G3', "studytime", "failures", "absences"]]

# Set the column to predict (output variable)
predict = 'G3'

# Split the data into input (x) and output (y) variables
x = data.drop([predict], axis=1).values
y = np.array(data[predict])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Train a linear regression model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# Calculate the accuracy score of the model on the test set
acc = linear.score(x_test, y_test)
print("Accuracy:", acc)

# Save the best model using pickle (commented out)
'''
best = 0
for i in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    
    print("Iteration", i+1, "Accuracy:", acc)
    
    if acc > best:
        best = acc
        with open('model.pkl', 'wb') as f:
            pickle.dump(linear, f)
            print("New best model saved with accuracy:", best)
            
print("Training complete. Best model accuracy:", best)
'''

# Load the best model using pickle
picke_in = open("model.pkl", "rb") 
linear = pickle.load(picke_in)

# Make predictions on the test set and print out the results
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])