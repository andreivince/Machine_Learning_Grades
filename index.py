import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=(";"))
data = data[["G1","G2", 'G3', "studytime", "failures", "absences"]]
predict = 'G3'

x = data.drop([predict], axis=1).values
y = np.array(data[predict]) #Getting all our labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)

predictions = linear.predict(x_test)

print(acc)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    