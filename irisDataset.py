from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#loading the iris dataset
iris=load_iris()
# print(iris)

#feature
X=iris.data
# print(X)

#label
y=iris.target
# print(y)

#split the data into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=2,random_state=42)

#train a supervised learning model with this training data
model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

#test the model with the test data
y_pred=model.predict(X_test)
print(y_pred)


#finding accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy is:",accuracy)

#find confusion matrix

cm=confusion_matrix(y_test,y_pred)
print("confusion matrix is",cm)

#print a classification report icluding f1 score etc
report=classification_report(y_test,y_pred)
print(report)
