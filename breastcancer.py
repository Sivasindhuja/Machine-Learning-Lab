from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#load the dataset
cancer=load_breast_cancer()

#features
X=cancer.data
#labels
y=cancer.target

#split into test and train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#train model-->KNN

#let k=1
k=30
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)

#test the model against the test data
y_pred=knn.predict(X_test)

#find accuracy
accuracy=accuracy_score(y_test,y_pred)
print("for k=30",accuracy)
