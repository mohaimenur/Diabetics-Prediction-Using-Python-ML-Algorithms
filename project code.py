import numpy as np
import pandas as pd
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as mtp





# data preprocessing

data_set=pd.read_csv("Diabetes Prediction Dataset.csv")
x=data_set.iloc[:,:-1]
y=data_set.iloc[:,8]



# train and test data devide & initiate the test size
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=0)


# data scaling
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.fit_transform(x_test)







# applying 5 classification algorithms



# logistic regression
lr=LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)

#print(y_pred_lr)

print("\n")
print("Result of Logistic Regression: \n")
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test,y_pred_lr))
print("Accuracy: \n", metrics.accuracy_score(y_test,y_pred_lr)*100)
print("Classification Report: \n", metrics.classification_report(y_test,y_pred_lr))

# roc curve for logistic regression
y_pred_proba_lr=lr.predict_proba(x_test)[:,1]
AUCLR=metrics.roc_auc_score(y_test,y_pred_proba_lr)
print("AUC Score: \n", AUCLR)
fprLR, tprLR,thresholdvalues=metrics.roc_curve(y_test,y_pred_proba_lr)
mtp.plot([0,1],[0,1], color="r", linestyle="dotted")
mtp.plot(fprLR,tprLR,label="Logistic Regression(area="+str(AUCLR)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()





# support vector machine(SVM)
sv=SVC(probability=(True))
sv.fit(x_train,y_train)
y_pred_sv=sv.predict(x_test)

#print(y_pred_sv)

print("\n")
print("Result of SVM: \n")
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test,y_pred_sv))
print("Accuracy: \n", metrics.accuracy_score(y_test,y_pred_sv)*100)
print("Classification Report: \n", metrics.classification_report(y_test,y_pred_sv))

# roc curve for support vector machine
y_pred_proba_sv=sv.predict_proba(x_test)[:,1]
AUCSV=metrics.roc_auc_score(y_test,y_pred_proba_sv)
print("AUC Score: \n", AUCSV)
fprSV, tprSV,thresholdvalues=metrics.roc_curve(y_test,y_pred_proba_sv)
mtp.plot([0,1],[0,1], color="r", linestyle="dotted")
mtp.plot(fprSV,tprSV,label="Support Vector(area="+str(AUCSV)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()





# random forest
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)

#print(y_pred_rf)

print("\n")
print("Result of Random Forest: \n")
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test,y_pred_rf))
print("Accuracy: \n", metrics.accuracy_score(y_test,y_pred_rf)*100)
print("Classification Report: \n", metrics.classification_report(y_test,y_pred_rf))

# roc curve for random forest
y_pred_proba_rf=rf.predict_proba(x_test)[:,1]
AUCRF=metrics.roc_auc_score(y_test,y_pred_proba_rf)
print("AUC Score: \n", AUCRF)
fprRF, tprRF,thresholdvalues=metrics.roc_curve(y_test,y_pred_proba_rf)
mtp.plot([0,1],[0,1], color="r", linestyle="dotted")
mtp.plot(fprRF,tprRF,label="Random Forest(area="+str(AUCRF)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()





# k-nearest neighbors
kn=KNeighborsClassifier(n_neighbors=2)
kn.fit(x_train,y_train)
y_pred_kn=kn.predict(x_test)

#print(y_pred_kn)

print("\n")
print("Result of K Nearest Neighbors: \n")
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test,y_pred_kn))
print("Accuracy: \n", metrics.accuracy_score(y_test,y_pred_kn)*100)
print("Classification Report: \n", metrics.classification_report(y_test,y_pred_kn))

# roc curve for k-nearest neighbors
y_pred_proba_kn=kn.predict_proba(x_test)[:,1]
AUCKN=metrics.roc_auc_score(y_test,y_pred_proba_kn)
print("AUC Score: \n", AUCKN)
fprKN, tprKN,thresholdvalues=metrics.roc_curve(y_test,y_pred_proba_kn)
mtp.plot([0,1],[0,1], color="r", linestyle="dotted")
mtp.plot(fprKN,tprKN,label="K-nearest neighbors(area="+str(AUCKN)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()





# AdaBoost
ab=AdaBoostClassifier(n_estimators=100)
ab.fit(x_train,y_train)
y_pred_ab=ab.predict(x_test)

#print(y_pred_ab)

print("\n")
print("Result of AdaBoost Classifier: \n")
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test,y_pred_ab))
print("Accuracy: \n", metrics.accuracy_score(y_test,y_pred_ab)*100)
print("Classification Report: \n", metrics.classification_report(y_test,y_pred_ab))

# roc curve for adaboost
y_pred_proba_ab=ab.predict_proba(x_test)[:,1]
AUCAB=metrics.roc_auc_score(y_test,y_pred_proba_ab)
print("AUC Score: \n", AUCAB)
fprAB, tprAB,thresholdvalues=metrics.roc_curve(y_test,y_pred_proba_ab)
mtp.plot([0,1],[0,1], color="r", linestyle="dotted")
mtp.plot(fprAB,tprAB,label="AdaBoost(area="+str(AUCAB)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()







