import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,precision_score,recall_score,f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

diabetes = pd.read_csv("diabetes.csv")

df = diabetes.copy()

print(df)

print(df.isna().sum())

print(df.info())

df['Outcome'] = df['Outcome'].astype('category')

df.info()

print(df['Outcome'].value_counts())

#print(df['Outcome'].value_counts().plot.pie())

#print(df['Outcome'].value_counts().plot.bar())

print(df.describe().T)

sns.boxplot(x='Pregnancies', data=df)

y = df['Outcome']
x = df.drop(['Outcome'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model = LogisticRegression(max_iter=200)

model.fit(x_train,y_train)

print(model.intercept_)

print(model.coef_)

y_pred = model.predict(x_test)

print(y_pred)

#Model Performance Calculation

dfP = pd.DataFrame(confusion_matrix(y_test, y_pred),columns=['Predicted Positive', 'Predicted Negative'], index=['Actual Positive', 'Actual Negative'])

print(dfP)

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test, y_pred))

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))

model.predict_proba(x_test)[0:10]


#ROC(Receiver Operating Characteristic) & AUC(Area Under the Curve)

modelRocAuc = roc_auc_score(y_test, model.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label='AUC (area = %0.2f)' % modelRocAuc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim(([0.0, 1.0]))
plt.ylim(([0.0, 1.05]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

