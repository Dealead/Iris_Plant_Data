import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris_data = pd.read_excel('C:/Users/mowab/Downloads/iris data 2.xlsx')
print(iris_data.head())

x = iris_data.iloc[:, :-1].values
y = iris_data.iloc[:, 4].values



sns.countplot(iris_data['Class of Plant'])
#plt.show()
sns.pairplot(iris_data, hue="Class of Plant", markers=["D", "o", "s"])
#plt.show()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(cm)
print("\n The Accuracy Score is: {} %" .format(score*100))

print("\n CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))


sns.heatmap(cm)
plt.show()

