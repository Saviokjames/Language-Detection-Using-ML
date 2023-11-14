# -*- coding: utf-8 -*-
"""sp-3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HjsB8XF7IOgjWGE1S3eYfwdL7Nl6lylC
"""



"""# SPLITING THE DATA INTO TRAINING AND TEST SETS"""

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

model = MultinomialNB()

"""# MODEL TRAINING AND EVALUATION"""

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, y_train_pred) * 100
print(f"Training Set Accuracy: {accuracy_train:.2f}")

from sklearn.metrics import precision_score
precision_train = precision_score(y_train, y_train_pred, average='weighted') * 100
print(f"Training Set Precision: {precision_train:.4f}")

from sklearn.metrics import f1_score
f1_train = f1_score(y_train, y_train_pred, average='weighted') * 100
print(f"Training Set F1-Score: {f1_train:.2f}%")

from sklearn.metrics import classification_report
report_train = classification_report(y_train, y_train_pred)
print("Training Set Classification Report:")
print(report_train)

from sklearn.metrics import confusion_matrix
class_names = ["Arabic", "Chinese", "Dutch", "English", "Estonian", "French", "Hindi", "Indonesian", "Japanese", "Korean", "Latin", "Persian", "Portuguese", "Pushto", "Romanian", "Russian", "Spanish", "Swedish", "Tamil", "Thai", "Turkish", "Urdu"]
confusion_train = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusion_train, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Training Set Confusion Matrix')
plt.show()

"""#  MODEL TESTING AND EVALUATION"""

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)*100
print(f"Accuracy: {accuracy:.2f}")

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='weighted')*100
print(f"Precision: {precision:.4f}")

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted') * 100
print(f"F1-Score: {f1:.2f}%")

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

from sklearn.metrics import confusion_matrix
class_names = ["Arabic", "Chinese", "Dutch", "English", "Estonian", "French", "Hindi", "Indonesian", "Japanese", "Korean", "Latin", "Persian", "Portuguese", "Pushto", "Romanian", "Russian", "Spanish", "Swedish", "Tamil", "Thai", "Turkish", "Urdu"]
confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

"""#DOWNLOAD MODEL

"""

import joblib
model_filename = "multinomial_nb_model.pkl"  # Provide a filename
joblib.dump(model, model_filename)
from google.colab import files

files.download(model_filename)

cv_filename = "count_vectorizer_model.pkl"  # Provide a filename
joblib.dump(cv, cv_filename)
files.download(cv_filename)