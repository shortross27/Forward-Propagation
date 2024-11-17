import numpy as np
import pandas as pd 
import seaborn as sns
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

def forward_propagation(X): 
    # Define the weights and biases 
    W1 = np.array([[0.1, 0.2, 0.3, 0.4], 
                   [0.5, 0.6, 0.7, 0.8], 
                   [0.9, 1.0, 1.1, 1.2]]) 
    
    b1 = np.array([0.1, 0.2, 0.3, 0.4]) 
    
    W2 = np.array([[0.5, 0.6],
                   [0.7, 0.8], 
                   [0.9, 1.0], 
                   [1.1, 1.2]]) 
    
    b2 = np.array([0.5, 0.6]) 
    Z1 = np.dot(X, W1) + b1 
    A1 = np.maximum(0, Z1) # ReLU activation 
    
    Z2 = np.dot(A1, W2) + b2 
    exp_scores = np.exp(Z2) 
    A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # Softmax activation 
    
    return A2

# Load Dataset
file_path = 'Churn_Modelling.csv'
data = pd.read_csv(file_path)

# Displaying the loaded data 
print(data.head())
print(data.info())

#Generating Matrix of Features (X) 
X = data.iloc[:, 3:-1].values

#Generating Dependent Variable Vector(Y) 
Y = data.iloc[:, -1].values

#Encoding Categorical Variable Gender 
LE1 = LabelEncoder() 
X[:, 2] = np.array(LE1.fit_transform(X[:, 2]))

#Encoding Categorical Variable Country 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough") 
X = np.array(ct.fit_transform(X))

#Splitting Dataset into Training and Testing Dataset 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=200, random_state=0)

#Performing Feature Scaling 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

#Initializing Artificial Neural Network 
ann = tf.keras.models.Sequential()

#Adding First Hidden Layer 
ann.add(tf.keras.layers.Dense(units=6, activation="relu")) 

#Adding Second Hidden Layer 
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

#Adding Output Layer 
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

#Compiling ANN 
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

#Fitting ANN 
ann.fit(X_train, Y_train, batch_size=32, epochs=10)


#Predicting result for Single Observation 
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])) > 0.5)


# Train the ANN and store the history
history = ann.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.2)

# Extract loss and accuracy
loss = history.history['loss']
accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']


# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss', linestyle='--')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy', linestyle='--')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict on the test set
Y_pred = (ann.predict(X_test) > 0.5).astype(int)


# Generate confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Churn', 'Churn'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# Generate ROC curve
Y_pred_prob = ann.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# Generate Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred_prob)

# Plot the Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='purple')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# Class Distribution
sns.countplot(x=Y, palette='viridis')
plt.title('Class Distribution')
plt.xlabel('Class (0 = Not Churn, 1 = Churn)')
plt.ylabel('Count')
plt.show()


