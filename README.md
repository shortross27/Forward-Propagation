# Churn Prediction Using Artificial Neural Networks

## Project Overview
This project aims to predict customer churn using an Artificial Neural Network (ANN). The model was trained on a dataset containing customer information, and various visualizations were created to evaluate and interpret the model's performance.

The project involves:
1. Data preprocessing, including feature encoding and scaling.
2. Building and training a neural network using TensorFlow/Keras.
3. Evaluating the model's performance with metrics like accuracy and loss.
4. Visualizing results to gain insights into the model and the dataset.

## Dataset
The dataset used is Churn_Modelling.csv, which includes:

- Customer demographic and account details.
- A binary target variable indicating churn (1 = Churn, 0 = Not Churn).

## Technologies Used
- Python
- Pandas and NumPy for data manipulation.
- TensorFlow/Keras for building the ANN.
- Matplotlib and Seaborn for visualizations.
- Scikit-learn for preprocessing and evaluation metrics.

## Data Preprocessing
**Steps:**
1. Encoding categorical variables:
  - Gender (binary encoding).
  - Country (one-hot encoding).
2. Splitting the dataset into training and test sets.
3. Standardizing features to improve model performance.

```python
#Example preprocessing code snippet
X = data.iloc[:, 3:-1].values
Y = data.iloc[:, -1].values

#Encoding categorical variables
LE1 = LabelEncoder()
X[:, 2] = np.array(LE1.fit_transform(X[:, 2]))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
```

## Model Training
A feedforward ANN was built with:

- Two hidden layers using ReLU activation.
- A sigmoid-activated output layer for binary classification.

```python
# Model architecture
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation="relu"),
    tf.keras.layers.Dense(units=6, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

# Compiling and training the ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
ann.fit(X_train, Y_train, batch_size=32, epochs=10)
```

## Visualizations
Several visualizations were created to evaluate the model:

**1. Loss and Accuracy Plots**
Displays the model's training progress over epochs.

```python
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()
```

![Training and Validation Loss](https://github.com/user-attachments/assets/daf2c395-490f-40ba-b6f2-c7e1670c191a)

![Training and Validation Accuracy](https://github.com/user-attachments/assets/cb7948f1-7042-4391-8c95-fb5dd7640a60)

**2. Confusion Matrix**
Shows the breakdown of correct and incorrect predictions.

```python
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Churn', 'Churn'])
disp.plot(cmap=plt.cm.Blues)
```

![Confusion Matrix](https://github.com/user-attachments/assets/99a6ad07-6e2e-4043-880a-400e4df392d7)


**3. ROC Curve**
Highlights the model's ability to distinguish between classes.

```python
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
```

![ROC Curve](https://github.com/user-attachments/assets/31d3eca2-b37b-428a-aced-a8fe8232e4dc)

**4. Class Distribution**
Explores class imbalances in the dataset.

```python
sns.countplot(x=Y, palette='viridis')
```

![Class Distribution](https://github.com/user-attachments/assets/e68a2263-c6d4-42cc-966e-56b3a7d97dd9)


## Setup Instructions
**Prerequisites:**
- Python 3.8 or higher.
- Install required packages:
```python
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

## Results
**1. Model Accuracy**
- The ANN achieved **85% accuracy** on the training set and **83.5% accuracy** on the validation set after 10 epochs, as shown in the **Training and Validation Accuracy Plot**.
- The accuracy trends indicate that the model learned effectively without significant overfitting.

**2. Confusion Matrix**
- The confusion matrix highlights the performance of the model in predicting churn versus non-churn cases:
  - **True Negatives (Not Churn predicted correctly):** 147
  - **True Positives (Churn predicted correctly):** 20
  - **False Positives (Not Churn predicted as Churn):** 5
  - **False Negatives (Churn predicted as Not Churn):** 28
- These results demonstrate that the model is particularly strong at predicting customers who will not churn, while there is room for improvement in identifying customers who are likely to churn.

**3. ROC Curve**
- The **Receiver Operating Characteristic (ROC) Curve** has an **AUC (Area Under Curve) of 0.92**, indicating excellent model performance.
The high AUC score reflects the model's ability to distinguish between customers likely to churn and those who are not, with a strong balance between true positive and false positive rates.

## Key Insights:
The ANN effectively separates churn and non-churn cases, achieving strong overall accuracy and reliability.
- **Strengths:** High precision in predicting non-churn cases.
- **Challenges:** Higher false negatives (28) suggest the model could benefit from further tuning to improve churn prediction accuracy.
