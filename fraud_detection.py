### Imports

!pip install gradio
!kaggle datasets download "bhadramohit/credit-card-fraud-detection"
!unzip -o credit-card-fraud-detection.zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

### LogisticRegression

lrdf = pd.read_csv('credit_card_fraud_dataset.csv')
original = lrdf

# add hour and drop date since its more descriptive
lrdf['TransactionDate'] = pd.to_datetime(lrdf['TransactionDate'])
lrdf['Hour'] = lrdf['TransactionDate'].dt.hour

lrdf.drop(columns=['TransactionID', 'TransactionDate'], inplace=True)


# feature engineering to provide more meaningful relations between a given feature
# and its likelihood of being fraudulent

# (number of fraudulent transactions at hour i) / (total number of transactions at hour i)
# after add noise to make it less specific

# merchant fraud rate
merchant_fraud_rate = lrdf[lrdf['IsFraud'] == 1]['MerchantID'].value_counts() / lrdf['MerchantID'].value_counts()
lrdf['MerchantFraudRate'] = lrdf['MerchantID'].map(merchant_fraud_rate).fillna(0)

# hour fraud rate
hour_fraud_rate = lrdf[lrdf['IsFraud'] == 1]['Hour'].value_counts() / lrdf['Hour'].value_counts()
lrdf['HourFraudRate'] = lrdf['Hour'].map(hour_fraud_rate).fillna(0)

# location fraud rate
location_fraud_rate = lrdf[lrdf['IsFraud'] == 1]['Location'].value_counts() / lrdf['Location'].value_counts()
lrdf['LocationFraudRate'] = lrdf['Location'].map(location_fraud_rate).fillna(0)

# amount and fraud rate
amount_fraud_rate = lrdf[lrdf['IsFraud'] == 1]['Amount'].value_counts() / lrdf['Amount'].value_counts()
lrdf['AmountFraudRate'] = lrdf['Amount'].map(amount_fraud_rate).fillna(0)

# transaction type fraud rate
transaction_type_fraud_rate = lrdf[lrdf['IsFraud'] == 1]['TransactionType'].value_counts() / lrdf['TransactionType'].value_counts()
lrdf['TransactionTypeFraudRate'] = lrdf['TransactionType'].map(transaction_type_fraud_rate).fillna(0)

# encode categorical variables
lrdf = pd.get_dummies(lrdf, columns=['Location', 'TransactionType'])

# X to data, y to target
X = lrdf.drop('IsFraud', axis=1)
y = lrdf['IsFraud']

# split data 3:7
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

all_columns = X_train.columns.tolist()

# scale data for linear regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# make logistic regression model
clf = LogisticRegression(
    class_weight={0: 1, 1: 99}, # tell the model its 1:99 imbalanced
    random_state=42
)

clf.fit(X_train_scaled, y_train)

# evaluate
y_pred = clf.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

lrdf

### MLP

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

data = pd.read_csv('credit_card_fraud_dataset.csv')

data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
data['Hour'] = data['TransactionDate'].dt.hour
data.drop(columns=['TransactionID', 'TransactionDate'], inplace=True)

merchant_fraud_rate = data[data['IsFraud'] == 1]['MerchantID'].value_counts() / data['MerchantID'].value_counts()
hour_fraud_rate = data[data['IsFraud'] == 1]['Hour'].value_counts() / data['Hour'].value_counts()
location_fraud_rate = data[data['IsFraud'] == 1]['Location'].value_counts() / data['Location'].value_counts()
amount_fraud_rate = data[data['IsFraud'] == 1]['Amount'].value_counts() / data['Amount'].value_counts()
transaction_type_fraud_rate = data[data['IsFraud'] == 1]['TransactionType'].value_counts() / data['TransactionType'].value_counts()

data['MerchantFraudRate'] = data['MerchantID'].map(merchant_fraud_rate).fillna(0)
data['HourFraudRate'] = data['Hour'].map(hour_fraud_rate).fillna(0)
data['LocationFraudRate'] = data['Location'].map(location_fraud_rate).fillna(0)
data['AmountFraudRate'] = data['Amount'].map(amount_fraud_rate).fillna(0)
data['TransactionTypeFraudRate'] = data['TransactionType'].map(transaction_type_fraud_rate).fillna(0)

encoder = preprocessing.LabelEncoder()
data['TransactionType'] = encoder.fit_transform(data['TransactionType'])
data['Location'] = encoder.fit_transform(data['Location'])

X = data.drop(['IsFraud'], axis=1)
y = data['IsFraud']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

all_columns_mlp = X_train.columns.tolist()

smote = SMOTE(sampling_strategy=0.2, random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation='relu',
    solver='adam',
    max_iter=100,
    learning_rate_init=0.01,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

mlp.fit(x_train_balanced, y_train_balanced)

predictions = mlp.predict(x_test)
y_pred = mlp.predict(x_test)

print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

data


### XGBoost

xgdf = pd.read_csv('credit_card_fraud_dataset.csv')
original_xg = xgdf


xgdf.drop(['TransactionID'], axis= 1, inplace=True)
xgdf.drop(['TransactionDate'], axis= 1, inplace=True)

# added encoder
encoder = OneHotEncoder(drop='first')
encoded_arr = encoder.fit_transform(xgdf[['TransactionType', 'Location']]).toarray()
encoded_features = encoder.categories_

encoded_xgdf = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(['TransactionType', 'Location']))
xgdf = pd.concat([xgdf, encoded_xgdf], axis=1).drop(['TransactionType', 'Location'], axis=1)

X, y = xgdf.drop(['IsFraud'], axis= 1), xgdf['IsFraud']


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

all_columns_xg = X_train.columns.tolist()


xgboost_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42, eta=0.5, max_depth=8
)

xgboost_model.fit(X_train, y_train)

y_pred_xgboost = xgboost_model.predict(X_test)
y_pred_prob_xgboost = xgboost_model.predict_proba(X_test)[:, 1]

print('Classification Report:')
print(classification_report(y_test, y_pred_xgboost))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_xgboost))
print('Accuracy Score:', accuracy_score(y_test, y_pred_xgboost))
print('ROC-AUC Score:', roc_auc_score(y_test, y_pred_prob_xgboost))

xgdf

### Interface

def predict_fraud(Amount, MerchantID, TransactionType, Location, Hour, model):
    # create df to store params
    input_data = pd.DataFrame([{
        'Amount': Amount,
        'MerchantID': MerchantID,
        'Hour': Hour,
        f'TransactionType_{TransactionType}': 1,
        f'Location_{Location}': 1
    }])

    input_data['MerchantFraudRate'] = merchant_fraud_rate.get(MerchantID, 0)
    input_data['HourFraudRate'] = hour_fraud_rate.get(Hour, 0)
    input_data['LocationFraudRate'] = location_fraud_rate.get(Location, 0)
    input_data['AmountFraudRate'] = amount_fraud_rate.get(Amount, 0)
    input_data['TransactionTypeFraudRate'] = transaction_type_fraud_rate.get(TransactionType, 0)

    # process for LogisticRegression
    if model == 'LogisticRegression':
        for col in all_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[all_columns]

        input_data_scaled = scaler.transform(input_data)

        prediction = clf.predict(input_data_scaled)[0]
        proba = clf.predict_proba(input_data_scaled)[0][1]

    # process for XGBoost
    elif model == 'XGBoost':
        for col in all_columns_xg:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[all_columns_xg]

        prediction = xgboost_model.predict(input_data)[0]
        proba = xgboost_model.predict_proba(input_data)[0][1]

    result = "FRAUDULENT" if prediction == 1 else "LEGITIMATE"
    return f"{result}\nFraud Probability: {proba:.2f}"

def generate_and_predict(num_samples, model):
    print(f"Generating data for model: {model}")

    # generate synthetic transaction
    synthetic_data = pd.DataFrame({
        'Amount': np.round(np.random.uniform(original['Amount'].min(), original['Amount'].max(), num_samples), 2),
        'MerchantID': np.random.choice(original['MerchantID'].unique(), num_samples),
        'TransactionType': np.random.choice(['purchase', 'refund'], num_samples, p=[0.8, 0.2]),
        'Location': np.random.choice(original['Location'].unique(), num_samples),
        'Hour': np.random.choice(range(24), num_samples)
    })

    if model == 'LogisticRegression':
        # add engineered features
        synthetic_data['MerchantFraudRate'] = synthetic_data['MerchantID'].map(merchant_fraud_rate).fillna(0)
        synthetic_data['HourFraudRate'] = synthetic_data['Hour'].map(hour_fraud_rate).fillna(0)
        synthetic_data['LocationFraudRate'] = synthetic_data['Location'].map(location_fraud_rate).fillna(0)
        synthetic_data['AmountFraudRate'] = synthetic_data['Amount'].map(amount_fraud_rate).fillna(0)
        synthetic_data['TransactionTypeFraudRate'] = synthetic_data['TransactionType'].map(transaction_type_fraud_rate).fillna(0)

        # repeat encoding for logistic regression
        synthetic_data = pd.get_dummies(synthetic_data, columns=['Location', 'TransactionType'])
        synthetic_data = synthetic_data.reindex(columns=all_columns, fill_value=0)

        # scale data
        synthetic_data_scaled = scaler.transform(synthetic_data)

    elif model == 'XGBoost':
        # repeat xg boost encoding
        encoded_arr = encoder.transform(synthetic_data[['TransactionType', 'Location']]).toarray()
        encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(['TransactionType', 'Location']))
        synthetic_data = pd.concat([synthetic_data.drop(['TransactionType', 'Location'], axis=1), encoded_df], axis=1)
        synthetic_data = synthetic_data.reindex(columns=all_columns_xg, fill_value=0)

    # predict fraud for each transaction
    predictions = []
    for i in range(num_samples):
        if model == 'LogisticRegression':
            input_data = synthetic_data_scaled[i:i+1]
            prediction = clf.predict(input_data)[0]
            proba = clf.predict_proba(input_data)[0][1]
        elif model == 'XGBoost':
            input_data = synthetic_data.iloc[i:i+1].values
            prediction = xgboost_model.predict(input_data)[0]
            proba = xgboost_model.predict_proba(input_data)[0][1]

        result = "FRAUDULENT" if prediction == 1 else "LEGITIMATE"
        if prediction == 1:
            predictions.append(f"Transaction #{i + 1}:\nAmount: {synthetic_data.iloc[i]['Amount']} \nMerchantID: {synthetic_data.iloc[i]['MerchantID']}\n{result}\nFraud Probability: {proba:.2f}")

    return predictions

def gradio_generate_and_predict(num_samples, model):
    predictions = generate_and_predict(num_samples, model)
    return "\n\n".join(predictions)

def gradio_predict_fraud(amount, merchant_id, transaction_type, location, hour, model_choice):
    return predict_fraud(
        Amount=amount,
        MerchantID=merchant_id,
        TransactionType=transaction_type,
        Location=location,
        Hour=hour,
        model=model_choice
    )

# interface setup
with gr.Blocks() as demo:
    gr.Markdown("## Credit Card Fraud Detection")

    with gr.Row():
        gr.Markdown("### Predict Fraud for a Single Transaction")
        amount = gr.Number(label="Transaction Amount")
        merchant_id = gr.Number(label="Merchant ID")
        transaction_type = gr.Dropdown(
            choices=['purchase', 'refund'], label="Transaction Type"
        )
        location = gr.Dropdown(
            choices=['San Antonio', 'Dallas', 'Houston', 'New York', 'Philadelphia',
                     'Phoenix', 'San Diego', 'San Jose', 'Los Angeles'],
            label="Transaction Location"
        )
        hour = gr.Number(label="Transaction Hour (0-23)")
        model_choice = gr.Dropdown(
            choices=['LogisticRegression', 'XGBoost'],
            label="Model"
        )
        transaction_output = gr.Textbox(label="Transaction Prediction")
        predict_button = gr.Button("Predict")
        predict_button.click(
            gradio_predict_fraud,
            inputs=[amount, merchant_id, transaction_type, location, hour, model_choice],
            outputs=transaction_output
        )

    with gr.Row():
        gr.Markdown("### Generate and Predict Fraud for Synthetic Transactions")
        num_samples = gr.Number(label="Number of Synthetic Transactions", value=1000)
        model_choice = gr.Dropdown(
            choices=['LogisticRegression', 'XGBoost'],
            label="Model"
        )
        batch_output = gr.Textbox(label="Batch Prediction Results")
        generate_button = gr.Button("Generate and Predict")
        generate_button.click(
            gradio_generate_and_predict,
            inputs=[num_samples, model_choice],
            outputs=batch_output
        )

demo.launch()
