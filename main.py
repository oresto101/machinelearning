import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def load_data(file_path):
    column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Glass_Type']
    df = pd.read_csv(file_path, names=column_names)
    return df


def split_data(df, test_size=0.3, random_state=42):
    x = df.drop('Glass_Type', axis=1)
    y = df['Glass_Type']
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val


def normalize_data(X_train, X_val):
    scaler = Normalizer()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    return X_train_normalized, X_val_normalized


def standardize_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_val_standardized = scaler.transform(X_val)
    return X_train_standardized, X_val_standardized


def apply_pca(X_train, X_val, n_components=2):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_val_pca


def train_and_evaluate_by_accuracy(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    return accuracy_score(y_val, predictions)


def train_and_evaluate_by_precision(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    return precision_score(y_val, predictions, average='macro', zero_division=1)


# def train_and_evaluate_by_recall(model, X_train, X_val, y_train, y_val):
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_val)
#     return recall_score(y_val, predictions)


def main():
    np.random.seed(42)
    df = load_data('glass.csv')
    print('Data Mining\n', df.describe())
    X_train, X_val, y_train, y_val = split_data(df)
    X_train_normalized, X_val_normalized = normalize_data(X_train, X_val)
    X_train_standardized, X_val_standardized = standardize_data(X_train, X_val)
    X_train_pca, X_val_pca = apply_pca(X_train, X_val)
    nb = GaussianNB()
    unprocessed = [train_and_evaluate_by_accuracy(nb, X_train, X_val, y_train, y_val),
                     train_and_evaluate_by_precision(nb, X_train, X_val, y_train, y_val)]
    normalized = [train_and_evaluate_by_accuracy(nb, X_train_normalized, X_val_normalized, y_train, y_val),
                     train_and_evaluate_by_precision(nb, X_train_normalized, X_val_normalized, y_train, y_val)]
    standardized =[train_and_evaluate_by_accuracy(nb, X_train_standardized, X_val_standardized, y_train, y_val),
                     train_and_evaluate_by_precision(nb, X_train_standardized, X_val_standardized, y_train, y_val)]
    pca = [train_and_evaluate_by_accuracy(nb, X_train_pca, X_val_pca, y_train, y_val),
                     train_and_evaluate_by_precision(nb, X_train_pca, X_val_pca, y_train, y_val)]
    print("Accuracy, Precision evaluation of unprocessed, normalized, standardized, pca-processed datasets using Naive-Bayes classifier")
    print(unprocessed)
    print(normalized)
    print(standardized)
    print(pca)
    dt = DecisionTreeClassifier()
    dt_params = [
        {"max_depth": None, "min_samples_split": 2},
        {"max_depth": 2, "min_samples_split": 2},
        {"max_depth": 5, "min_samples_split": 5},
    ]
    for item in dt_params:
        dt.set_params(**item)
        unprocessed = [train_and_evaluate_by_accuracy(dt, X_train, X_val, y_train, y_val),
                       train_and_evaluate_by_precision(dt, X_train, X_val, y_train, y_val)]
        normalized = [train_and_evaluate_by_accuracy(dt, X_train_normalized, X_val_normalized, y_train, y_val),
                      train_and_evaluate_by_precision(dt, X_train_normalized, X_val_normalized, y_train, y_val)]
        standardized = [train_and_evaluate_by_accuracy(dt, X_train_standardized, X_val_standardized, y_train, y_val),
                        train_and_evaluate_by_precision(dt, X_train_standardized, X_val_standardized, y_train, y_val)]
        pca = [train_and_evaluate_by_accuracy(dt, X_train_pca, X_val_pca, y_train, y_val),
               train_and_evaluate_by_precision(dt, X_train_pca, X_val_pca, y_train, y_val)]
        print("Accuracy, Precision evaluation of unprocessed, normalized, standardized, pca-processed datasets using Decision Tree classifier")
        print(f"Hyperparameters: {item}")
        print(unprocessed)
        print(normalized)
        print(standardized)
        print(pca)


if __name__ == "__main__":
    main()
