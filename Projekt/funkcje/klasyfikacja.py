import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from funkcje import czyszczenie_tekstu

pozytywne = pd.read_csv('../Data/pozytywne.csv', usecols=['Review'])
pozytywne['Sentyment'] = 'Pozytywny'
neutralne = pd.read_csv('../Data/neutralne.csv', usecols=['Review'])
neutralne['Sentyment'] = 'Neutralny'
negatywne = pd.read_csv('../Data/negatywne.csv', usecols=['Review'])
negatywne['Sentyment'] = 'Negatywny'

calosc = pd.concat([pozytywne, neutralne, negatywne])

X = calosc['Review']
y = calosc['Sentyment']

X_train, X_test, y_train, y_test = train_test_split(calosc['Review'], calosc['Sentyment'], test_size=0.2,
                                                    shuffle=True, stratify=calosc['Sentyment'])

vectorizer = CountVectorizer(tokenizer=czyszczenie_tekstu.tokenizowanie_tekstu)
X_train_transform = vectorizer.fit_transform(X_train)
X_test_transform = vectorizer.transform(X_test)


def heatmap(y_test, y_pred, classifier):
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, annot=True, cmap="RdBu_r", fmt='g', cbar=False),
    ax.xaxis.set_ticklabels(['Pozytywny', 'Neutralny', 'Negatywny'])
    ax.yaxis.set_ticklabels(['Pozytywny', 'Neutralny', 'Negatywny'])
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(f'{classifier} Confusion Matrix')
    plt.show()


def bagg_class():
    bc_model = BaggingClassifier()
    bc_model.fit(X_train_transform, y_train)
    bc_predykcja = bc_model.predict(X_test_transform)
    print(f'BaggingClassifier \n {classification_report(y_test, bc_predykcja)}')
    heatmap(y_test, bc_predykcja, "BaggingClassifier")


def rfc_class():
    rfc = RandomForestClassifier()
    rfc.fit(X_train_transform, y_train)
    rfc_predykcja = rfc.predict(X_test_transform)
    print(f'RandomForestClassifier \n {classification_report(y_test, rfc_predykcja)}')
    heatmap(y_test, rfc_predykcja, "RandomForestClassifier")


def svm_class():
    svm = SVC()
    svm.fit(X_train_transform, y_train)
    svm_predykcja = svm.predict(X_test_transform)
    print(f'SVM  \n {classification_report(y_test, svm_predykcja)}')
    heatmap(y_test, svm_predykcja, "SVM")


def dt_class():
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(X_train_transform, y_train)
    dtc_predykcja = dtc.predict(X_test_transform)
    print(f'DecisionTreeClassifier \n {classification_report(y_test, dtc_predykcja)}')
    heatmap(y_test, dtc_predykcja, "DecisionTreeClassifier")


def ab_class():
    ab = AdaBoostClassifier()
    ab = ab.fit(X_train_transform, y_train)
    ab_predykcja = ab.predict(X_test_transform)
    print(f'AdaBoostClassifier \n {classification_report(y_test, ab_predykcja)}')
    heatmap(y_test, ab_predykcja, "AdaBoostClassifier")


def log_reg():
    log_r = LogisticRegression(max_iter=3000)
    log_r.fit(X_train_transform, y_train)
    log_r_predykcja = log_r.predict(X_test_transform)
    print(f'Logistic Regression \n {classification_report(y_test, log_r_predykcja)}')
    heatmap(y_test, log_r_predykcja, "Logistic Regression")
