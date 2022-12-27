import os
import datetime as dt
import pandas as pd
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier


# базовые аргументы DAG
args = {
    'owner': 'airflow',  # Информация о владельце DAG
    'start_date': dt.datetime(2020, 12, 23),  # Время начала выполнения пайплайна
    'retries': 1,  # Количество повторений в случае неудач
    'retry_delay': dt.timedelta(minutes=1),  # Пауза между повторами
}


def extract_dataset():
    
    url = "https://raw.githubusercontent.com/Murcha1990/churn_clients/main/ClientsData.csv"
    data = pd.read_csv(url)
    data.to_csv("churn_clients.csv",index=False)
    

def transform_dataset():
    
    data = pd.read_csv("churn_clients.csv")
    
    X = data.drop('TARGET', axis=1)
    y = data['TARGET']

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)
    
    ss = StandardScaler()
    ss.fit(Xtrain)

    Xtrain = pd.DataFrame(ss.transform(Xtrain), columns=X.columns)
    Xtest = pd.DataFrame(ss.transform(Xtest), columns=X.columns)
    
    Xtrain.to_csv("Xtrain.csv",index=False)
    Xtest.to_csv("Xtest.csv",index=False)
    ytrain.to_csv("ytrain.csv",index=False)
    ytest.to_csv("ytest.csv",index=False)
    

def apply_model(model=LogisticRegression()):
    
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv")
    
    model.fit(Xtrain, ytrain)
    probs = model.predict_proba(Xtest)[:,1]
    
    Xtest['res'] = probs
    
    Xtest[['res']].to_csv("probs.csv",index=False)
    


def evaluate_model(threshold):

    ytest = pd.read_csv("ytest.csv")
    probs = pd.read_csv("probs.csv")

    classes = probs['res'] > threshold
    
    print('Accuracy:', accuracy_score(ytest, classes))
    print('Confusion matrix:', confusion_matrix(ytest, classes))
    print('Recall:', recall_score(ytest, classes))

    
dag = DAG(
    dag_id='churn_clients',  # Имя DAG
    schedule_interval=None,  # Периодичность запуска, например, "00 15 * * *"
    default_args=args,  # Базовые аргументы
)

extract_dataset = PythonOperator(
    task_id='extract_dataset',
    python_callable=extract_dataset,
    dag=dag,
)

transform_dataset = PythonOperator(
    task_id='transform_dataset',
    python_callable=transform_dataset,
    dag=dag,
)

apply_model = PythonOperator(
    task_id='apply_model',
    python_callable=apply_model,
    dag=dag,
)

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    op_kwargs={'threshold':0.12},
    dag=dag,
)

extract_dataset >> transform_dataset >> apply_model >> evaluate_model


    
    