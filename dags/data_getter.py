import sys
import time

import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

try:
    from DataGetters.reddit_task import main
except ImportError:
    import os
    sys.path.append(os.getcwd())
    from DataGetters.reddit_task import main

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(1)
}

dag = DAG(
    dag_id='data_getter', default_args=args,
    schedule_interval="@once")

run_this = PythonOperator(task_id='reddit_comments',
                          provide_context=True,
                          python_callable=main,
                          dag=dag)
