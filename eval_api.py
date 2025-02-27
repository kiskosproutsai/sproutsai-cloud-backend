import asyncio
from fastapi import HTTPException
import os
from ragaai_catalyst import RagaAICatalyst, Experiment, Tracer, Dataset
from ragaai_catalyst import Evaluation
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import pandas as pd
import time
load_dotenv(override=True)

project_name = "TestProject003"

catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_ACCESS_KEY'),
    secret_key=os.getenv('RAGAAI_SECRET_KEY'),
    base_url=os.getenv('RAGAAI_BASE_URL')
)

# # Create a dataset from CSV
# dataset_manager.create_from_csv(
#     csv_path='MyDataset.csv',
#     dataset_name='mydataset002',
#     schema_mapping=schema_mapping2
# )

# # List existing datasets
# datasets = dataset_manager.list_datasets()
# print("Existing Datasets:", datasets)

# # Create a dataset from JSONl
# dataset_manager.create_from_jsonl(
#     jsonl_path='jsonl_path',
#     dataset_name='MyDataset',
#     schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
# )





# # Get project schema mapping
# dataset_manager.get_schema_mapping()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/evaluate")
async def evaluate(request: Request):

    try:

        # Initialize Dataset management for a specific project
        dataset_manager = Dataset(project_name=project_name)
        print('dataset_manager', dataset_manager)

        # schema_mapping = {
        #     'Prompt': 'prompt',
        #     'Response': 'response',
        #     'Context': 'context'
        # }

        schema_mapping = {
            'prompt': 'prompt',
            'response': 'response',
            'context': 'context'
        }

        data = await request.json()
        print(data)

        df = pd.DataFrame(data)

        temp_dataset_name = 'data_for_eval_' + time.strftime("%Y%m%d_%H%M%S") #+ time.ctime()
        temp_dataset_name = temp_dataset_name.replace(' ', '_').replace(':', '_').lower()

        # # Create a dataset from dataframe
        # dataset_manager.create_from_df(
        #     df=df,
        #     dataset_name=temp_dataset_name,
        #     schema_mapping=schema_mapping
        # )

        df.to_csv(temp_dataset_name+'.csv', index=False)

        print('df\n\n', df)

        print('START Creating dataset from csv')
        temp_response = dataset_manager.create_from_csv(
            csv_path=temp_dataset_name+'.csv',
            dataset_name=temp_dataset_name,
            schema_mapping=schema_mapping
        )
        print('temp_response', temp_response)
        print('END Creating dataset from csv')

        # for i in range(10):
        #     print('i', i)
        #     time.sleep(1)

        #delete temp_dataset_name+'.csv'
        os.remove(temp_dataset_name+'.csv')

        print('START Creating evaluation')
        # Create an experiment
        evaluation = Evaluation(
            project_name=project_name,
            dataset_name=temp_dataset_name,
        )
        print('END Creating evaluation')

        print('START Adding metrics')
        # THIS ADDS A JOB
        # Add single metric
        evaluation.add_metrics(
            metrics=[
            #   {"name": "PII", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.232323}}, "column_name": "PII001", "schema_mapping": schema_mapping1},
            #   {"name": "Ragas/Factual Correctness", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.232323}}, "column_name": "FactCheck001", "schema_mapping": schema_mapping2},
            {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.232323}}, "column_name": "hallucination", "schema_mapping": schema_mapping},
            # {"name": "Context Relevancy", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.232323}}, "column_name": "context_relevancy", "schema_mapping": schema_mapping},
            
            ]
        )
        print('END Adding metrics')
        # Get the status of the experiment
        status = evaluation.get_status()
        print("Experiment Status:", status)

        # Get the results of the experiment
        results = evaluation.get_results()
        print("Experiment Results:", results)
        print('results.columns', results.columns)


        return {"message": "evaluation done"}
    except Exception as e:
        return {"message": str(e)}
    