import os
from ragaai_catalyst import RagaAICatalyst, Experiment, Tracer, Dataset
from dotenv import load_dotenv
load_dotenv(override=True)

catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_ACCESS_KEY'),
    secret_key=os.getenv('RAGAAI_SECRET_KEY'),
    base_url=os.getenv('RAGAAI_BASE_URL')
)

print('catalyst', catalyst)

from ragaai_catalyst import Dataset
import pandas as pd

# df = pd.read_csv('chat-dataset.csv')
df = pd.read_csv('MyDataset.csv')
# print(df)
temp_dict = {}  
for col in df.columns:
    temp_dict[col] = col

schema_mapping2={
    'Query': 'prompt',
    'response': 'response',
    'Context': 'context',
    'expectedResponse': 'expected_response',
}

# Initialize Dataset management for a specific project
dataset_manager = Dataset(project_name="TestProject003")
print('dataset_manager', dataset_manager)

# Create a dataset from CSV
dataset_manager.create_from_csv(
    csv_path='MyDataset.csv',
    dataset_name='mydataset002',
    schema_mapping=schema_mapping2
)

# # List existing datasets
# datasets = dataset_manager.list_datasets()
# print("Existing Datasets:", datasets)

# # Create a dataset from JSONl
# dataset_manager.create_from_jsonl(
#     jsonl_path='jsonl_path',
#     dataset_name='MyDataset',
#     schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
# )

# # Create a dataset from dataframe
# dataset_manager.create_from_df(
#     df=df,
#     dataset_name='MyDataset',
#     schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
# )

# Get project schema mapping
dataset_manager.get_schema_mapping()
