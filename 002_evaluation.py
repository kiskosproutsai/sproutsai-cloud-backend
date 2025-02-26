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

from ragaai_catalyst import Evaluation

# Create an experiment
evaluation = Evaluation(
    project_name="TestProject003",
    dataset_name="mydataset",
)

# Get list of available metrics
metrics_list = evaluation.list_metrics()
print('metrics_list', metrics_list)

# Add metrics to the experiment

schema_mapping1={
    'Query': 'prompt',
    'Context': 'context',
    'expectedResponse': 'expected_response',
    'response':'text'
}

schema_mapping2={
    'Query': 'prompt',
    'response': 'response',
    'Context': 'context',
    'expectedResponse': 'expected_response',
}


# THIS ADDS A JOB
# Add single metric
evaluation.add_metrics(
    metrics=[
      {"name": "PII", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.232323}}, "column_name": "PII001", "schema_mapping": schema_mapping1},
      {"name": "Ragas/Factual Correctness", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.232323}}, "column_name": "FactCheck001", "schema_mapping": schema_mapping2},
    
    ]
)
# RESULT: Metric Evaluation Job scheduled successfully?

# ## THIS ADDS A JOB
# # Add multiple metrics
# evaluation.add_metrics(
#     metrics=[
#         {"name": "Faithfulness", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"gte": 0.323}}, "column_name": "Faithfulness_gte1", "schema_mapping": schema_mapping},
#         {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"lte": 0.323}}, "column_name": "Hallucination_lte1", "schema_mapping": schema_mapping},
#         {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "provider": "openai", "threshold": {"eq": 0.323}}, "column_name": "Hallucination_eq1", "schema_mapping": schema_mapping},
#     ]
# )
# ##Metric Evaluation Job scheduled successfully
# ##Job in progress. Please wait while the job completes.

# Get the status of the experiment
status = evaluation.get_status()
print("Experiment Status:", status)

# Get the results of the experiment
results = evaluation.get_results()
print("Experiment Results:", results)
print('results.columns', results.columns)

# # Appending Metrics for New Data
# # If you've added new rows to your dataset, you can calculate metrics just for the new data:
# evaluation.append_metrics(display_name="Faithfulness_v1")