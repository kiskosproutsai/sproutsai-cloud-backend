import os
from ragaai_catalyst import RagaAICatalyst, Experiment, Tracer, Dataset
from dotenv import load_dotenv
load_dotenv(override=True)

catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_ACCESS_KEY'),
    secret_key=os.getenv('RAGAAI_SECRET_KEY'),
    base_url=os.getenv('RAGAAI_BASE_URL')
)

# #Creating of project
# new_project = catalyst.create_project(
#     project_name="project001",
#     # description="Your project description"
# )

projects = catalyst.list_projects(num_projects=5)
print(f"Projects: {projects}")


#METRICS
from ragaai_catalyst import Experiment

experiment_manager = Experiment(
    project_name="project001",
    experiment_name="Your Experiment Name",
    experiment_description="Experiment description",
    dataset_name="your_dataset_name"
)

response = experiment_manager.add_metrics(
    metrics=[
        {"name": "Hallucination", "config": {"model": "gpt-4o-mini", "reason": True, "provider": "OpenAI"}},
        {"name": "Faithfulness", "config": {"model": "azure/azure-gpt-4o-mini", "reason": True, "provider": "Azure"}}
    ]
)

print("Metric Response:", response)