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

# # Create a project
# project = catalyst.create_project(
#     project_name="Test-RAG-App-1",
#     usecase="Chatbot"
# )

# Get project usecases
catalyst.project_use_cases()

# List projects
projects = catalyst.list_projects()
print(projects)