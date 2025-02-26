
import os
from ragaai_catalyst import RagaAICatalyst, Experiment, Tracer, Dataset
from dotenv import load_dotenv
load_dotenv(override=True)

catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_ACCESS_KEY'),
    secret_key=os.getenv('RAGAAI_SECRET_KEY'),
    base_url=os.getenv('RAGAAI_BASE_URL')
)

from ragaai_catalyst import RagaAICatalyst, Tracer, trace_llm, trace_tool, trace_agent, current_span

agentic_tracing_dataset_name = "agentic_tracing_dataset_name"

tracer = Tracer(
    project_name=agentic_tracing_project_name,
    dataset_name=agentic_tracing_dataset_name,
    tracer_type="Agentic",
)

from ragaai_catalyst import trace_llm, trace_tool, trace_agent, current_span

from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv(override=True)

@trace_llm(name="llm_call", tags=["default_llm_call"])
def llm_call(prompt, max_tokens=512, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.85,
    )
    # Span level context
    current_span().add_context("name = span level in summary_agent, context = some span level context")

    # Span level execute metrics
    current_span().execute_metrics(
        name="Hallucination",
        model="gpt-4o",
        provider="openai"
    )
    response_data = response.choices[0].message.content.strip()
    print('response_data: ', response_data)
    return response_data

class SummaryAgent:
    def __init__(self, persona="Summary Agent"):
        self.persona = persona

    @trace_agent(name="summary_agent")
    def summarize(self, text):
        prompt = f"Please summarize this text concisely: {text}"

        # Span level metric
        current_span().add_metrics(name='Accuracy', score=0.5, reasoning='some reasoning')

        # Span level context
        current_span().add_context("name = span level in summary_agent, context = some span level context")

        summary = llm_call(prompt)
        return summary


class AnalysisAgent:
    def __init__(self, persona="Analysis Agent"):
        self.persona = persona
        self.summary_agent = SummaryAgent()

    @trace_agent(name="analysis_agent")
    def analyze(self, text):
        summary = self.summary_agent.summarize(text)

        prompt = f"Given this summary: {summary}\nProvide a brief analysis of the main points."

        # Span level metric
        current_span().add_metrics(name='correctness', score=0.5, reasoning='some reasoning')
        analysis = llm_call(prompt)

        return {
            "summary": summary,
            "analysis": analysis
        }

class RecommendationAgent:
    def __init__(self, persona="Recommendation Agent"):
        self.persona = persona
        self.analysis_agent = AnalysisAgent()

    @trace_agent(name="recommendation_agent", tags=['coordinator_agent'])
    def recommend(self, text):
        analysis_result = self.analysis_agent.analyze(text)

        prompt = f"""Given this summary: {analysis_result['summary']}
        And this analysis: {analysis_result['analysis']}
        Provide 2-3 actionable recommendations."""

        recommendations = llm_call(prompt)

        return {
            "summary": analysis_result["summary"],
            "analysis": analysis_result["analysis"],
            "recommendations": recommendations
        }
#Defining agent tracer
@trace_agent(name="get_recommendation", tags=['coordinator_agent'])
def get_recommendation(agent, text):
    recommendation = agent.recommend(text)
    return recommendation

def main():
    text = """
    Artificial Intelligence has transformed various industries in recent years.
    From healthcare to finance, AI applications are becoming increasingly prevalent.
    Machine learning models are being used to predict market trends, diagnose diseases,
    and automate routine tasks. The impact of AI on society continues to grow,
    raising both opportunities and challenges for the future.
    """

    recommendation_agent = RecommendationAgent()
    result = get_recommendation(recommendation_agent, text)


    # Trace level metric
    tracer.add_metrics(name='hallucination_1', score=0.5, reasoning='some reasoning')

# Run tracer
with tracer:
    main()