from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

from openai import OpenAI

from ivcap_service import getLogger, Service, JobContext
from ivcap_ai_tool import start_tool_server, ivcap_ai_tool, ToolOptions, logging_init

load_dotenv()

logging_init()
logger = getLogger("app")

service = Service(
    name="Simplistic AI Report Writer w/ Fact Checker",
    description= """
Attempts tow write a report on some topic and uses a separate fact checking agent to validate
the reference used.
""",
    contact={
        "name": "Max Ott",
        "email": "max.ott@data61.csiro.au",
    },
    license={
        "name": "MIT",
        "url": "https://opensource.org/license/MIT",
    },
)

class FactChecker(BaseModel):
    agent_id: str = Field(..., description="ID of the fact checker agent to use")
    model: Optional[str] = Field("gpt-4.1", description="Model to use for fact checking (optional)")
    temperature: Optional[float] = Field(0.3, description="Temperature parameter for fact checker model (optional)")

class ReportRequest(BaseModel):
    topic: str = Field(..., description="the topic of the report requested", example="The Solar System")
    fact_checker: Optional[FactChecker] = Field(None, description="the fact checker agent to use")
    model: Optional[str] = Field("gpt-4.1", description="Model to use for the report writer (optional)")
    temperature: Optional[float] = Field(0.7, description="Temperature parameter for model (optional)")

class ReferenceAssessment(BaseModel):
    reference: str = Field(..., description="The original reference text")
    assessment: Optional[str] = Field(None, description="LLM's assessment of credibility and relevance")

class ReportResponse(BaseModel):
    topic: str
    content: str
    references: List[ReferenceAssessment]

@ivcap_ai_tool("/", opts=ToolOptions(tags=["Report Writerr"]))
def generate_report(request: ReportRequest, ctxt: JobContext) -> ReportResponse:
    """Attempts tow write a report on some topic and uses a separate fact checking agent to validate the reference used."""

    logger.debug(f"Generating report for topic: {request.topic}")
    report_text = generate_initial_report(request)
    logger.debug(f"report: '{report_text[:40]}...'")

    references = check_references(report_text, request, ctxt)
    return ReportResponse(topic=request.topic, content=report_text, references=references)

def generate_initial_report(request: ReportRequest) -> str:
    """Generate an initial report on the given topic."""
    topic = request.topic
    client = get_client()

    response = client.chat.completions.create(
        model=request.model,
        messages=[
            {"role": "system", "content": "You are a science writer."},
            {"role": "user", "content": f"""
            Write a concise summary about "{topic}". Include at least 2 well-formatted references at the end, like:

            [1] Author/Source - URL
            [2] Author/Source - URL

            """},
        ],
        temperature=request.temperature,
    )
    report_text = response.choices[0].message.content
    return report_text

def check_references(report_text: str, request: ReportRequest, ctxt: JobContext):
    """Check the references using the fact checker service."""

    references = [line.strip() for line in report_text.splitlines() if line.strip().startswith("[")]

    fact_checker = request.fact_checker
    if not fact_checker:
        result = [{ "references": r } for r in references]
        return result

    agent_id = fact_checker.agent_id
    if not agent_id:
        raise ValueError("Fact checker agent ID is required")
    agent = ctxt.ivcap.get_agent(agent_id)
    req_model = agent.request_model
    req = req_model(
        references=references,
        model=fact_checker.model,
        temperature=fact_checker.temperature,
    )
    job = agent.exec_agent(req)
    if not job.succeeded:
        raise RuntimeError(f"Fact checking job failed: {job.error}")

    result = job.result["results"]
    return result

def get_client():
    litellm_proxy = os.environ.get("LITELLM_PROXY")
    if litellm_proxy:
        # Ensure the proxy URL ends with /v1 for OpenAI compatibility
        base_url = litellm_proxy.rstrip("/") + "/v1"
        return OpenAI(
            api_key="sk-xxx",  # dummy key, required by the client
            base_url=base_url
        )
    else:
        return OpenAI()

if __name__ == "__main__":
    start_tool_server(service)
