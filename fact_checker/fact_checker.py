from pydantic import BaseModel, Field
from typing import List
import openai
import os
from dotenv import load_dotenv

from openai import OpenAI

from ivcap_service import getLogger, Service
from ivcap_ai_tool import start_tool_server, ivcap_ai_tool, ToolOptions, logging_init

#openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

logging_init()
logger = getLogger("app")


service = Service(
    name="Simplistic AI Fact Checker Agent",
    description= """
Provides a simple agent which assess the credibility and relevance of a specific reference.
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

class FactCheckInput(BaseModel):
    references: List[str] = Field(..., description="List of references to be checked")

class ReferenceAssessment(BaseModel):
    reference: str = Field(..., description="The original reference text")
    assessment: str = Field(..., description="LLM's assessment of credibility and relevance")

class FactCheckOutput(BaseModel):
    results: List[ReferenceAssessment]

@ivcap_ai_tool("/", opts=ToolOptions(tags=["Fact Checker"], service_id="/"))
async def verify_references(input: FactCheckInput) -> FactCheckOutput:
    """Verify and assess the quality of a list of references and return
    an assement returned by an LLM."""
    verified_refs = []
    client = OpenAI()
    for ref in input.references:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a critical academic reviewer."},
                {"role": "user", "content": f"Assess the credibility and relevance of this reference: {ref}"}
            ],
            temperature=0.3
        )
        assessment = response.choices[0].message.content
        verified_refs.append(ReferenceAssessment(reference=ref, assessment=assessment))
    return FactCheckOutput(results=verified_refs)

if __name__ == "__main__":
    import argparse
    def custom_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
        parser.add_argument('--litellm-proxy', type=str, help='Address of the the LiteLlmProxy')
        args = parser.parse_args()
        if args.litellm_proxy != None:
            os.setenv("LITELLM_PROXY", args.litellm_proxy)
        return args

    start_tool_server(service, custom_args=custom_args)
