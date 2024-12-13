import re
import json
from pydantic import BaseModel
from openai import AzureOpenAI
import os

def model_houseloanterms(house_terms):
    """
    Processes and structures house loan terms data.
    """
    structured_text = clean_text_data(house_terms)

    # Pass structured text to OpenAI for further processing
    final_json = create_structured_json(structured_text)

    # Structure the final document
    document = {
        'id': "house_loan_terms",
        'content': final_json
    }
    return document

def clean_text_data(json_data):
    """
    Extract and clean text from the provided JSON data.
    """
    content = []
    pages = json_data.get("pages", [])
    for page in pages:
        for line in page.get("lines", []):
            content.append(line.get("text", "").strip())
    return " ".join(content)

def create_structured_json(house_loan_terms_text):
    """
    Uses Azure OpenAI to process and structure house loan terms text.
    """
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-08-01-preview"
    )

    model = os.getenv("AZURE_OPENAI_MODEL")

    class HouseLoanTerms(BaseModel):
        introduction: str
        loan_amount_and_purpose: str
        interest_rates: str
        loan_tenure: str
        monthly_repayments: str
        late_payments: str
        loan_security: str
        loan_processing_fees: str
        default_and_foreclosure: str
        early_repayment: str
        changes_to_terms: str
        insurance_requirements: str
        loan_cancellation: str
        dispute_resolution: str
        governing_law: str
        contact_information: str

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "Extract the information about house loan terms in a structured format."},
            {"role": "user", "content": house_loan_terms_text},
        ],
        response_format=HouseLoanTerms,
    )

    return completion.model_dump_json(indent=2)
