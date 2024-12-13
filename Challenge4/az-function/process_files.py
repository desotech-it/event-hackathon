import os
from dotenv import find_dotenv, load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient as OldDocumentIntelligenceClient, AnalyzeResult as OldAnalyzeResult
import httpx
import json

load_dotenv(find_dotenv())

def get_blob_service_client():
    connection_string = os.getenv('STORAGE_CONNECTION_STRING')
    if connection_string is None:
        raise ValueError("L'ambiente non contiene la variabile STORAGE_CONNECTION_STRING.")
    return BlobServiceClient.from_connection_string(connection_string)

def generate_sas_url(blob_service_client, container_name, blob_name, expiry_hours=1):
    start_time = datetime.utcnow()
    expiry_time = start_time + timedelta(hours=expiry_hours)

    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry_time,
        start=start_time,
    )

    sas_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    return sas_url

def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (word.span.offset + word.span.length) <= (span.offset + span.length):
            return True
    return False

def get_words(page, line):
    result = []
    if page.words and line.spans:
        for word in page.words:
            if _in_span(word, line.spans):
                result.append(word)
    return result

def analyze_layout(sas_url):
    endpoint = os.getenv("DOC_AI_ENDPOINT")
    api_key = os.getenv("DOC_AI_KEY")

    if not endpoint or not isinstance(endpoint, str):
        raise ValueError("La variabile DOC_AI_ENDPOINT non è impostata o non è una stringa.")
    if not api_key or not isinstance(api_key, str):
        raise ValueError("La variabile DOC_AI_KEY non è impostata o non è una stringa.")

    document_analysis_client = OldDocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    # Eseguiamo l'analisi usando i bytes del documento ottenuti tramite SAS URL
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-layout", httpx.Client().get(sas_url).read()
    )

    result: OldAnalyzeResult = poller.result()

    analysis_result = {
        "handwritten": any([style.is_handwritten for style in result.styles]) if result.styles else False,
        "pages": [],
        "tables": []
    }

    # Pagine
    for page in result.pages:
        page_info = {
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "unit": page.unit,
            "lines": [],
            "selection_marks": []
        }

        if page.lines:
            for line in page.lines:
                line_info = {
                    "text": line.content,
                    "polygon": line.polygon,
                    "words": [{"content": word.content, "confidence": word.confidence} for word in get_words(page, line)]
                }
                page_info["lines"].append(line_info)

        if page.selection_marks:
            for selection_mark in page.selection_marks:
                selection_mark_info = {
                    "state": selection_mark.state,
                    "polygon": selection_mark.polygon,
                    "confidence": selection_mark.confidence
                }
                page_info["selection_marks"].append(selection_mark_info)

        analysis_result["pages"].append(page_info)

    # Tabelle
    if result.tables:
        for table in result.tables:
            table_info = {
                "row_count": table.row_count,
                "column_count": table.column_count,
                "bounding_regions": [{"page_number": region.page_number, "polygon": region.polygon} for region in table.bounding_regions] if table.bounding_regions else [],
                "cells": [{"row_index": cell.row_index, "column_index": cell.column_index, "content": cell.content, "bounding_regions": [{"page_number": region.page_number, "polygon": region.polygon} for region in cell.bounding_regions] if cell.bounding_regions else []} for cell in table.cells]
            }
            analysis_result["tables"].append(table_info)

    return analysis_result

def save_analysis_results(blob_service_client, container_name, blob_name, analysis_results):
    if analysis_results is None:
        print(f"No analysis results for {blob_name}. Skipping save.")
        return

    results_blob_name = f"{blob_name}_results.json"
    results_json = json.dumps(analysis_results, indent=4)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=results_blob_name)
    blob_client.upload_blob(results_json, overwrite=True)

    print(f"Saved analysis results to {results_blob_name}")

if __name__ == "__main__":
    container_name = "data"
    blob_service_client = get_blob_service_client()

    blob_list = blob_service_client.get_container_client(container_name).list_blobs()

    for blob in blob_list:
        blob_name = blob.name
        print(f"Processing blob: {blob_name}")

        # Formati supportati
        supported_formats = ['.pdf', '.jpeg', '.jpg', '.png', '.tiff']
        if not any(blob_name.lower().endswith(ext) for ext in supported_formats):
            print(f"Skipping unsupported file format: {blob_name}")
            continue

        sas_url = generate_sas_url(blob_service_client, container_name, blob_name)
        print(f"Generated SAS URL: {sas_url}")

        analysis_results = analyze_layout(sas_url)
        save_analysis_results(blob_service_client, container_name, blob_name, analysis_results)
