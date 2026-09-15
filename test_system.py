import json
import os

import requests

BASE_URL = "http://localhost:8000"
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "change_me_to_a_random_secret")
HEADERS = {"X-Internal-Key": INTERNAL_KEY}

USER_ID = "test-user"
DOCUMENT_ID = "test-doc-1"


def test_complete_flow():
    print("Testing RAG service (called the way the Next.js backend would)...\n")

    # This service ingests by URL, not multipart upload — point it at any
    # publicly reachable PDF for a smoke test.
    print("1. Ingesting a document by URL...")
    resp = requests.post(
        f"{BASE_URL}/ingest",
        headers=HEADERS,
        json={
            "document_id": DOCUMENT_ID,
            "user_id": USER_ID,
            "file_url": "https://arxiv.org/pdf/1706.03762",
            "filename": "attention-is-all-you-need.pdf",
            "content_type": "application/pdf",
        },
    )
    resp.raise_for_status()
    print(f"   {resp.json()}\n")

    print("2. Querying with streaming response...")
    with requests.post(
        f"{BASE_URL}/query",
        headers=HEADERS,
        json={
            "query": "What is this document about?",
            "user_id": USER_ID,
            "history": [],
            "document_ids": [DOCUMENT_ID],
        },
        stream=True,
    ) as resp:
        resp.raise_for_status()
        answer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            if event["type"] == "sources":
                print(f"   sources: {event['sources']}\n")
            elif event["type"] == "token":
                answer += event["content"]
            elif event["type"] == "done":
                break
        print(f"   answer: {answer}\n")

    print("3. Deleting the document's vectors...")
    resp = requests.delete(
        f"{BASE_URL}/documents/{DOCUMENT_ID}", headers=HEADERS, params={"user_id": USER_ID}
    )
    resp.raise_for_status()
    print(f"   {resp.json()}\n")

    print("All checks passed.")


if __name__ == "__main__":
    test_complete_flow()
