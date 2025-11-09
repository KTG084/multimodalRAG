import requests
import json

BASE_URL = "http://localhost:8000"

def test_complete_flow():
    print("🧪 Testing complete RAG system...\n")
    
    # 1. Create session
    print("1️⃣  Creating session...")
    resp = requests.post(f"{BASE_URL}/session/create")
    session_id = resp.json()["session_id"]
    print(f"   ✅ Session: {session_id}\n")
    
    # 2. Upload PDF
    print("2️⃣  Uploading PDF...")
    with open("test.pdf", "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/upload", files=files)
        print(f"   ✅ {resp.json()}\n")
    
    # 3. Ask question
    print("3️⃣  Asking question...")
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": "What is this document about?",
            "session_id": session_id
        }
    )
    result = resp.json()
    print(f"   ✅ Answer: {result['answer']}\n")
    
    # 4. Follow-up question
    print("4️⃣  Follow-up question...")
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": "Can you elaborate?",
            "session_id": session_id
        }
    )
    result = resp.json()
    print(f"   ✅ Answer: {result['answer']}\n")
    
    # 5. Check history
    print("5️⃣  Checking history...")
    resp = requests.get(f"{BASE_URL}/session/{session_id}/history")
    history = resp.json()
    print(f"   ✅ Messages: {history['message_count']}\n")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_complete_flow()
