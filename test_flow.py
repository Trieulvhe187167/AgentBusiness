import httpx
import time

def run():
    with open("chat_output.txt", "w", encoding="utf-8") as out:
        # 1. Trigger ingestion
        out.write("Triggering ingestion...\n")
        r = httpx.post("http://127.0.0.1:8081/api/ingest/all", timeout=60.0)
        out.write(f"Ingest Response: {r.status_code} {r.text}\n")
        
        # wait a bit for background job
        time.sleep(3)

        # 2. Check stats
        r = httpx.get("http://127.0.0.1:8081/api/kb/stats", timeout=60.0)
        out.write(f"KB Stats: {r.status_code} {r.text}\n")
        
        # 3. Test chat
        out.write("\nTesting chat...\n")
        payload = {
            "session_id": "test_session_1",
            "message": "phí giao hàng là bao nhiêu?",
            "lang": "vi"
        }
        with httpx.stream("POST", "http://127.0.0.1:8081/api/chat", json=payload, timeout=60.0) as r:
            for line in r.iter_lines():
                if line:
                    out.write(line + "\n")

if __name__ == "__main__":
    run()
