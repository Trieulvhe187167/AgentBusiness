import httpx

url = "http://127.0.0.1:8081/api/admin/upload"
filepath = "kb_sample.csv"

with open(filepath, "rb") as f:
    files = {"file": (filepath, f, "text/csv")}
    response = httpx.post(url, files=files, timeout=60.0)
    print("Status:", response.status_code)
    print("Response:", response.text)
