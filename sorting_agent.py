from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SortRequest(BaseModel):
    files: list[dict]

@app.post("/sort")
async def sort_files(request: SortRequest):
    buckets = {
        "short":  [],   # 1–5 chars
        "medium": [],   # 6–15 chars
        "long":   []    # 16+ chars
    }

    for f in request.files:
        length = f["name_length"]
        if length <= 5:
            buckets["short"].append(f)
        elif length <= 15:
            buckets["medium"].append(f)
        else:
            buckets["long"].append(f)

    return buckets

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)