"""FastAPI entrypoint.

Phase 1 skeleton only — no document/graph/rag routers yet (those land in
PLAN.md Phase 17, once the graph + vector engines exist).
"""

from fastapi import FastAPI

app = FastAPI(title="WeaveX")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)