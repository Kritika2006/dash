# cfo_dashboard/backend/main.py
from fastapi import FastAPI
from .api import router

app = FastAPI(
    title="AI CFO Dashboard API",
    description="Backend services for financial analysis and insights."
)

# Include the main router
app.include_router(router.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI CFO API"}