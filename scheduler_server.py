import os
import threading
import asyncio
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from scheduler import run_manual_scrape, schedule_job, scheduler

SCHEDULER_SECRET = os.getenv("SCHEDULER_SECRET", "changeme")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the scheduler."""
    print("🚀 Starting LUKE Scheduler...")
    schedule_job()
    scheduler.start()
    print("✓ APScheduler is running (weekly cron active).")
    yield  # <-- App runs while this yields
    print("🛑 Shutting down LUKE Scheduler...")
    scheduler.shutdown()

app = FastAPI(
    title="LUKE Scheduler Service",
    lifespan=lifespan
)

@app.post("/run")
async def run_now(request: Request):
    """Trigger the scraper manually from main API."""
    data = await request.json()
    if data.get("secret") != SCHEDULER_SECRET:
        return {"error": "Unauthorized"}
    threading.Thread(target=run_manual_scrape).start()
    return {"status": "Manual scrape triggered"}

@app.get("/")
def root():
    return {"message": "Scheduler service running", "weekly_job": "Saturday 02:00 PST"}
