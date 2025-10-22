# run_scheduler.py
from scheduler import run_manual_scrape

if __name__ == "__main__":
    print("🚀 Starting LUKE Scheduler (Cron mode)...")
    run_manual_scrape()
    print("✅ Scrape completed, shutting down.")
