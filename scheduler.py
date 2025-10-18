# scheduler.py

import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Import all the necessary components
from documents_scraper import scrape_by_subject, URLS_BY_SUBJECT
from versioning import manage_version_rotation, process_regulatory_changes
from ingestion import main as ingest_main

# Create the scheduler instance
scheduler = AsyncIOScheduler(timezone="Asia/Karachi")

async def run_weekly_scrape_cycle():
    """
    Complete automated weekly pipeline.
    Correct operational order:
    1. Scrape new documents → versions/new/
    2. Rotate versions (new→latest, latest→previous)
    3. Detect changes (compare new latest vs previous)
    4. Ingest updated documents to Pinecone
    """
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'='*80}")
    print(f"SCHEDULER: Starting weekly cycle at {timestamp}")
    print(f"{'='*80}\n")
    
    loop = asyncio.get_running_loop()
    
    try:
        # --- STEP 1: SCRAPE ALL SUBJECTS ---
        print("\n--- STEP 1: Scraping new documents ---")
        all_subjects = list(URLS_BY_SUBJECT.keys())
        
        for i, subject in enumerate(all_subjects, 1):
            print(f"\nScraping subject {i}/{len(all_subjects)}: {subject}")
            try:
                # Correct function signature: (subject_query, max_depth, headless)
                result = await loop.run_in_executor(
                    None, 
                    scrape_by_subject,
                    subject,  # subject_query
                    2,        # max_depth (increase for deeper crawl)
                    True      # headless
                )
                if result:
                    print(f"✓ Successfully scraped: {result}")
                else:
                    print(f"⚠ No match found for: {subject}")
            except Exception as e:
                print(f"✗ Error scraping {subject}: {e}")
                # Continue with other subjects even if one fails
                continue
        
        print("\n✓ SCRAPING COMPLETE")
       
        # --- STEP 2: ROTATE VERSIONS ---
        print("\n--- STEP 2: Rotating versions ---")
        print("  - Clearing old 'previous'")
        print("  - Moving 'latest' → 'previous'")
        print("  - Promoting 'new' → 'latest'")
        await loop.run_in_executor(None, manage_version_rotation, "versions/new")
        print("✓ VERSION ROTATION COMPLETE")

        # --- STEP 3: DETECT CHANGES ---
        print("\n--- STEP 3: Detecting regulatory changes ---")
        print("  - Comparing new 'latest' vs 'previous'")
        await loop.run_in_executor(None, process_regulatory_changes)
        print("✓ CHANGE DETECTION COMPLETE")

        # --- STEP 4: INGEST NEW DATA ---
        print("\n--- STEP 4: Ingesting to Pinecone ---")
        print("  - Processing new/modified PDFs")
        print("  - Updating vector embeddings")
        await loop.run_in_executor(None, ingest_main)
        print("✓ INGESTION COMPLETE")
        
        print(f"\n{'='*80}")
        print(f"WEEKLY CYCLE COMPLETED SUCCESSFULLY at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"{'='*80}\n")
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"!!! CRITICAL ERROR: Weekly cycle failed !!!")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()


def schedule_job():
    """Adds the single weekly job to the scheduler."""
    print("\n--- Configuring the weekly scrape cycle job ---")
    scheduler.add_job(
        run_weekly_scrape_cycle,
        trigger='cron',
        day_of_week='sat',  
        hour=2,             
        minute=0,
        id="weekly_scrape_cycle",
        replace_existing=True
    )
    print("✓ Weekly job scheduled for every Saturday at 02:00 UTC")
    job = scheduler.get_job("weekly_scrape_cycle")



async def start_scheduler():
    """Start the scheduler and keep it running."""
    schedule_job()
    scheduler.start()
    print("\n✓ Scheduler started successfully")
    print("  Press Ctrl+C to stop\n")
    
    try:
        # Keep the scheduler running
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n--- Shutting down scheduler ---")
        scheduler.shutdown()


if __name__ == "__main__":
    print("="*80)
    print("LUKE REGULATORY MONITORING SCHEDULER")
    print("="*80)
    asyncio.run(start_scheduler())


def run_manual_scrape():
    """Simple synchronous entry for manual or cron-based trigger."""
    import asyncio
    asyncio.run(run_weekly_scrape_cycle())
