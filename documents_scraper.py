# documents_scraper.py

"""
SPIJ PDF Scraper - Deep crawls Peru's Legal Information System
Systematically explores all pages and uploads PDFs directly to Supabase storage,
organizing uploads into subject-specific folders using an LLM for matching.

Requires: selenium, webdriver-manager, openai, python-dotenv, supabase
Install: pip install selenium webdriver-manager openai python-dotenv supabase
"""

import os
import json
import time
import hashlib
from datetime import datetime
from collections import deque
from urllib.parse import urlparse, urlunparse
import tempfile

# --- OpenAI and Environment Key Setup ---
import openai
from dotenv import load_dotenv

# --- Supabase Setup ---
from supabase import create_client, Client

# Load environment variables from a .env file
load_dotenv()

# --- Selenium and WebDriver Imports ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains


class SPIJScraper:
    def __init__(self, headless=False, subject_folder=None):
        """Initialize the scraper with Chrome WebDriver and Supabase client"""
        
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_bucket = os.getenv("SUPABASE_BUCKET")
        
        if not all([supabase_url, supabase_key, self.supabase_bucket]):
            raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL, SUPABASE_KEY, and SUPABASE_BUCKET in .env file")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.subject_folder = subject_folder or "general"
        
        print(f"✓ Supabase client initialized")
        print(f"✓ Bucket: {self.supabase_bucket}")
        print(f"✓ Subject folder: {self.subject_folder}")
        
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless=new')
        
        # Essential Chrome arguments
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--window-size=1920,1080')

        # Disable automation flags
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Use temporary directory for downloads
        self.temp_download_dir = tempfile.mkdtemp(prefix='spij_temp_')
        print(f"✓ Temporary download directory: {self.temp_download_dir}")
        
        # Download preferences
        prefs = {
            "download.default_directory": self.temp_download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_setting_values.automatic_downloads": 1,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # Detect Render or local
        if os.environ.get("RENDER"):
            # Running on Render – use system binaries
            chrome_options.binary_location = shutil.which("chromium") or "/usr/bin/chromium"
            service = Service(shutil.which("chromedriver") or "/usr/bin/chromedriver")
        else:
            # Local dev – use WebDriverManager
            service = Service(ChromeDriverManager().install())

        # Initialize driver
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

        # Allow downloads in headless mode
        self.driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {"behavior": "allow", "downloadPath": self.temp_download_dir}    
        )

        print("✓ Chrome WebDriver initialized successfully.")
        
        self.base_url = "https://spij.minjus.gob.pe"
        
        # Tracking sets and dicts
        self.visited_urls = set()
        self.pdf_urls = set()
        self.external_pdf_urls = set()
        self.uploaded_count = 0
        self.uploaded_pdfs = {}  # filename -> sha256 hash
        self._processed_identifiers = set()  # Track processed URLs to avoid duplicates
        
        # Queue for BFS traversal
        self.url_queue = deque()

    def compute_file_hash(self, file_bytes):
        """Compute SHA256 hash of file bytes"""
        sha = hashlib.sha256()
        sha.update(file_bytes)
        return sha.hexdigest()

    def upload_to_supabase(self, local_file_path, pdf_identifier):
        """Upload a PDF file to Supabase storage"""
        try:
            # Keep original filename without modification
            filename = os.path.basename(local_file_path)
            
            # Create storage path: subject_folder/filename (no timestamp)
            storage_path = f"versions/new/{self.subject_folder}/{filename}"
            
            # Read the file
            with open(local_file_path, 'rb') as f:
                file_data = f.read()
            
            # Compute hash
            file_hash = self.compute_file_hash(file_data)
            
            # Upload to Supabase with upsert to overwrite if exists
            response = self.supabase.storage.from_(self.supabase_bucket).upload(
                path=storage_path,
                file=file_data,
                file_options={"content-type": "application/pdf", "upsert": "true"}
            )
            
            print(f"    ✓ Uploaded to Supabase: {storage_path}")
            
            # Track hash and identifier
            self.uploaded_pdfs[filename] = file_hash
            self._processed_identifiers.add(pdf_identifier)
            
            # Delete local file after successful upload
            os.remove(local_file_path)
            
            return storage_path
            
        except Exception as e:
            print(f"    ✗ Error uploading to Supabase: {e}")
            # Try to delete local file even if upload failed
            try:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
            except:
                pass
            return None

    def upload_hashes_json(self):
        """Upload a per-subject hashes.json with all file hashes."""
        hashes_json = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_files": len(self.uploaded_pdfs),
                "subject": self.subject_folder
            },
            "files": {name: {"hash": h} for name, h in self.uploaded_pdfs.items()}
        }
        
        json_bytes = json.dumps(hashes_json, indent=2).encode("utf-8")
        path = f"versions/new/{self.subject_folder}/hashes.json"
        
        try:
            self.supabase.storage.from_(self.supabase_bucket).upload(
                path=path,
                file=json_bytes,
                file_options={"content-type": "application/json", "upsert": "true"}
            )
            print(f"✓ Uploaded hashes.json to Supabase: {path}")
        except Exception as e:
            print(f"✗ Error uploading hashes.json: {e}")

    def normalize_url(self, url):
        """Normalize URL — ensure '/#/' exists for SPA navigation."""
        url = url.strip()
        parsed = urlparse(url)

        # Only fix URLs from the SPIJ site
        if "spij.minjus.gob.pe" in parsed.netloc:
            # If missing '/#/' but contains '/detallenorma/', insert it
            if '/detallenorma/' in parsed.path and '/#/' not in url:
                fixed_path = parsed.path.replace('/detallenorma/', '/#/detallenorma/')
                url = urlunparse((parsed.scheme, parsed.netloc, fixed_path, '', '', ''))

        return url

    def is_same_page_section(self, current_url, new_url):
        """Check if new_url is just a minor variation of current page"""
        current_parsed = urlparse(current_url)
        new_parsed = urlparse(new_url)
        if current_parsed.path != new_parsed.path: return False
        
        current_frag = current_parsed.fragment
        new_frag = new_parsed.fragment
        if current_frag == new_frag: return True
        
        if '/detallenorma/' in current_frag and '/detallenorma/' in new_frag:
            current_id = current_frag.split('/detallenorma/')[-1].split('/')[0].split('#')[0]
            new_id = new_frag.split('/detallenorma/')[-1].split('/')[0].split('#')[0]
            if current_id == new_id: return True
        return False
    
    def wait_for_page_load(self, timeout=30):
        """Wait for page to load"""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            time.sleep(4)
        except Exception as e:
            print(f"  Page load timeout: {e}")
    
    def click_ingresar_button(self):
        """Click INGRESAR button once at start"""
        try:
            print("Looking for INGRESAR button...")
            wait = WebDriverWait(self.driver, 15)
            button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'INGRESAR')]")))
            print("Found INGRESAR button, clicking...")
            self.driver.execute_script("arguments[0].click();", button)
            time.sleep(4)
            print("Clicked INGRESAR successfully")
            return True
        except Exception as e:
            print(f"Error clicking INGRESAR: {e}")
            return False
    
    def click_legislation_by_subject(self):
        """Click 'LEGISLATION BY SUBJECT' button"""
        try:
            print("Looking for LEGISLATION BY SUBJECT button...")
            wait = WebDriverWait(self.driver, 10)
            button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(.//label, 'LEGISLATION BY SUBJECT')]")))
            print("Found LEGISLATION BY SUBJECT button, clicking...")
            self.driver.execute_script("arguments[0].click();", button)
            time.sleep(3)
            print("Clicked LEGISLATION BY SUBJECT successfully")
            return True
        except Exception as e:
            print(f"Error clicking LEGISLATION BY SUBJECT: {e}")
            return False
    
    def wait_for_new_file(self, before_files, timeout=60):
        """Wait for a new or updated PDF to appear in the download directory."""
        start = time.time()
        before_snapshot = {f: os.path.getmtime(os.path.join(self.temp_download_dir, f))
                           for f in before_files if os.path.exists(os.path.join(self.temp_download_dir, f))}

        while time.time() - start < timeout:
            current_files = [f for f in os.listdir(self.temp_download_dir) if f.lower().endswith('.pdf')]
            for f in current_files:
                file_path = os.path.join(self.temp_download_dir, f)

                # New file
                if f not in before_snapshot:
                    return file_path

                # Updated file (timestamp changed — likely re-downloaded)
                if os.path.getmtime(file_path) > before_snapshot[f]:
                    return file_path

            # Still downloading? wait a bit
            temp_files = [f for f in os.listdir(self.temp_download_dir) if f.endswith('.crdownload')]
            if temp_files:
                time.sleep(2)
                continue

            time.sleep(1)

        return None

    def handle_download_dialog(self, pdf_identifier):
        """Handle download dialog: select PDF, download, and upload to Supabase"""
        try:
            print("    Waiting for download dialog...")
            time.sleep(3)
            
            if pdf_identifier in self._processed_identifiers:
                print(f"    ⚠ PDF already uploaded, skipping: {pdf_identifier}")
                try:
                    close_btns = self.driver.find_elements(By.XPATH, "//button[contains(@class, 'close') or contains(., 'Cerrar') or contains(., 'Cancelar')]")
                    if close_btns:
                        close_btns[0].click()
                        time.sleep(1)
                        print("    ✓ Dialog closed")
                except: pass
                return False
            
            try:
                WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.XPATH, "//mat-radio-button[@value='2']")))
            except:
                print("    Dialog not found")
                return False
            
            pdf_radio_selectors = ["//mat-radio-button[@value='2']//label", "//mat-radio-button[@value='2']//input", "//mat-radio-button[@value='2']", "//input[@value='2' and @type='radio']", "//mat-radio-button[@value='2']//span[@class='mat-radio-container']"]
            pdf_selected = False
            for selector in pdf_radio_selectors:
                try:
                    radio = self.driver.find_element(By.XPATH, selector)
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", radio)
                    time.sleep(0.5)
                    self.driver.execute_script("arguments[0].click();", radio)
                    time.sleep(1)
                    print("    ✓ PDF option selected")
                    pdf_selected = True
                    break
                except Exception: continue
            
            if not pdf_selected:
                print("    ⚠ Could not select PDF option")
                return False
            
            download_selectors = ["//button[contains(.//font, 'Discharge')]", "//button[contains(., 'Discharge')]", "//mat-card-actions//button[contains(@class, 'mat-warn')]", "//mat-card-actions//button", "//button[contains(.//span, 'Descargar')]"]
            download_clicked = False
            for selector in download_selectors:
                try:
                    button = self.driver.find_element(By.XPATH, selector)
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                    time.sleep(0.5)

                    ActionChains(self.driver).move_to_element(button).click().perform()
                    print("    ✓ Download button clicked")

                    # Track existing files before click
                    before_files = set(os.listdir(self.temp_download_dir))

                    # Wait for an actual new .pdf file to appear
                    downloaded_file = self.wait_for_new_file(before_files, timeout=70)
                    
                    if downloaded_file:
                        print(f"    ✓ PDF downloaded locally: {os.path.basename(downloaded_file)}")
                        
                        # Upload to Supabase
                        storage_path = self.upload_to_supabase(downloaded_file, pdf_identifier)
                        
                        if storage_path:
                            self.uploaded_count += 1
                            print(f"    ✓ Upload complete (Total: {self.uploaded_count})")
                            return True
                        else:
                            print("    ⚠ Upload to Supabase failed")
                            return False
                    else:
                        print("    ⚠ No new file detected — download likely blocked or failed")
                        return False

                except Exception: continue
            
            if not download_clicked: print("    ⚠ Could not click download button")
            return False
        except Exception as e:
            print(f"    Error in download dialog: {e}")
            return False
    
    def process_download_icons(self):
        """Find and process all download icons on current page"""
        try:
            icon_selectors = ["//mat-icon[text()='download']", "//mat-icon[contains(@mattooltip, 'Descargar')]"]
            icons = []
            for selector in icon_selectors:
                try:
                    found = self.driver.find_elements(By.XPATH, selector)
                    icons.extend(found)
                except: continue
            
            icons = list({id(icon): icon for icon in icons}.values())
            if not icons: return 0
            
            print(f"  Found {len(icons)} download icon(s)")
            
            success_count = 0
            for i, icon in enumerate(icons[:10], 1):
                try:
                    print(f"  Processing download {i}/{len(icons)}...")
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", icon)
                    time.sleep(1)
                    
                    pdf_identifier = self.normalize_url(self.driver.current_url)
                    print(f"    PDF Identifier: {pdf_identifier}")
                    
                    self.driver.execute_script("arguments[0].click();", icon)
                    time.sleep(2)
                    
                    if self.handle_download_dialog(pdf_identifier):
                        success_count += 1
                    
                    try:
                        close_btns = self.driver.find_elements(By.XPATH, "//button[contains(@class, 'close') or contains(., 'Cerrar') or contains(., 'Cancelar')]")
                        if close_btns:
                            close_btns[0].click()
                            time.sleep(1)
                    except: pass
                except Exception as e:
                    print(f"    Error processing download {i}: {e}")
                    continue
            
            return success_count
        except Exception as e:
            print(f"  Error in process_download_icons: {e}")
            return 0
    
    def extract_all_links(self, current_url):
        """Extract all internal links from current page"""
        links = set()
        try:
            elements = self.driver.find_elements(By.TAG_NAME, 'a')
            for elem in elements:
                try:
                    href = elem.get_attribute('href')
                    if not href or not href.startswith(self.base_url) or '.pdf' in href.lower(): continue
                    if self.is_same_page_section(current_url, href): continue
                    
                    normalized = self.normalize_url(href)
                    if normalized not in self.visited_urls:
                        links.add(normalized)
                except: continue
        except Exception as e:
            print(f"  Error extracting links: {e}")
        return links
    
    def extract_pdf_urls(self):
        """Extract all PDF URLs from current page"""
        try:
            elements = self.driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            for elem in elements:
                try:
                    href = elem.get_attribute('href')
                    if href and '.pdf' in href.lower():
                        if href.startswith(self.base_url):
                            self.pdf_urls.add(href)
                        else:
                            self.external_pdf_urls.add(href)
                except: continue
        except Exception as e:
            print(f"  Error extracting PDF URLs: {e}")
    
    def crawl_page(self, url, depth, max_depth):
        """Crawl a single page: download PDFs and extract links based on depth"""
        if url in self.visited_urls: return
        
        print(f"\n{'='*70}")
        print(f"Crawling (Depth {depth}): {url}")
        print(f"Progress: {len(self.visited_urls)} visited, {self.uploaded_count} uploaded, {len(self.url_queue)} in queue")
        print(f"{'='*70}")
        
        self.visited_urls.add(url)
        
        try:
            self.driver.get(url)
            self.wait_for_page_load()
            
            self.extract_pdf_urls()
            self.process_download_icons()
            
            if depth < max_depth:
                print(f"  Current depth ({depth}) is less than max depth ({max_depth}). Searching for new links...")
                new_links = self.extract_all_links(url)
                for link in new_links:
                    if link not in self.visited_urls and not any(item[0] == link for item in self.url_queue):
                        self.url_queue.append((link, depth + 1))
                print(f"  Found {len(new_links)} new link(s) to explore at depth {depth + 1}")
            else:
                print(f"  Reached max depth ({max_depth}). Not searching for more links from this page.")
            
        except Exception as e:
            print(f"  ✗ Error crawling page: {e}")
            
    def deep_crawl(self, start_urls, max_depth):
        """Deep crawl starting from list of URLs using BFS with max depth"""
        for url in start_urls:
            self.url_queue.append((self.normalize_url(url), 0))
        
        print(f"\n{'*'*70}\nStarting deep crawl up to depth {max_depth}\n{'*'*70}\n")
        
        while self.url_queue:
            url, depth = self.url_queue.popleft()
            self.crawl_page(url, depth, max_depth)
        
        print(f"\n{'*'*70}\nDeep crawl complete!\n{'*'*70}")

    def scrape(self, start_urls=None, max_depth=2):
        """Main scraping method"""
        if start_urls is None:
            start_urls = ["https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682716"]
        
        try:
            self.driver.get("https://spij.minjus.gob.pe/spij-ext-web/#/sidenav/legislacion")
            self.wait_for_page_load()
            
            if not self.click_ingresar_button(): return
            self.click_legislation_by_subject()
            
            self.deep_crawl(start_urls, max_depth)
            
        finally:
            self.driver.quit()
            # Clean up temp directory
            try:
                import shutil
                shutil.rmtree(self.temp_download_dir, ignore_errors=True)
                print(f"✓ Cleaned up temporary directory")
            except:
                pass
        
        # Upload hashes.json BEFORE saving results
        self.upload_hashes_json()
        
        self.save_results()
        self.print_summary()
        
    def save_results(self, filename='spij_results.json'):
        """Save results to a JSON file in current directory and upload to Supabase"""
        results = {
            'summary': {
                'pages_visited': len(self.visited_urls),
                'pdfs_uploaded': self.uploaded_count,
                'unique_pdfs_tracked': len(self.uploaded_pdfs),
                'internal_pdf_urls': len(self.pdf_urls),
                'external_pdf_urls': len(self.external_pdf_urls),
                'supabase_bucket': self.supabase_bucket,
                'subject_folder': self.subject_folder
            },
            'visited_urls': sorted(list(self.visited_urls)),
            'internal_pdf_urls': sorted(list(self.pdf_urls)),
            'external_pdf_urls': sorted(list(self.external_pdf_urls)),
            'uploaded_pdfs_identifiers': sorted(list(self._processed_identifiers))
        }
        
        # Save locally
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved locally to {filename}")
        
        # Upload results JSON to Supabase
        try:
            storage_path = f"{self.subject_folder}/{filename}"
            with open(filename, 'rb') as f:
                self.supabase.storage.from_(self.supabase_bucket).upload(
                    path=storage_path,
                    file=f.read(),
                    file_options={"content-type": "application/json", "upsert": "true"}
                )
            print(f"Results also uploaded to Supabase: {storage_path}")
        except Exception as e:
            print(f"Could not upload results to Supabase: {e}")

    def print_summary(self):
        """Prints a final summary of the crawl."""
        print(f"\n{'='*70}\nSCRAPING COMPLETE - SUMMARY\n{'='*70}")
        print(f"Pages visited: {len(self.visited_urls)}")
        print(f"PDFs uploaded to Supabase: {self.uploaded_count}")
        print(f"Unique PDFs tracked: {len(self.uploaded_pdfs)}")
        print(f"Internal PDF URLs found: {len(self.pdf_urls)}")
        print(f"External PDF URLs found: {len(self.external_pdf_urls)}")
        print(f"Supabase bucket: {self.supabase_bucket}")
        print(f"Subject folder: {self.subject_folder}\n{'='*70}\n")


# Central dictionary mapping subjects to their starting URLs
URLS_BY_SUBJECT = {
    # "Anti-corruption": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682716",
    # "Anti-terrorism": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682717",
    # "Commercial": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682721",
    # "Constitutional": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682723",
    "State Contracts and Acquisitions": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682724",
    # "Free Competition": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682762",
    # "Consumer Protection": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682779",
    # "Registry": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682784",
    # "Taxation": "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H682804",
}

def get_llm_subject_match(subject_query: str, all_subjects: list) -> str | None:
    """
    Uses an OpenAI LLM to find the best subject match from a list.

    Args:
        subject_query (str): The user's input string (e.g., "tax laws").
        all_subjects (list): The list of valid subject names.

    Returns:
        str | None: The best matching subject name from the list, or None if no match is found.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your key.")
        return None

    try:
        client = openai.OpenAI(api_key=api_key)
        subject_list_str = "\n".join(f"- {s}" for s in all_subjects)
        
        prompt = f"""
        You are an expert text-matching assistant. Your task is to find the single best match for a user's query from a predefined list of subjects.
        
        User Query: "{subject_query}"
        
        Predefined Subject List:
        {subject_list_str}
        
        Based on the user query, which one subject from the list is the most relevant match?
        Respond with ONLY the name of the matched subject from the list and nothing else. Do not add any explanation or punctuation.
        """
        
        print(f"Asking LLM to match query: '{subject_query}'...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful text-matching assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        
        llm_match = response.choices[0].message.content.strip()

        if llm_match in all_subjects:
            return llm_match
        else:
            print(f"Warning: LLM returned an invalid subject '{llm_match}'. Falling back to search.")
            for subject in all_subjects:
                if subject.lower() in llm_match.lower() or llm_match.lower() in subject.lower():
                     return subject
            return None

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return None

def scrape_by_subject(subject_query, max_depth=1, headless=False):
    """
    Finds a subject using an LLM and starts the scraper with Supabase upload.
    Returns the matched subject name on success, None on failure.
    """
    all_subjects = list(URLS_BY_SUBJECT.keys())
    
    matched_subject = get_llm_subject_match(subject_query, all_subjects)

    if not matched_subject:
        print(f"Error: Could not find a valid subject matching '{subject_query}' using the LLM.")
        print("Available subjects are:", ", ".join(all_subjects))
        return None

    start_url = URLS_BY_SUBJECT[matched_subject]
    
    print(f"\n--- LLM Matched '{subject_query}' to subject: '{matched_subject}' ---")
    
    try:
        # Initialize and run the scraper - PDFs will be uploaded to Supabase in subject-specific folder
        scraper = SPIJScraper(headless=headless, subject_folder=matched_subject)
        scraper.scrape(start_urls=[start_url], max_depth=max_depth)
        return matched_subject
    except Exception as e:
        print(f"✗ Scraping failed: {e}")
        return None



def run_scraper():
    """
    Entry point for Render scheduler or FastAPI background trigger.
    Automatically scrapes all subjects in URLS_BY_SUBJECT (headless mode).
    """
    print("="*80)
    print("Starting LUKE Regulatory Scraper (Render Trigger Mode)")
    print("="*80)

    try:
        for subject in URLS_BY_SUBJECT.keys():
            print(f"\n--- Starting scrape for subject: {subject} ---")
            scrape_by_subject(subject_query=subject, max_depth=1, headless=True)
        print("\n✓ All subjects scraped successfully.")
    except Exception as e:
        print(f"✗ Error running scraper: {e}")
    finally:
        print("="*80)
        print("Scraper run complete.")
        print("="*80)



def main():
    """
    Example usage of the subject-based scraper with LLM matching and Supabase upload.
    """
    # Scrape using a natural language query
    # The LLM will match this to "State Contracts and Acquisitions"
    scrape_by_subject(
        subject_query="laws about state contracts", 
        max_depth=1,
        headless=False
    )
    
    # Another Example: Scrape for competition laws
    # scrape_by_subject(
    #     subject_query="competition", 
    #     max_depth=1,
    #     headless=False
    # )

if __name__ == "__main__":
    main()

