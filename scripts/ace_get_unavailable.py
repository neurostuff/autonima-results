import argparse
import random
import time
from pathlib import Path
from ace import scrape
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


CHALLENGE_PATTERNS = (
    "<title>client challenge</title>",
    "/_fs-ch-",
    "javascript is disabled in your browser",
    "a required part of this site couldn",
    "please contact the service administrator",
)


def _looks_like_client_challenge(html):
    if not html:
        return False
    html_lower = html.lower()
    return any(pattern in html_lower for pattern in CHALLENGE_PATTERNS)


_ORIGINAL_VALIDATE_SCRAPE = scrape._validate_scrape


def _validate_scrape_with_client_challenge(html):
    if _looks_like_client_challenge(html):
        return False
    return _ORIGINAL_VALIDATE_SCRAPE(html)


scrape._validate_scrape = _validate_scrape_with_client_challenge


class ChallengeAwareScraper(scrape.Scraper):
    def __init__(self, store, api_key=None, browser_retries=4, challenge_timeout=35.0):
        super().__init__(store, api_key=api_key)
        self.browser_retries = max(1, int(browser_retries))
        self.challenge_timeout = max(5.0, float(challenge_timeout))

    def _new_driver(self, headless):
        return scrape.Driver(
            uc=True,
            headless2=headless,
            agent=random.choice(scrape.USER_AGENTS),
        )

    @staticmethod
    def _safe_page_source(driver, retries=3):
        for _ in range(retries):
            try:
                return driver.page_source
            except WebDriverException:
                time.sleep(0.8)
        return ""

    def _open_with_reconnect(self, driver, url, attempt):
        if hasattr(driver, "uc_open_with_reconnect"):
            reconnect_time = min(14, 5 + attempt * 2)
            driver.uc_open_with_reconnect(url, reconnect_time=reconnect_time)
            return
        driver.get(url)

    def _wait_for_content(self, driver, timeout):
        deadline = time.time() + timeout
        last_html = self._safe_page_source(driver)
        stable_non_challenge_samples = 0

        while time.time() < deadline:
            html = self._safe_page_source(driver)
            if html:
                last_html = html

            if _looks_like_client_challenge(html):
                stable_non_challenge_samples = 0
                time.sleep(1.0)
                continue

            try:
                ready = driver.execute_script("return document.readyState")
            except Exception:
                ready = "complete"

            if ready in ("interactive", "complete"):
                stable_non_challenge_samples += 1
                if stable_non_challenge_samples >= 2:
                    return html
            else:
                stable_non_challenge_samples = 0

            time.sleep(1.0)

        return last_html

    def _load_article_html(self, driver, url, journal, attempt):
        driver.set_page_load_timeout(20)
        self._open_with_reconnect(driver, url, attempt)
        resolved_url = driver.current_url

        html = self._wait_for_content(driver, timeout=self.challenge_timeout)
        substitute_url = self.check_for_substitute_url(resolved_url, html, journal)
        if substitute_url != resolved_url:
            self._open_with_reconnect(driver, substitute_url, attempt)
            html = self._wait_for_content(driver, timeout=self.challenge_timeout)

        journal_name = journal.lower()
        if journal_name in [
            "human brain mapping",
            "european journal of neuroscience",
            "brain and behavior",
            "epilepsia",
        ]:
            try:
                WebDriverWait(driver, 6).until(
                    EC.presence_of_element_located((By.ID, "relatedArticles"))
                )
            except TimeoutException:
                pass

        if journal_name in ["journal of neuroscience", "j neurosci"]:
            table_links = driver.find_elements(By.CLASS_NAME, "table-expand-inline")
            for link in table_links:
                try:
                    driver.execute_script("arguments[0].scrollIntoView();", link)
                    link.click()
                    time.sleep(0.5 + random.random())
                except Exception:
                    continue
        elif " - ScienceDirect" in html:
            try:
                WebDriverWait(driver, 7).until(
                    EC.presence_of_element_located((By.ID, "abstracts"))
                )
            except TimeoutException:
                pass
        elif "Wiley Online Library</title>" in html:
            try:
                WebDriverWait(driver, 7).until(
                    EC.presence_of_element_located((By.ID, "article__content"))
                )
            except TimeoutException:
                pass

        return self._wait_for_content(driver, timeout=min(12.0, self.challenge_timeout))

    def get_html(self, url, journal, mode="browser", headless=True):
        if mode != "browser":
            return super().get_html(url, journal, mode=mode, headless=headless)

        last_html = None
        for attempt in range(1, self.browser_retries + 1):
            driver = None
            try:
                driver = self._new_driver(headless=headless)
                html = self._load_article_html(driver, url, journal, attempt)
                if html:
                    last_html = html
                if html and not _looks_like_client_challenge(html):
                    return html
                scrape.logger.info(
                    "Detected client challenge/interstitial for %s (attempt %s/%s).",
                    journal,
                    attempt,
                    self.browser_retries,
                )
            except TimeoutException:
                scrape.logger.info(
                    "Timeout while loading %s (attempt %s/%s).",
                    journal,
                    attempt,
                    self.browser_retries,
                )
            except Exception as err:
                scrape.logger.info(
                    "Browser scrape attempt failed (%s/%s): %s",
                    attempt,
                    self.browser_retries,
                    err,
                )
            finally:
                if driver is not None:
                    try:
                        driver.quit()
                    except Exception:
                        pass

            if attempt < self.browser_retries:
                backoff_seconds = min(12.0, 2.0 * attempt + random.random())
                time.sleep(backoff_seconds)

        return last_html


def main():
    parser = argparse.ArgumentParser(
        description='Retrieve unavailable articles by PMID'
    )
    parser.add_argument(
        'scrape_path',
        help='Path to store scraped articles'
    )
    parser.add_argument(
        'pmid_file',
        nargs='?',
        help='File containing PMIDs (one per line)'
    )
    parser.add_argument(
        '--pmids',
        nargs='+',
        help='List of PMIDs to process'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=3.0,
        help='Delay between requests (default: 3.0)'
    )
    parser.add_argument(
        '--mode',
        choices=['browser', 'requests'],
        default='browser',
        help='Scraping mode (default: browser)'
    )
    parser.add_argument(
        '--prefer-pmc-source',
        action='store_true',
        default=True,
        help='Prefer PMC source when available (default: True)'
    )
    parser.add_argument(
        '--no-prefer-pmc-source',
        action='store_false',
        dest='prefer_pmc_source',
        help='Do not prefer PMC source'
    )
    parser.add_argument(
        '--metadata-store',
        help='Path to store metadata (default: scrape_path/metadata)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=False,
        help='Run browser in headless mode (default: False)'
    )
    parser.add_argument(
        '--browser-retries',
        type=int,
        default=4,
        help='Max browser retries for anti-bot/challenge pages (default: 4)'
    )
    parser.add_argument(
        '--challenge-timeout',
        type=float,
        default=35.0,
        help='Seconds to wait for challenge pages to resolve (default: 35.0)'
    )
    
    args = parser.parse_args()
    
    scrape_path = args.scrape_path
    
    # Get PMIDs either from file or command line
    if args.pmids:
        pmids = args.pmids
    elif args.pmid_file:
        print(f"Reading PMIDs from {args.pmid_file}...")
        with open(args.pmid_file, 'r') as f:
            pmids = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Either pmid_file or --pmids must be provided")
    
    print(f"Found {len(pmids)} PMIDs to process.")
    
    # Determine metadata store path
    if args.metadata_store:
        metadata_store = Path(args.metadata_store)
    else:
        metadata_store = Path(scrape_path) / 'metadata'
    
    # Initialize scraper
    scraper = ChallengeAwareScraper(
        scrape_path,
        browser_retries=args.browser_retries,
        challenge_timeout=args.challenge_timeout,
    )
    
    # Retrieve articles by PMID list
    invalid_articles = scraper.retrieve_articles(
        pmids=pmids,
        delay=args.delay,
        mode=args.mode,
        prefer_pmc_source=args.prefer_pmc_source,
        metadata_store=metadata_store,
        headless=args.headless
    )
    
    print("\nProcessing complete!")
    print(f"Invalid articles: {len(invalid_articles)}")
    
    if invalid_articles:
        # Save invalid articles to a file
        invalid_file = Path(scrape_path) / 'invalid_pmids.txt'
        with open(invalid_file, 'w') as f:
            for pmid in invalid_articles:
                f.write(f"{pmid}\n")
        print(f"Invalid PMIDs saved to {invalid_file}")


if __name__ == '__main__':
    main()
