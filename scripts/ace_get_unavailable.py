import argparse
import logging
import os
import random
import re
import shutil
import socket
import time
from pathlib import Path
from ace import scrape
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


CHALLENGE_PATTERNS = (
    "<title>client challenge</title>",
    "<title>just a moment...</title>",
    "<title>attention required!</title>",
    "checking if you are a human",
    "checking if the site connection is secure",
    "enable javascript and cookies to continue",
    "please turn javascript on and reload the page",
    "verify you are human",
    "cf-chl-",
    "cf-turnstile",
    "__cf_bm",
    "cloudflare ray id",
    "/_fs-ch-",
)


class SkipURLRequested(Exception):
    """Raised when a URL should be skipped based on configured substrings."""


def _looks_like_client_challenge(html):
    if not html:
        return False
    html_lower = html.lower()
    return any(pattern in html_lower for pattern in CHALLENGE_PATTERNS)


def _get_challenge_match_details(html, context_chars=70, max_excerpt_chars=180):
    if not html:
        return None
    html_lower = html.lower()
    for pattern in CHALLENGE_PATTERNS:
        start_idx = html_lower.find(pattern)
        if start_idx == -1:
            continue
        end_idx = start_idx + len(pattern)
        excerpt_start = max(0, start_idx - context_chars)
        excerpt_end = min(len(html), end_idx + context_chars)
        excerpt = " ".join(html[excerpt_start:excerpt_end].split())
        if len(excerpt) > max_excerpt_chars:
            excerpt = excerpt[: max_excerpt_chars - 3] + "..."
        return pattern, excerpt
    return None


_ORIGINAL_VALIDATE_SCRAPE = scrape._validate_scrape


def _validate_scrape_with_client_challenge(html):
    if _looks_like_client_challenge(html):
        return False
    return _ORIGINAL_VALIDATE_SCRAPE(html)


def _is_valid_scrape(html):
    return bool(html) and _validate_scrape_with_client_challenge(html)


scrape._validate_scrape = _validate_scrape_with_client_challenge


class ChallengeAwareScraper(scrape.Scraper):
    def __init__(
        self,
        store,
        api_key=None,
        browser="chrome",
        firefox_binary=None,
        browser_retries=4,
        challenge_timeout=35.0,
        page_load_timeout=20.0,
        wiley_content_timeout=7.0,
        final_content_timeout=12.0,
        skip_on_challenge=False,
        use_uc_reconnect=True,
        use_uc=True,
        uc_debug_port=0,
    ):
        super().__init__(store, api_key=api_key)
        self.browser = str(browser).strip().lower()
        if self.browser not in {"chrome", "firefox"}:
            raise ValueError(f"Unsupported browser: {self.browser!r}. Use 'chrome' or 'firefox'.")
        self.firefox_binary = firefox_binary
        self.browser_retries = max(1, int(browser_retries))
        self.challenge_timeout = max(5.0, float(challenge_timeout))
        self.page_load_timeout = max(1.0, float(page_load_timeout))
        self.wiley_content_timeout = max(0.0, float(wiley_content_timeout))
        self.final_content_timeout = max(1.0, float(final_content_timeout))
        self.skip_on_challenge = bool(skip_on_challenge)
        self.use_uc_reconnect = bool(use_uc_reconnect)
        self.use_uc = bool(use_uc)
        self.uc_debug_port = int(uc_debug_port)

    @staticmethod
    def _pick_free_local_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.getsockname()[1]

    @staticmethod
    def _is_probably_elf_binary(path):
        if os.name != "posix":
            return True
        try:
            with open(path, "rb") as f:
                return f.read(4) == b"\x7fELF"
        except OSError:
            return False

    def _resolve_firefox_binary(self):
        if self.firefox_binary:
            binary = str(self.firefox_binary).strip()
            if not os.path.exists(binary):
                raise ValueError(f"Firefox binary does not exist: {binary}")
            return binary

        candidates = [
            "/snap/firefox/current/usr/lib/firefox/firefox",
            "/usr/lib/firefox/firefox",
            "/usr/lib/firefox-esr/firefox-esr",
            "/opt/firefox/firefox",
        ]
        discovered = shutil.which("firefox")
        if discovered:
            candidates.append(discovered)
        candidates.extend(
            [
                "/usr/local/bin/firefox",
                "/usr/bin/firefox",
            ]
        )
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if not os.path.exists(candidate):
                continue
            if self._is_probably_elf_binary(candidate):
                return candidate

        return discovered

    def _new_driver(self, headless):
        use_uc = self.use_uc and self.browser == "chrome"
        if self.use_uc and self.browser != "chrome":
            scrape.logger.info(
                "UC mode requested, but %s does not support UC. Continuing without UC.",
                self.browser,
            )
        scrape.logger.info(
            "Initializing browser driver (browser=%s, uc=%s, headless=%s).",
            self.browser,
            use_uc,
            headless,
        )
        if self.browser == "firefox":
            firefox_options = FirefoxOptions()
            user_agent = random.choice(scrape.USER_AGENTS)
            firefox_options.set_preference("general.useragent.override", user_agent)
            if headless:
                firefox_options.add_argument("-headless")
            firefox_binary = self._resolve_firefox_binary()
            if firefox_binary:
                scrape.logger.info("Using Firefox binary: %s", firefox_binary)
                firefox_options.binary_location = firefox_binary
            return webdriver.Firefox(options=firefox_options)

        driver_kwargs = {
            "browser": self.browser,
            "agent": random.choice(scrape.USER_AGENTS),
        }
        if self.browser == "chrome":
            driver_kwargs["uc"] = use_uc
            driver_kwargs["headless2"] = headless
        else:
            driver_kwargs["headless"] = headless

        if use_uc:
            uc_port = self.uc_debug_port or self._pick_free_local_port()
            driver_kwargs["chromium_arg"] = f"remote-debugging-port={uc_port}"
            scrape.logger.info("Using UC remote debugging port: %s", uc_port)
        return scrape.Driver(**driver_kwargs)

    @staticmethod
    def _safe_page_source(driver, retries=3):
        for _ in range(retries):
            try:
                return driver.page_source
            except WebDriverException:
                time.sleep(0.8)
        return ""

    @staticmethod
    def _normalize_skip_url_substrings(skip_url_substrings):
        if skip_url_substrings is None:
            return tuple()
        if isinstance(skip_url_substrings, str):
            values = [skip_url_substrings]
        else:
            values = skip_url_substrings
        normalized = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip().lower()
            if text:
                normalized.append(text)
        return tuple(normalized)

    @staticmethod
    def _match_skip_substring(url, skip_url_substrings):
        if not url:
            return None
        url_lower = str(url).lower()
        for substring in skip_url_substrings:
            if substring in url_lower:
                return substring
        return None

    def _raise_if_skipped_url(self, url, skip_url_substrings, context):
        matched_substring = self._match_skip_substring(url, skip_url_substrings)
        if matched_substring:
            self._skip_article_requested = True
            self._skip_article_due_to_url = True
            scrape.logger.info(
                "Skipping URL due to configured substring %r (%s): %s",
                matched_substring,
                context,
                url,
            )
            raise SkipURLRequested()

    def _raise_if_skipped_url_in_html(self, html, skip_url_substrings, context):
        if not html or not skip_url_substrings:
            return
        for candidate_url in set(re.findall(r'https?://[^"\'\s<>]+', html)):
            matched_substring = self._match_skip_substring(candidate_url, skip_url_substrings)
            if not matched_substring:
                continue
            self._skip_article_requested = True
            self._skip_article_due_to_url = True
            scrape.logger.info(
                "Skipping URL due to configured substring %r (%s, discovered in HTML): %s",
                matched_substring,
                context,
                candidate_url,
            )
            raise SkipURLRequested()

    def _open_with_reconnect(self, driver, url, attempt):
        if (
            self.browser == "chrome"
            and self.use_uc
            and self.use_uc_reconnect
            and hasattr(driver, "uc_open_with_reconnect")
        ):
            reconnect_time = min(14, 5 + attempt * 2)
            scrape.logger.info(
                "Opening URL with uc reconnect (attempt %s, reconnect=%ss): %s",
                attempt,
                reconnect_time,
                url,
            )
            driver.uc_open_with_reconnect(url, reconnect_time=reconnect_time)
            return
        scrape.logger.info("Opening URL with standard driver.get (attempt %s): %s", attempt, url)
        driver.get(url)

    def _wait_for_content(self, driver, timeout, ready_markers=None):
        deadline = time.time() + timeout
        last_html = self._safe_page_source(driver)
        stable_non_challenge_samples = 0
        next_status_log = time.time() + 5.0
        markers = tuple((marker or "").lower() for marker in (ready_markers or ()) if marker)

        while time.time() < deadline:
            html = self._safe_page_source(driver)
            if html:
                last_html = html

            if _looks_like_client_challenge(html):
                if self.skip_on_challenge:
                    match_details = _get_challenge_match_details(html)
                    if match_details:
                        marker, excerpt = match_details
                        scrape.logger.info(
                            "Skipping %s after challenge/interstitial detection "
                            "(--skip-on-challenge enabled; marker=%r; excerpt=%r).",
                            "article",
                            marker,
                            excerpt,
                        )
                    else:
                        scrape.logger.info(
                            "Skipping article after challenge/interstitial detection "
                            "(--skip-on-challenge enabled)."
                        )
                    self._skip_article_requested = True
                    self._skip_article_due_to_challenge = True
                    raise SkipURLRequested()
                stable_non_challenge_samples = 0
                if time.time() >= next_status_log:
                    remaining = max(0.0, deadline - time.time())
                    match_details = _get_challenge_match_details(html)
                    if match_details:
                        marker, excerpt = match_details
                        scrape.logger.info(
                            "Still on challenge/interstitial page (%.1fs remaining; marker=%r; excerpt=%r).",
                            remaining,
                            marker,
                            excerpt,
                        )
                    else:
                        scrape.logger.info(
                            "Still on challenge/interstitial page (%.1fs remaining in wait window).",
                            remaining,
                        )
                    next_status_log = time.time() + 5.0
                time.sleep(1.0)
                continue

            if markers:
                html_lower = (html or "").lower()
                for marker in markers:
                    if marker in html_lower:
                        return html

            try:
                ready = driver.execute_script("return document.readyState")
            except Exception:
                ready = "complete"

            if time.time() >= next_status_log:
                remaining = max(0.0, deadline - time.time())
                scrape.logger.info(
                    "Waiting for page readiness: state=%s (%.1fs remaining).",
                    ready,
                    remaining,
                )
                next_status_log = time.time() + 5.0

            if ready in ("interactive", "complete"):
                stable_non_challenge_samples += 1
                if stable_non_challenge_samples >= 2:
                    return html
            else:
                stable_non_challenge_samples = 0

            time.sleep(1.0)

        return last_html

    def _load_article_html(self, driver, url, journal, attempt, skip_url_substrings):
        driver.set_page_load_timeout(self.page_load_timeout)
        scrape.logger.info("Loading article for %s (attempt %s).", journal, attempt)
        self._open_with_reconnect(driver, url, attempt)
        resolved_url = driver.current_url
        self._raise_if_skipped_url(
            resolved_url,
            skip_url_substrings,
            f"resolved URL for {journal} attempt {attempt}",
        )

        html = self._wait_for_content(driver, timeout=self.challenge_timeout)
        self._raise_if_skipped_url_in_html(
            html,
            skip_url_substrings,
            f"loaded HTML for {journal} attempt {attempt}",
        )
        scrape.logger.info("Initial page load complete for %s (attempt %s).", journal, attempt)
        substitute_url = self.check_for_substitute_url(resolved_url, html, journal)
        if substitute_url != resolved_url:
            self._raise_if_skipped_url(
                substitute_url,
                skip_url_substrings,
                f"substitute URL for {journal} attempt {attempt}",
            )
            scrape.logger.info("Following substitute URL for %s: %s", journal, substitute_url)
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
            if self.wiley_content_timeout > 0:
                try:
                    WebDriverWait(driver, self.wiley_content_timeout).until(
                        EC.presence_of_element_located((By.ID, "article__content"))
                    )
                except TimeoutException:
                    pass
            return self._wait_for_content(
                driver,
                timeout=min(self.final_content_timeout, self.challenge_timeout),
                ready_markers=('id="article__content"', "id='article__content'"),
            )

        return self._wait_for_content(
            driver,
            timeout=min(self.final_content_timeout, self.challenge_timeout),
        )

    def get_html(self, url, journal, mode="browser", headless=True, skip_url_substrings=None):
        skip_url_substrings = self._normalize_skip_url_substrings(skip_url_substrings)
        self._raise_if_skipped_url(url, skip_url_substrings, f"initial URL for {journal}")

        if mode != "browser":
            return super().get_html(
                url,
                journal,
                mode=mode,
                headless=headless,
                skip_url_substrings=skip_url_substrings,
            )

        last_html = None
        for attempt in range(1, self.browser_retries + 1):
            driver = None
            try:
                scrape.logger.info(
                    "Browser scrape attempt %s/%s for %s.",
                    attempt,
                    self.browser_retries,
                    journal,
                )
                driver = self._new_driver(headless=headless)
                scrape.logger.info(
                    "Browser driver initialized for %s (attempt %s/%s).",
                    journal,
                    attempt,
                    self.browser_retries,
                )
                html = self._load_article_html(
                    driver,
                    url,
                    journal,
                    attempt,
                    skip_url_substrings=skip_url_substrings,
                )
                if html:
                    last_html = html
                if _is_valid_scrape(html):
                    scrape.logger.info(
                        "Successfully retrieved valid HTML for %s on attempt %s/%s.",
                        journal,
                        attempt,
                        self.browser_retries,
                    )
                    return html
                if html and _looks_like_client_challenge(html):
                    match_details = _get_challenge_match_details(html)
                    if match_details:
                        marker, excerpt = match_details
                        scrape.logger.info(
                            "Detected client challenge/interstitial for %s (attempt %s/%s, marker=%r, excerpt=%r).",
                            journal,
                            attempt,
                            self.browser_retries,
                            marker,
                            excerpt,
                        )
                    else:
                        scrape.logger.info(
                            "Detected client challenge/interstitial for %s (attempt %s/%s).",
                            journal,
                            attempt,
                            self.browser_retries,
                        )
                    if self.skip_on_challenge:
                        self._skip_article_requested = True
                        self._skip_article_due_to_challenge = True
                        scrape.logger.info(
                            "Skipping %s after challenge/interstitial detection "
                            "(--skip-on-challenge enabled).",
                            journal,
                        )
                        return None
                else:
                    scrape.logger.info(
                        "Retrieved HTML for %s failed ACE validation (attempt %s/%s); likely interstitial/blocked page.",
                        journal,
                        attempt,
                        self.browser_retries,
                    )
            except SkipURLRequested:
                return None
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
                scrape.logger.info(
                    "Retrying %s after %.1fs backoff (next attempt %s/%s).",
                    journal,
                    backoff_seconds,
                    attempt + 1,
                    self.browser_retries,
                )
                time.sleep(backoff_seconds)

        return last_html


def main():
    def _parse_prefer_pmc_source(value):
        if isinstance(value, bool):
            return value
        value_normalized = str(value).strip().lower()
        if value_normalized in {'true', '1', 'yes', 'y'}:
            return True
        if value_normalized in {'false', '0', 'no', 'n'}:
            return False
        if value_normalized == 'only':
            return 'only'
        raise argparse.ArgumentTypeError(
            "Invalid value for --prefer-pmc-source. Use true, false, or only."
        )

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
        '--browser',
        choices=['chrome', 'firefox'],
        default='chrome',
        help='Browser engine for Selenium mode (default: chrome)'
    )
    parser.add_argument(
        '--firefox-binary',
        help='Path to Firefox executable when using --browser firefox'
    )
    parser.add_argument(
        '--prefer-pmc-source',
        nargs='?',
        const=True,
        default=True,
        type=_parse_prefer_pmc_source,
        metavar='{true,false,only}',
        help='Prefer PMC source when available. Use "only" to fetch only articles with PMC source (default: true)'
    )
    parser.add_argument(
        '--no-prefer-pmc-source',
        action='store_false',
        dest='prefer_pmc_source',
        help='Do not prefer PMC source (equivalent to --prefer-pmc-source false)'
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
        default=2,
        help='Max browser retries for anti-bot/challenge pages (default: 2)'
    )
    parser.add_argument(
        '--challenge-timeout',
        type=float,
        default=35.0,
        help='Seconds to wait for challenge pages to resolve (default: 35.0)'
    )
    parser.add_argument(
        '--skip-on-challenge',
        action='store_true',
        help='Skip an article immediately when a challenge/interstitial page is detected.'
    )
    parser.add_argument(
        '--page-load-timeout',
        type=float,
        default=12.0,
        help='Selenium page-load timeout in seconds per navigation (default: 12.0)'
    )
    parser.add_argument(
        '--wiley-content-timeout',
        type=float,
        default=4.0,
        help='Seconds to wait for Wiley article content element before final HTML capture (default: 4.0)'
    )
    parser.add_argument(
        '--final-content-timeout',
        type=float,
        default=5.0,
        help='Final post-load content wait in seconds after site-specific handling (default: 5.0)'
    )
    parser.add_argument(
        '--no-uc-reconnect',
        action='store_true',
        help='Disable uc_open_with_reconnect and use standard driver.get'
    )
    parser.add_argument(
        '--no-uc',
        action='store_true',
        help='Disable undetected browser mode (Chrome only; ignored for Firefox)'
    )
    parser.add_argument(
        '--uc-debug-port',
        type=int,
        default=0,
        help='UC remote debugging port (Chrome+UC only; default: 0 = auto-select per attempt)'
    )
    parser.add_argument(
        '--skip-url-substring',
        action='append',
        default=[],
        metavar='SUBSTRING',
        help='Skip retrieval when a resolved URL contains this substring. Repeat to add multiple values.'
    )
    parser.add_argument(
        '--skip-elsevier',
        action='store_true',
        help='Shortcut to skip Elsevier URLs (adds: elsevier, sciencedirect, linkinghub).'
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='warning',
        help='Logging level (default: warning)'
    )
    verbosity_group.add_argument(
        '--verbose',
        '--debug',
        dest='verbose',
        action='store_true',
        help='Enable info-level logging'
    )
    
    args = parser.parse_args()

    log_level_name = 'INFO' if args.verbose else args.log_level.upper()
    log_level = getattr(logging, log_level_name)
    logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)
    scrape.logger.setLevel(log_level)

    skip_url_substrings = [
        value.strip()
        for value in args.skip_url_substring
        if value and value.strip()
    ]
    if args.skip_elsevier:
        skip_url_substrings.extend(["elsevier", "sciencedirect", "linkinghub"])
    seen_substrings = set()
    skip_url_substrings = [
        value
        for value in skip_url_substrings
        if not (value.lower() in seen_substrings or seen_substrings.add(value.lower()))
    ]
    
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
        browser=args.browser,
        firefox_binary=args.firefox_binary,
        browser_retries=args.browser_retries,
        challenge_timeout=args.challenge_timeout,
        page_load_timeout=args.page_load_timeout,
        wiley_content_timeout=args.wiley_content_timeout,
        final_content_timeout=args.final_content_timeout,
        skip_on_challenge=args.skip_on_challenge,
        use_uc_reconnect=not args.no_uc_reconnect,
        use_uc=not args.no_uc,
        uc_debug_port=args.uc_debug_port,
    )
    
    # Retrieve articles by PMID list
    invalid_articles = scraper.retrieve_articles(
        pmids=pmids,
        delay=args.delay,
        mode=args.mode,
        prefer_pmc_source=args.prefer_pmc_source,
        metadata_store=metadata_store,
        headless=args.headless,
        skip_url_substrings=skip_url_substrings,
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
