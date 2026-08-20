#!/usr/bin/env python3
"""Open PubMed and/or publisher pages from a list of PMIDs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


PMID_RE = re.compile(r"^\d+$")
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}"
PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"
ELINK_PRLINKS_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    "?dbfrom=pubmed&retmode=json&cmd=prlinks&id={joined_pmids}"
)
PMC_IDCONV_URL = (
    "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    "?ids={joined_pmids}&idtype=pmid&format=json"
)
UT_EZPROXY_DOMAIN = "ezproxy.lib.utexas.edu"
DEFAULT_PROXY_PREFIX = "http://ezproxy.lib.utexas.edu/login?url="
SCIENCEDIRECT_HOSTS = {"www.sciencedirect.com", "sciencedirect.com"}
ELSEVIER_LINKING_HOSTS = {"linkinghub.elsevier.com"}
DEFAULT_USER_AGENT = "open_pubmed_in_browser/2.0 (+manual-fulltext-workflow)"
REDIRECT_PAGE_DIR = Path("downloaded_files/open_pubmed_redirects")
SAVE_BOOKMARKLET = (
    "javascript:(async function(){var pm='';"
    "async function lookup(term){"
    "try{"
    "var u='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=1&term='+encodeURIComponent(term);"
    "var r=await fetch(u,{credentials:'omit'});"
    "if(!r.ok){return '';}"
    "var j=await r.json();"
    "var ids=((j||{}).esearchresult||{}).idlist||[];"
    "return ids[0]||'';"
    "}catch(e){return '';}"
    "}"
    "function meta(name){"
    "var el=document.querySelector('meta[name=\"'+name+'\"]');"
    "return el?(el.getAttribute('content')||'').trim():'';"
    "}"
    "function download(){"
    "var html='<!doctype html>\\n'+document.documentElement.outerHTML;"
    "var blob=new Blob([html],{type:'text/html;charset=utf-8'});"
    "var a=document.createElement('a');"
    "a.href=URL.createObjectURL(blob);"
    "a.download=pm+'.html';"
    "document.body.appendChild(a);"
    "a.click();"
    "setTimeout(function(){URL.revokeObjectURL(a.href);a.remove();},1500);"
    "}"
    "var n=(window.name||'').match(/(?:^|[;&\\s])pmid=(\\d+)(?:$|[;&\\s])/i);"
    "if(n){pm=n[1];}"
    "var m=(location.href||'').match(/[?#&]pmid=(\\d+)/i);"
    "if(!pm&&m){pm=m[1];}"
    "if(!pm){m=location.pathname.match(/\\/(\\d{6,10})(?:\\/)?$/);if(m){pm=m[1];}}"
    "if(!pm){pm=meta('citation_pmid')||meta('pmid');}"
    "if(!pm){"
    "var doi=meta('citation_doi')||meta('dc.identifier');"
    "doi=doi.replace(/^https?:\\/\\/doi\\.org\\//i,'').replace(/^doi\\s*:\\s*/i,'').trim();"
    "if(doi){pm=await lookup(doi+'[doi]');}"
    "}"
    "if(!pm){"
    "var pii=meta('citation_pii');"
    "if(pii){pm=await lookup('\"'+pii+'\"[pii]');}"
    "}"
    "if(!pm){pm=prompt('PMID for filename:');}"
    "if(!pm){return;}"
    "download();"
    "})();"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open PMIDs in a browser and optionally resolve direct publisher links "
            "via NCBI E-utilities."
        )
    )
    parser.add_argument("pmid_file", type=Path, help="Path to a text file with one PMID per line.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs instead of opening browser windows.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between opening individual URLs (default: 0).",
    )
    parser.add_argument(
        "--max-at-once",
        type=int,
        default=10,
        help=(
            "Maximum number of PMIDs to process per batch before prompting to continue "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--browser",
        choices=["firefox", "chrome"],
        default="firefox",
        help="Browser to use for opening pages (default: firefox).",
    )
    parser.add_argument(
        "--mode",
        choices=["pubmed", "publisher", "both"],
        default="publisher",
        help=(
            "Which pages to open per PMID: pubmed page only, direct publisher link only, "
            "or both (default: pubmed)."
        ),
    )
    parser.add_argument(
        "--elsevier-filter",
        choices=["all", "exclude", "only"],
        default="all",
        help=(
            "Filter PMIDs by publisher when resolving publisher links: all (default), "
            "exclude Elsevier/ScienceDirect, or only Elsevier/ScienceDirect."
        ),
    )
    parser.add_argument(
        "--pmc-filter",
        choices=["all", "exclude", "only"],
        default="all",
        help=(
            "Filter PMIDs by PubMed Central availability: all (default), "
            "exclude PubMed Central links, or only PubMed Central links. "
            "Whenever PMCID is available, direct PMC article URLs are opened."
        ),
    )
    parser.add_argument(
        "--fallback-to-pubmed",
        dest="fallback_to_pubmed",
        action="store_true",
        default=True,
        help="If publisher URL cannot be resolved, open PubMed URL instead (default: enabled).",
    )
    parser.add_argument(
        "--no-fallback-to-pubmed",
        dest="fallback_to_pubmed",
        action="store_false",
        help="If publisher URL cannot be resolved, skip that PMID instead of opening PubMed.",
    )
    parser.add_argument(
        "--proxy-prefix",
        type=str,
        default=DEFAULT_PROXY_PREFIX,
        help=(
            "Enable proxy rewriting for opened URLs. For UT ezproxy values "
            "(contains 'ezproxy.lib.utexas.edu'), URLs are transformed to "
            "the hostname-rewrite format (e.g., www-sciencedirect-com.ezproxy...). "
            "For non-UT values, legacy prefix wrapping is used."
        ),
    )
    parser.add_argument(
        "--attach-pmid-fragment",
        dest="attach_pmid_fragment",
        action="store_true",
        default=True,
        help=(
            "Append '#pmid=<PMID>' to opened URLs and preserve PMID through redirects "
            "for the save bookmarklet (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-proxy",
        dest="proxy_prefix",
        action="store_const",
        const="",
        help="Disable proxy rewriting entirely (PMC/PubMed are never proxied regardless).",
    )
    parser.add_argument(
        "--no-attach-pmid-fragment",
        dest="attach_pmid_fragment",
        action="store_false",
        help="Disable appending '#pmid=<PMID>' to opened URLs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV path to write PMID -> opened URL manifest.",
    )
    parser.add_argument(
        "--firefox-profile",
        type=str,
        default=None,
        help=(
            "Launch Firefox with this profile directory (e.g. the SOCKS-proxied "
            "'beast-proxy' profile), leaving your default profile untouched."
        ),
    )
    parser.add_argument(
        "--done-dirs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Directories already holding <PMID>.html (searched recursively). PMIDs found "
            "there are reported as done and skipped unless --no-skip-done. Defaults to "
            "articles/ace_outputs/html and articles/elsevier_output when they exist."
        ),
    )
    parser.add_argument(
        "--no-skip-done",
        dest="skip_done",
        action="store_false",
        default=True,
        help="Report already-downloaded PMIDs but still open them (manual copy supersedes).",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help=(
            "CSV tracking file. Records pmid,status,checked_at so a session can resume; "
            "updated on each run with which PMIDs are still outstanding."
        ),
    )
    parser.add_argument(
        "--print-save-bookmarklet",
        action="store_true",
        help="Print a bookmarklet that saves the current page as PMID.html.",
    )
    parser.add_argument(
        "--elink-batch-size",
        type=int,
        default=100,
        help="PMIDs per E-utilities request when resolving publisher links (default: 100).",
    )
    parser.add_argument(
        "--elink-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for each E-utilities request (default: 20).",
    )
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Preserve PMID file order instead of sorting by PMID descending.",
    )
    return parser.parse_args()


def iter_pmids(path: Path) -> list[str]:
    pmids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not PMID_RE.fullmatch(line):
                print(f"Skipping invalid PMID on line {line_number}: {line}")
                continue
            if line in seen:
                continue
            seen.add(line)
            pmids.append(line)
    return pmids


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def pick_primary_publisher_url(objurls: list[dict]) -> str | None:
    for entry in objurls:
        value = str(entry.get("url", {}).get("value") or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return None


def resolve_primary_publisher_urls(
    pmids: list[str], *, batch_size: int, timeout: float
) -> dict[str, str | None]:
    url_by_pmid: dict[str, str | None] = {pmid: None for pmid in pmids}
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    def fetch(batch: list[str], attempt: int = 1) -> dict | None:
        """One elink call with retries; on persistent failure split the batch."""
        endpoint = ELINK_PRLINKS_URL.format(joined_pmids=",".join(batch))
        try:
            with urlopen(Request(endpoint, headers=headers), timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if attempt < 4:
                sleep_for = 2.0 * attempt
                print(
                    f"[WARN] elink attempt {attempt} failed for batch of {len(batch)} "
                    f"(starting {batch[0]}): {exc}; retrying in {sleep_for:.0f}s"
                )
                time.sleep(sleep_for)
                return fetch(batch, attempt + 1)
            if len(batch) > 1:
                mid = len(batch) // 2
                print(
                    f"[WARN] elink still failing for batch of {len(batch)}; "
                    f"splitting into {mid} + {len(batch) - mid}"
                )
                merged: dict = {"linksets": []}
                for half in (batch[:mid], batch[mid:]):
                    got = fetch(half)
                    if got:
                        merged["linksets"].extend(got.get("linksets", []))
                return merged
            print(f"[WARN] elink permanently failed for PMID {batch[0]}: {exc}")
            return None

    for batch in chunked(pmids, batch_size):
        payload = fetch(batch)
        if not payload:
            continue

        for linkset in payload.get("linksets", []):
            for id_entry in linkset.get("idurllist", []):
                pmid = str(id_entry.get("id") or "").strip()
                if pmid not in url_by_pmid:
                    continue
                url_by_pmid[pmid] = pick_primary_publisher_url(id_entry.get("objurls", []))

    return url_by_pmid


def normalize_pmcid(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper().startswith("PMC"):
        text = text[3:]
    if not text:
        return None
    if not re.fullmatch(r"[A-Za-z0-9]+", text):
        return None
    return text


def resolve_pmcids(pmids: list[str], *, batch_size: int, timeout: float) -> dict[str, str | None]:
    pmcid_by_pmid: dict[str, str | None] = {pmid: None for pmid in pmids}
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    for batch in chunked(pmids, batch_size):
        joined_pmids = ",".join(batch)
        endpoint = PMC_IDCONV_URL.format(joined_pmids=joined_pmids)
        request = Request(endpoint, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to resolve PMCID for batch starting PMID {batch[0]}: {exc}")
            continue

        for record in payload.get("records", []):
            pmid = str(record.get("pmid") or record.get("requested-id") or "").strip()
            if pmid not in pmcid_by_pmid:
                continue
            pmcid_by_pmid[pmid] = normalize_pmcid(record.get("pmcid"))

    return pmcid_by_pmid


def pmcid_to_pmc_url(pmcid: str) -> str:
    return PMC_ARTICLE_URL.format(pmcid=pmcid)


def attach_pmid_fragment(url: str, pmid: str, enabled: bool) -> str:
    if not enabled:
        return url
    parts = urlsplit(url)
    fragment = parts.fragment or ""
    if "pmid=" in fragment:
        new_fragment = fragment
    elif fragment:
        new_fragment = f"{fragment}&pmid={pmid}"
    else:
        new_fragment = f"pmid={pmid}"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, new_fragment))


def wrap_with_pmid_redirect(url: str, pmid: str, enabled: bool) -> str:
    if not enabled:
        return url

    # Store the PMID in window.name before navigating so the bookmarklet can still
    # recover it after one or more publisher-side redirects.
    redirect_dir = REDIRECT_PAGE_DIR.expanduser().resolve()
    redirect_dir.mkdir(parents=True, exist_ok=True)
    redirect_path = redirect_dir / f"{pmid}.html"
    html = (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'>"
        f"<title>Opening PMID {pmid}</title>"
        "<meta http-equiv='refresh' content='0'>"
        "</head><body>"
        f"<p>Opening PMID {pmid}...</p>"
        "<script>"
        f"window.name='pmid={pmid}';"
        f"location.replace({json.dumps(url)});"
        "</script>"
        "</body></html>"
    )
    redirect_path.write_text(html, encoding="utf-8")
    return redirect_path.as_uri()


def is_elsevier_publisher_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return False
    return "elsevier.com" in host or "sciencedirect.com" in host


PMC_PROXY_EXEMPT_HOSTS = {
    "pmc.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "europepmc.org",
    "www.europepmc.org",
}


def is_proxy_exempt(url: str) -> bool:
    """PMC/PubMed are free and break under ezproxy; never proxy them."""
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return False
    return host in PMC_PROXY_EXEMPT_HOSTS or host.endswith(".ncbi.nlm.nih.gov")



def default_done_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    cands = [here / "articles" / "ace_outputs" / "html",
             here / "articles" / "elsevier_output"]
    return [c for c in cands if c.exists()]


def scan_done(dirs: list[Path]) -> dict[str, str]:
    """Map pmid -> where it was found. <pmid>.html files or <pmid>/ dirs both count."""
    found: dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.html"):
            stem = p.stem
            if PMID_RE.match(stem):
                found.setdefault(stem, str(d))
        for p in d.iterdir():
            if p.is_dir() and PMID_RE.match(p.name):
                found.setdefault(p.name, str(d))
    return found


def write_progress(path: Path, pmids: list[str], done: dict[str, str]) -> None:
    import datetime
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pmid", "status", "source", "checked_at"])
        for pm in pmids:
            w.writerow([pm, "done" if pm in done else "outstanding", done.get(pm, ""), ts])
    n_done = sum(1 for pm in pmids if pm in done)
    print(f"[progress] {n_done}/{len(pmids)} done, {len(pmids)-n_done} outstanding -> {path}")

def maybe_wrap_proxy(url: str, proxy_prefix: str) -> str:
    if not proxy_prefix:
        return url
    if is_proxy_exempt(url):
        return url
    if "ezproxy.lib.utexas.edu" in proxy_prefix.lower():
        return to_ut_ezproxy_url(url)
    encoded_url = quote(url, safe="")
    if "{url}" in proxy_prefix:
        return proxy_prefix.replace("{url}", encoded_url)
    return f"{proxy_prefix}{encoded_url}"


def extract_sciencedirect_pii(path: str) -> str | None:
    match = re.search(r"/science/article/(?:abs/)?pii/([^/?#]+)", path)
    if match:
        return match.group(1)
    match = re.search(r"/retrieve/pii/([^/?#]+)", path)
    if match:
        return match.group(1)
    return None


def normalize_sciencedirect_url(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host not in SCIENCEDIRECT_HOSTS and host not in ELSEVIER_LINKING_HOSTS:
        return url

    pii = extract_sciencedirect_pii(parts.path)
    if not pii:
        return url

    # Canonical form expected by UT proxy routing.
    return urlunsplit(
        (
            "https",
            "www.sciencedirect.com",
            f"/science/article/pii/{pii}",
            "via%3Dihub",
            parts.fragment,
        )
    )


def to_ut_ezproxy_url(url: str) -> str:
    normalized = normalize_sciencedirect_url(url)
    parts = urlsplit(normalized)
    host = (parts.hostname or "").lower()
    if not host:
        return url
    if host.endswith(f".{UT_EZPROXY_DOMAIN}") or host == UT_EZPROXY_DOMAIN:
        return normalized

    proxied_host = f"{host.replace('.', '-')}.{UT_EZPROXY_DOMAIN}"
    proxied_netloc = proxied_host
    if parts.port is not None:
        proxied_netloc = f"{proxied_host}:{parts.port}"

    return urlunsplit((parts.scheme or "https", proxied_netloc, parts.path, parts.query, parts.fragment))


def launch_browser(browser: str, urls: list[str], delay: float = 0.0,
                   firefox_profile: str | None = None) -> None:
    if not urls:
        return

    for index, url in enumerate(urls):
        if browser == "firefox":
            command = ["firefox"]
            if firefox_profile:
                command += ["--profile", firefox_profile]
            command += ["--new-window", url]
        elif browser == "chrome":
            command = ["google-chrome", "--new-window", url]
        else:
            raise ValueError(f"Unsupported browser: {browser}")

        try:
            subprocess.run(command, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Browser executable not found for '{browser}'. "
                "For chrome, ensure 'google-chrome' is on PATH."
            ) from exc

        if delay > 0 and index < len(urls) - 1:
            time.sleep(delay)


def write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pmid", "target_type", "resolved_url", "opened_url"]
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote manifest: {path}")


def build_target_records(args: argparse.Namespace, pmids: list[str]) -> list[dict[str, str]]:
    pubmed_by_pmid = {pmid: PUBMED_URL.format(pmid=pmid) for pmid in pmids}
    publisher_by_pmid: dict[str, str | None] = {}
    pmcid_by_pmid = resolve_pmcids(pmids, batch_size=args.elink_batch_size, timeout=args.elink_timeout)
    if args.mode in {"publisher", "both"}:
        publisher_by_pmid = resolve_primary_publisher_urls(
            pmids, batch_size=args.elink_batch_size, timeout=args.elink_timeout
        )

    records: list[dict[str, str]] = []
    unresolved_publisher_pmids: list[str] = []
    excluded_by_elsevier_filter: list[str] = []
    excluded_by_pmc_filter: list[str] = []
    for pmid in pmids:
        targets: list[tuple[str, str]] = []
        seen_target_urls: set[str] = set()
        publisher_url = publisher_by_pmid.get(pmid) if args.mode in {"publisher", "both"} else None
        publisher_is_elsevier = bool(publisher_url and is_elsevier_publisher_url(publisher_url))
        pmcid = pmcid_by_pmid.get(pmid)
        pmc_available = bool(pmcid)

        if args.elsevier_filter == "exclude" and publisher_is_elsevier:
            excluded_by_elsevier_filter.append(pmid)
            continue
        if args.elsevier_filter == "only" and not publisher_is_elsevier:
            excluded_by_elsevier_filter.append(pmid)
            continue
        if args.pmc_filter == "exclude" and pmc_available:
            excluded_by_pmc_filter.append(pmid)
            continue
        if args.pmc_filter == "only" and not pmc_available:
            excluded_by_pmc_filter.append(pmid)
            continue

        if args.mode in {"pubmed", "both"}:
            if pmcid:
                pmc_url = pmcid_to_pmc_url(pmcid)
                targets.append(("pmc", pmc_url))
                seen_target_urls.add(pmc_url)
            else:
                targets.append(("pubmed", pubmed_by_pmid[pmid]))

        if args.mode in {"publisher", "both"}:
            if pmcid:
                pmc_url = pmcid_to_pmc_url(pmcid)
                if pmc_url not in seen_target_urls:
                    targets.append(("pmc", pmc_url))
                    seen_target_urls.add(pmc_url)
            elif publisher_url:
                targets.append(("publisher", publisher_url))
            elif args.fallback_to_pubmed:
                targets.append(("pubmed_fallback", pubmed_by_pmid[pmid]))
                unresolved_publisher_pmids.append(pmid)
            else:
                unresolved_publisher_pmids.append(pmid)

        for target_type, resolved_url in targets:
            with_fragment = attach_pmid_fragment(
                resolved_url, pmid=pmid, enabled=args.attach_pmid_fragment
            )
            proxied_url = maybe_wrap_proxy(with_fragment, proxy_prefix=args.proxy_prefix)
            opened_url = wrap_with_pmid_redirect(
                proxied_url, pmid=pmid, enabled=args.attach_pmid_fragment
            )
            records.append(
                {
                    "pmid": pmid,
                    "target_type": target_type,
                    "resolved_url": resolved_url,
                    "opened_url": opened_url,
                }
            )

    if unresolved_publisher_pmids:
        suffix = (
            ""
            if len(unresolved_publisher_pmids) <= 10
            else f" ... (+{len(unresolved_publisher_pmids) - 10} more)"
        )
        preview = ", ".join(unresolved_publisher_pmids[:10])
        action = "used PubMed fallback" if args.fallback_to_pubmed else "skipped"
        print(
            f"[WARN] Publisher URL unresolved for {len(unresolved_publisher_pmids)} PMID(s); "
            f"{action}: {preview}{suffix}"
        )

    if excluded_by_elsevier_filter:
        suffix = (
            ""
            if len(excluded_by_elsevier_filter) <= 10
            else f" ... (+{len(excluded_by_elsevier_filter) - 10} more)"
        )
        preview = ", ".join(excluded_by_elsevier_filter[:10])
        print(
            f"[INFO] Skipped {len(excluded_by_elsevier_filter)} PMID(s) due to --elsevier-filter="
            f"{args.elsevier_filter}: {preview}{suffix}"
        )

    if excluded_by_pmc_filter:
        suffix = (
            ""
            if len(excluded_by_pmc_filter) <= 10
            else f" ... (+{len(excluded_by_pmc_filter) - 10} more)"
        )
        preview = ", ".join(excluded_by_pmc_filter[:10])
        print(
            f"[INFO] Skipped {len(excluded_by_pmc_filter)} PMID(s) due to --pmc-filter="
            f"{args.pmc_filter}: {preview}{suffix}"
        )

    return records


def print_bookmarklet() -> None:
    print("\nSave-as-PMID bookmarklet:")
    print("Create a browser bookmark and set URL to:")
    print(SAVE_BOOKMARKLET)
    print(
        "\nUsage: PMID URL fragment is enabled by default; click the bookmark in the article tab.\n"
        "It downloads current page HTML as <PMID>.html."
    )


def main() -> None:
    args = parse_args()

    # Print the bookmarklet before doing any work; it is a standalone action.
    if args.print_save_bookmarklet:
        print(SAVE_BOOKMARKLET)
        return

    pmid_file = args.pmid_file.expanduser().resolve()

    if args.max_at_once <= 0:
        raise ValueError("--max-at-once must be a positive integer.")
    if args.elink_batch_size <= 0:
        raise ValueError("--elink-batch-size must be a positive integer.")
    if args.elink_timeout <= 0:
        raise ValueError("--elink-timeout must be positive.")
    if args.elsevier_filter != "all" and args.mode == "pubmed":
        raise ValueError("--elsevier-filter requires --mode publisher or --mode both.")

    if not pmid_file.exists():
        raise FileNotFoundError(f"PMID file not found: {pmid_file}")

    pmids = iter_pmids(pmid_file)
    if not pmids:
        print("No valid PMIDs found.")
        return

    if not args.preserve_order:
        # Open newest/higher PMIDs first by default.
        pmids = sorted(pmids, key=int, reverse=True)
    print(f"Found {len(pmids)} valid PMIDs in {pmid_file}")

    # --- completion tracking -------------------------------------------------
    done_dirs = args.done_dirs if args.done_dirs is not None else default_done_dirs()
    done = scan_done([Path(d).expanduser() for d in done_dirs]) if done_dirs else {}
    if done:
        already = [pm for pm in pmids if pm in done]
        if already:
            print(f"[done] {len(already)}/{len(pmids)} already downloaded "
                  f"(scanned {len(done_dirs)} dir(s))")
            if args.skip_done:
                pmids = [pm for pm in pmids if pm not in done]
                print(f"[done] skipping them; {len(pmids)} remain "
                      f"(use --no-skip-done to open anyway)")
    if args.progress_file:
        write_progress(args.progress_file.expanduser(), iter_pmids(pmid_file), done)
    if not pmids:
        print("Nothing outstanding.")
        return

    records = build_target_records(args, pmids)
    if not records:
        print("No URLs to open after resolution.")
        return

    if args.manifest is not None:
        write_manifest(args.manifest, records)

    records_by_pmid: dict[str, list[dict[str, str]]] = {pmid: [] for pmid in pmids}
    for row in records:
        records_by_pmid[row["pmid"]].append(row)

    eligible_pmids = [pmid for pmid in pmids if records_by_pmid.get(pmid)]
    total = len(eligible_pmids)
    for start in range(0, total, args.max_at_once):
        batch_pmids = eligible_pmids[start : start + args.max_at_once]
        batch_end = start + len(batch_pmids)
        batch_rows = [row for pmid in batch_pmids for row in records_by_pmid.get(pmid, [])]
        batch_urls = [row["opened_url"] for row in batch_rows]

        print(
            f"Processing PMIDs {start + 1}-{batch_end} of {total} "
            f"({len(batch_urls)} URL(s))..."
        )
        print(f"PMIDs: {', '.join(batch_pmids)}")

        if args.dry_run:
            for row in batch_rows:
                print(f'{row["pmid"]}\t{row["target_type"]}\t{row["opened_url"]}')
        else:
            launch_browser(args.browser, batch_urls, delay=args.delay,
                           firefox_profile=args.firefox_profile)

        if batch_end >= total:
            break

        next_count = min(args.max_at_once, total - batch_end)
        try:
            response = input(
                f"Press Enter to process the next {next_count} PMID(s), or 'q' to quit: "
            ).strip()
        except EOFError:
            response = "q"
        if response.lower() in {"q", "quit"}:
            print("Stopping before remaining PMIDs.")
            break

    if args.print_save_bookmarklet:
        print_bookmarklet()


if __name__ == "__main__":
    main()
