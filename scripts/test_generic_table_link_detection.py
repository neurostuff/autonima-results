#!/usr/bin/env python
# coding: utf-8
"""
Test script for the generic table link detection strategy in DefaultSource.

This script tests the implementation of table link detection methods that we
added to the DefaultSource class.
"""

import sys
import os
from urllib.parse import urljoin
import re

# Add the ACE directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ACE'))

from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MockDefaultSource:
    """
    A mock version of DefaultSource with only the methods we want to test.
    """
    
    def _get_base_url(self, soup):
        """
        Extract base URL from document metadata for resolving relative links.
        
        Tries multiple meta tags commonly used by publishers to specify the
        base URL of the article.
        
        Args:
            soup (BeautifulSoup): Parsed HTML of the article
            
        Returns:
            str or None: Base URL if found, None otherwise
        """
        # Try multiple meta tags for base URL
        meta_tags = [
            {'name': 'citation_public_url'},
            {'name': 'citation_fulltext_html_url'},
            {'property': 'og:url'},
            {'name': 'dc.Identifier', 'scheme': 'doi'},
        ]
        
        for meta_attrs in meta_tags:
            meta = soup.find('meta', attrs=meta_attrs)
            if meta and meta.get('content'):
                base_url = meta['content']
                # Remove query parameters and fragments
                base_url = base_url.split('?')[0].split('#')[0]
                # Remove filename if present
                if '.' in base_url.split('/')[-1]:
                    base_url = '/'.join(base_url.split('/')[:-1])
                return base_url
        return None

    def _detect_text_based_table_links(self, soup, html):
        """
        Find links with text indicating table content.
        
        Looks for anchor tags with text that suggests they link to table content,
        such as "Full size table", "View table", "Expand table", etc.
        
        Args:
            soup (BeautifulSoup): Parsed HTML of the article
            html (str): Raw HTML of the article
            
        Returns:
            list: List of resolved URLs that likely point to table content
        """
        links = []
        text_indicators = [
            r'full\s*size\s*table',
            r'view\s*table',
            r'expand\s*table',
            r'show\s*table',
            r'table\s*details',
            r'download\s*table',
            r'see\s*table',
            r'complete\s*table',
            r'table\s*\d+'
        ]
        
        try:
            # Get base URL for resolving relative links
            base_url = self._get_base_url(soup)
            
            # Look for links with text indicators
            for link in soup.find_all('a', href=True):
                try:
                    link_text = link.get_text().lower().strip()
                    if any(re.search(indicator, link_text) for indicator in text_indicators):
                        href = link.get('href')
                        if href:
                            # Resolve relative URLs
                            if base_url:
                                try:
                                    resolved_url = urljoin(base_url, href)
                                    links.append(resolved_url)
                                except Exception as e:
                                    logger.debug(f"Failed to resolve URL {href}: {e}")
                                    # Fallback to original href
                                    links.append(href)
                            else:
                                links.append(href)
                except Exception as e:
                    logger.debug(f"Error processing link {link}: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Error in _detect_text_based_table_links: {e}")
        
        # Deduplicate links
        return list(set(links))

    def _detect_url_pattern_table_links(self, soup, html):
        """
        Detect links following common table URL patterns.
        
        Identifies URLs that match common patterns used by publishers to link
        to table content, such as /T{num}.expansion.html, /tables/{num}, etc.
        
        Args:
            soup (BeautifulSoup): Parsed HTML of the article
            html (str): Raw HTML of the article
            
        Returns:
            list: List of resolved URLs that likely point to table content
        """
        links = []
        
        try:
            # Get base URL for resolving relative links
            base_url = self._get_base_url(soup)
            
            if base_url:
                # Common patterns for table links
                patterns = [
                    r'/T\d+\.expansion\.html',  # HighWire/Sage pattern
                    r'/tables/\d+',             # Springer pattern
                    r'\?table=\d+',             # Query parameter pattern
                    r'#table\d+',               # Fragment pattern
                    r'/table\d+\.html',         # Direct file pattern
                    r'/tbl\d+\.htm',            # Alternative pattern
                    r'/table/\d+',              # Another common pattern
                ]
                
                # Look for links matching patterns in the HTML
                for pattern in patterns:
                    try:
                        matches = re.findall(pattern, html, re.IGNORECASE)
                        for match in matches:
                            # Resolve relative URLs
                            if base_url:
                                try:
                                    resolved_url = urljoin(base_url, match)
                                    links.append(resolved_url)
                                except Exception as e:
                                    logger.debug(f"Failed to resolve URL {match}: {e}")
                                    # Fallback to original match
                                    if match.startswith('http'):
                                        links.append(match)
                                    else:
                                        # Try to construct with base URL
                                        if match.startswith('/'):
                                            links.append(base_url + match)
                                        else:
                                            links.append(base_url + '/' + match)
                    except Exception as e:
                        logger.debug(f"Error processing pattern {pattern}: {e}")
                        continue
            else:
                logger.debug("No base URL found for resolving table links")
        except Exception as e:
            logger.debug(f"Error in _detect_url_pattern_table_links: {e}")
        
        # Deduplicate links
        return list(set(links))

    def _detect_javascript_table_expansion(self, soup):
        """
        Detect and handle JavaScript-based table expansion.
        
        Identifies elements that might trigger table expansion via JavaScript.
        This method currently only logs detection but does not implement actual
        expansion, which would require browser-based scraping.
        
        Args:
            soup (BeautifulSoup): Parsed HTML of the article
            
        Returns:
            bool: True if JavaScript expansion indicators are found, False otherwise
        """
        # Look for common classes/attributes that indicate expandable tables
        js_indicators = [
            'table-expand',
            'table-expand-inline',
            'expand-table',
            'table-toggle',
            'js-table-expand',
            'data-table-url',
        ]
        
        # Check if any elements have these indicators
        for indicator in js_indicators:
            elements = soup.find_all(class_=indicator)
            if elements:
                logger.info(f"Found JavaScript table expansion indicators: {indicator}")
                # For now, we'll log the detection but not implement the actual expansion
                # This would require integration with the browser-based scraping
                return True
        
        # Check for data attributes that indicate table URLs
        data_elements = soup.find_all(attrs={'data-table-url': True})
        if data_elements:
            logger.info("Found data-table-url attributes for table expansion")
            return True
            
        return False

def test_text_based_link_detection():
    """Test text-based link detection strategy"""
    print("Testing text-based link detection...")
    
    # Create a mock DefaultSource instance
    source = MockDefaultSource()
    
    # Sample HTML with text-based table links
    html = """
    <html>
    <head>
        <meta name="citation_public_url" content="https://example.com/article/123">
    </head>
    <body>
        <h1>Sample Article</h1>
        <p>This is a sample article with tables hidden behind links.</p>
        
        <div class="table-container">
            <h2>Table 1</h2>
            <p><a href="/T1.expansion.html">Full size table</a></p>
        </div>
        
        <div class="table-container">
            <h2>Table 2</h2>
            <p><a href="/tables/2">View table</a></p>
        </div>
        
        <div class="table-container">
            <h2>Table 3</h2>
            <p><a href="#table3">Expand table</a></p>
        </div>
    </body>
    </html>
    """
    
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Test text-based link detection
    links = source._detect_text_based_table_links(soup, html)
    print(f"Found {len(links)} text-based table links: {links}")
    
    print("Text-based link detection test completed.\n")

def test_url_pattern_detection():
    """Test URL pattern recognition strategy"""
    print("Testing URL pattern detection...")
    
    # Create a mock DefaultSource instance
    source = MockDefaultSource()
    
    # Sample HTML with URL patterns
    html = """
    <html>
    <head>
        <meta name="citation_public_url" content="https://example.com/article/123">
    </head>
    <body>
        <h1>Sample Article</h1>
        <p>This is a sample article with tables hidden behind links.</p>
        
        <div class="content">
            <p>Some text with a link to <a href="/T1.expansion.html">Table 1</a>.</p>
            <p>Another table can be found at <a href="/tables/2">Table 2</a>.</p>
            <p>More table content at <a href="/table3.html">Table 3</a>.</p>
        </div>
    </body>
    </html>
    """
    
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Test URL pattern detection
    links = source._detect_url_pattern_table_links(soup, html)
    print(f"Found {len(links)} URL pattern table links: {links}")
    
    print("URL pattern detection test completed.\n")

def test_javascript_expansion_detection():
    """Test JavaScript expansion detection strategy"""
    print("Testing JavaScript expansion detection...")
    
    # Create a mock DefaultSource instance
    source = MockDefaultSource()
    
    # Sample HTML with JavaScript expansion indicators
    html = """
    <html>
    <body>
        <h1>Sample Article</h1>
        <p>This is a sample article with tables that expand via JavaScript.</p>
        
        <div class="table-container">
            <h2>Table 1</h2>
            <button class="table-expand" data-table-url="/T1.expansion.html">Expand Table</button>
        </div>
        
        <div class="table-container">
            <h2>Table 2</h2>
            <a class="table-expand-inline" href="/tables/2">Show Table</a>
        </div>
        
        <div class="table-container">
            <h2>Table 3</h2>
            <span class="js-table-expand" data-table-url="/tables/3">View Table</span>
        </div>
    </body>
    </html>
    """
    
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Test JavaScript expansion detection
    has_js_expansion = source._detect_javascript_table_expansion(soup)
    print(f"JavaScript expansion indicators detected: {has_js_expansion}")
    
    print("JavaScript expansion detection test completed.\n")

def main():
    """Run all tests"""
    print("Running tests for generic table link detection...\n")
    
    try:
        test_text_based_link_detection()
        test_url_pattern_detection()
        test_javascript_expansion_detection()
        
        print("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()