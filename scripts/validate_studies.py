

#!/usr/bin/env python3
# Script to validate HTML scrapes and detect bot detection interference.

"""
Script to validate HTML scrapes and detect bot detection interference.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys


def _validate_scrape(html: str) -> Tuple[bool, str]:
    """Checks to see if scraping was successful.
    
    Args:
        html: The HTML content to validate
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    
    # Bot detection and CAPTCHA patterns
    bot_detection_patterns = [
        'Checking if you are a human',
        'Please turn JavaScript on and reload the page',
        'Checking if the site connection is secure',
        'Enable JavaScript and cookies to continue',
        'used Cloudflare to restrict access',
        'Ray ID:',  # Cloudflare error pages
        'cf-browser-verification',  # Cloudflare verification
        'Just a moment...',  # Cloudflare challenge page
        'attention required',  # Cloudflare attention
        'Why have I been blocked?',  # Cloudflare block
        'Access denied | ',  # Cloudflare access denied
        'Security check',  # Generic security check
        'I\'m not a robot',  # reCAPTCHA
        'Please verify you are a human',
        'Verify you are human',
        'captcha-delivery',  # CAPTCHA systems
        'g-recaptcha',  # Google reCAPTCHA
        'hcaptcha',  # hCaptcha
        'Are you a robot?',
        'Bot detection',
        'Suspected bot',
        'Automated access',
    ]
    
    # HTTP error patterns
    http_error_patterns = [
        '403 Forbidden',
        '404 Not Found',
        '429 Too Many Requests',
        '500 Internal Server Error',
        '502 Bad Gateway',
        '503 Service Unavailable',
        '504 Gateway Timeout',
        'HTTP Error',
    ]
    
    # Publisher-specific error patterns
    publisher_error_patterns = [
        '<title>Page not available - PMC</title>',
        'Page not found — ScienceDirect',
        'Article not found',
        'Content not available',
        'There was a problem providing the content you requested',
        'Your request cannot be processed at this time. '
        'Please try again later',
        'This content is not available',
        'Access to this page has been denied',
        'The page you requested could not be found',
        'Sorry, we couldn\'t find that page',
    ]
    
    # Redirect and connection patterns
    redirect_patterns = [
        '<title>Redirecting</title>',
        'This site can\'t be reached',
        'ERR_CONNECTION_REFUSED',
        'ERR_CONNECTION_TIMED_OUT',
        'Unable to connect',
        'Connection refused',
        'Too many redirects',
    ]
    
    # Paywall and subscription patterns
    paywall_patterns = [
        'Subscribe to continue reading',
        'Sign in to access',
        'Subscription required',
        'Purchase this article',
        'Login required',
        'Register to read',
        'Create a free account',
    ]
    
    # Empty or minimal content patterns
    if not html or len(html.strip()) < 100:
        return False, "HTML content is empty or too short"
    
    empty_html = '<html></html>'
    minimal_html = '<html><body></body></html>'
    if html.strip() == empty_html or html.strip() == minimal_html:
        return False, "HTML contains no content"
    
    # Check all patterns
    all_patterns = [
        ("bot_detection", bot_detection_patterns),
        ("http_error", http_error_patterns),
        ("publisher_error", publisher_error_patterns),
        ("redirect", redirect_patterns),
        ("paywall", paywall_patterns),
    ]
    
    for category, patterns in all_patterns:
        for pattern in patterns:
            if pattern.lower() in html.lower():
                return False, f"{category}: {pattern}"
    
    return True, ""


def validate_html_file(html_path: Path) -> Dict:
    """Validate a single HTML file.
    
    Args:
        html_path: Path to the HTML file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'file': str(html_path),
        'valid': False,
        'error': None,
        'file_size': 0,
    }
    
    try:
        if not html_path.exists():
            result['error'] = 'File does not exist'
            return result
        
        result['file_size'] = html_path.stat().st_size
        
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read()
        
        is_valid, error_msg = _validate_scrape(html)
        result['valid'] = is_valid
        if not is_valid:
            result['error'] = error_msg
            
    except Exception as e:
        result['error'] = f'Exception: {str(e)}'
    
    return result


def validate_html_files(
    html_paths: List[Path], output_json: Path = None
) -> Dict:
    """Validate multiple HTML files.
    
    Args:
        html_paths: List of paths to HTML files
        output_json: Optional path to save results as JSON
        
    Returns:
        Dictionary with summary and detailed results
    """
    results = []
    
    for html_path in html_paths:
        result = validate_html_file(html_path)
        results.append(result)
        
        # Print status
        if result['valid']:
            status = "✓ VALID"
        else:
            status = f"✗ INVALID ({result['error']})"
        print(f"{status}: {html_path}")
    
    # Create summary
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    
    summary = {
        'total': len(results),
        'valid': valid_count,
        'invalid': invalid_count,
        'results': results,
    }
    
    # Group errors by type
    error_counts = {}
    for r in results:
        if not r['valid'] and r['error']:
            if ':' in r['error']:
                error_type = r['error'].split(':')[0]
            else:
                error_type = r['error']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    summary['error_types'] = error_counts
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_json}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Validate HTML scrapes for bot detection and errors'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    # Collect all HTML files
    html_files = ['/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/10607399.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neuroreport/10943684.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/11352615.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/11496124.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/11850635.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/12202103.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Psychological science/15327630.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Trends in cognitive sciences/15668098.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/15701234.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/17055981.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Psychological science/17100784.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/17404215.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Journal of cognitive neuroscience/17536964.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neuropsychobiology/17986835.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/NeuroImage/18234518.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/18313858.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/18370602.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Proceedings of the National Academy of Sciences of the United States of America/18375769.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/18439411.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/18585742.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/18586108.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/18985118.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/18985119.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/19015088.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Journal of clinical and experimental neuropsychology/19048446.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/19136216.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/19199417.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/19199419.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/19304843.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/19596021.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/19630890.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/19739909.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/20071521.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Annual review of neuroscience/20350167.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/20350171.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/20350187.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Social neuroscience/20401807.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Proceedings of the National Academy of Sciences of the United States of America/20439736.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/20460301.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/20534459.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Journal of comparative psychology (Washington, D.C. : 1983)/20695655.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/20705602.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The Journal of neuroscience : the official journal of the Society for Neuroscience/20943931.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neuroscience and biobehavioral reviews/21036192.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Psychological science/21164174.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21278194.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21606658.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/21723130.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Human brain mapping/21761508.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Journal of visualized experiments : JoVE/21775952.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21812564.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21849560.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21862446.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21864459.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21908447.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/21940454.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/European child & adolescent psychiatry/22038344.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22125232.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/22257745.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Human brain mapping/22290781.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22308468.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22349798.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22360624.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22403154.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Journal of the Royal Society of Medicine/22434810.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22490923.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22507230.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Biological psychiatry/22507699.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22563005.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Social cognitive and affective neuroscience/22563008.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22623534.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22661409.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22848759.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The European journal of neuroscience/22909094.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/22956840.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Infant behavior & development/22982277.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23142071.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23249349.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23327932.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Experimental brain research/23435496.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23460073.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23482624.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23508477.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/23528247.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23574585.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23620602.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23696200.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23720575.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23740868.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23770622.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23804962.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23876243.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23887806.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Social cognitive and affective neuroscience/23887810.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/23911672.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Brain and nerve = Shinkei kenkyu no shinpo/23917500.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Social cognitive and affective neuroscience/23988759.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Hearing research/24036130.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neuropsychobiology/24051621.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24068815.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24084068.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24089495.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24097375.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24106333.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24144548.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24265613.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Seishin shinkeigaku zasshi = Psychiatria et neurologia Japonica/24341069.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The Journal of neuroscience : the official journal of the Society for Neuroscience/24381271.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The European journal of neuroscience/24447026.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The Journal of neuroscience : the official journal of the Society for Neuroscience/24478377.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24493846.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/24607363.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24633532.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Proceedings of the National Academy of Sciences of the United States of America/24639542.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24652858.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24666131.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24700584.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/24795436.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/25042446.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/25088911.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/25244113.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/25298009.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Social cognitive and affective neuroscience/25338630.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Proceedings of the National Academy of Sciences of the United States of America/25489093.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/25499683.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Proceedings of the National Academy of Sciences of the United States of America/25512496.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference/25571093.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/25770039.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Cognitive neuroscience/25893437.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/25944965.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/25987597.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26117505.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/26162239.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26168793.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26200892.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26206505.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/26318628.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26337369.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26342221.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26348613.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26385612.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26598684.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26604273.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26608245.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26621704.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26656563.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference/26736634.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference/26737187.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26759479.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/26825440.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26920683.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Current topics in behavioral neurosciences/26946502.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/26969865.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27007121.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27109357.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The Journal of neuroscience : the official journal of the Society for Neuroscience/27122031.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/NeuroImage/27129758.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27129794.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Psychological science/27150109.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27165762.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27167401.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The world journal of biological psychiatry : the official journal of the World Federation of Societies of Biological Psychiatry/27170266.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Proceedings of the National Academy of Sciences of the United States of America/27185915.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27217118.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27247125.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27272314.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27319001.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neurologia/27340019.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27358450.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27405334.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27458363.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27496338.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27506384.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27510495.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27528669.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27576746.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27579051.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Communication monographs/27642220.html', None, None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27716474.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27798249.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27803286.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/27815729.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/27856343.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Current biology : CB/27866893.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28008075.html', None, None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28242678.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28289200.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28338962.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Development and psychopathology/28393755.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28465434.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Science (New York, N.Y.)/28522533.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28540647.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28557690.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Human brain mapping/28608647.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28643894.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28748572.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28766324.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/28807871.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/28966083.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/29073111.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/29077925.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/29111359.html', None, None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/29408539.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/29455860.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Advances in child development and behavior/29455864.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Annual review of neuroscience/29561702.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/29760183.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/29771359.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Brain connectivity/29896995.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Cerebral cortex (New York, N.Y. : 1991)/29931116.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/29959970.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neuron/30017395.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Trends in cognitive sciences/30041864.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30154703.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/30195053.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Genes, brain, and behavior/30221467.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Brain imaging and behavior/30374665.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30389840.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neuropsychology/30411904.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/30414457.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The Journal of neuroscience : the official journal of the Society for Neuroscience/30455187.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Psychiatry research. Neuroimaging/30594068.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30610911.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30658938.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30698656.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30794869.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/30807820.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/30825583.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The Journal of neuroscience : the official journal of the Society for Neuroscience/30842248.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30852994.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30852995.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/30930310.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/31028922.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/The Journal of neuroscience : the official journal of the Society for Neuroscience/31036762.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Biological psychology/31051206.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/31063817.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/31157395.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Neuron/31170400.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/31268615.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/31322784.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Hippocampus/31589003.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/31729396.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/31740271.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Manual/31783116.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/pond/31798816.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Nature human behaviour/31844272.html', None, '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/Pediatric research/38867029.html', '/home/zorro/repos/autonima-results/social-processing/ace_scrape/articles/html/IEEE transactions on pattern analysis and machine intelligence/40773391.html']

    html_files = [Path(f) for f in html_files if f is not None]
    print(f"Validating {len(html_files)} HTML file(s)...\n")
    
    # Validate files
    output_path = Path(args.output) if args.output else None
    summary = validate_html_files(html_files, output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {summary['total']}")
    print(f"Valid: {summary['valid']}")
    print(f"Invalid: {summary['invalid']}")
    
    if summary['error_types']:
        print("\nError types:")
        sorted_errors = sorted(
            summary['error_types'].items(),
            key=lambda x: -x[1]
        )
        for error_type, count in sorted_errors:
            print(f"  {error_type}: {count}")
    
    # Exit with error code if any invalid files
    return 0 if summary['invalid'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())