import logging
from groq import Groq
import os
from dotenv import load_dotenv
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from logger_config import setup_logging

# Load environment variables
load_dotenv()

# Get logger from root configuration
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def fetch_full_text(pmid):
    """
    Attempt to fetch full text of article using various methods:
    1. PubMed Central (if available)
    2. Europe PMC API
    3. Unpaywall API (future)
    """
    try:
        # First check PMC
        pmc_text = try_pubmed_central(pmid)
        if pmc_text:
            logger.info(f"Retrieved full text from PMC for PMID {pmid}")
            return pmc_text
            
        # Try Europe PMC
        europe_pmc_text = try_europe_pmc(pmid)
        if europe_pmc_text:
            logger.info(f"Retrieved full text from Europe PMC for PMID {pmid}")
            return europe_pmc_text
            
        logger.warning(f"Could not find full text for PMID {pmid}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching full text for PMID {pmid}: {str(e)}")
        return None

def try_pubmed_central(pmid):
    """Try to get full text from PubMed Central"""
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
    try:
        response = requests.get(f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}")
        if response.ok:
            data = response.json()
            if 'records' in data and data['records']:
                pmcid = data['records'][0].get('pmcid')
                if pmcid:
                    article_url = urljoin(base_url, pmcid)
                    response = requests.get(article_url)
                    if response.ok:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        return extract_article_text(soup)
    except Exception as e:
        logger.error(f"PMC fetch error for {pmid}: {str(e)}")
    return None

def try_europe_pmc(pmid):
    """Try to get full text from Europe PMC"""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmid}/fullText"
    try:
        response = requests.get(url)
        if response.ok:
            soup = BeautifulSoup(response.text, 'xml')
            return extract_article_text(soup)
    except Exception as e:
        logger.error(f"Europe PMC fetch error for {pmid}: {str(e)}")
    return None

def extract_article_text(soup):
    """Extract article text from BeautifulSoup object"""
    # Remove irrelevant sections
    for tag in soup.find_all(['table', 'figure', 'ref']):
        tag.decompose()
        
    # Get main text sections
    sections = []
    
    # Try different possible section classes/IDs
    for section in soup.find_all(['div', 'section'], 
                               class_=['section', 'body', 'article-body']):
        text = section.get_text(separator=' ', strip=True)
        if text:
            sections.append(text)
            
    return '\n\n'.join(sections)

def summarize_article(question, article_text, config):
    """
    Summarize an article in relation to the clinical question.
    
    Args:
        question: The clinical question being researched
        article_text: The article text (full text or abstract)
        config: Configuration dict with model, temperature, etc.
    
    Returns:
        str: A focused summary of the article
    """
    try:
        pmid = extract_pmid_from_text(article_text)
        logger.info(f"Starting summarization for PMID: {pmid}")
        messages = [
            {"role": "system", "content": config['system_role']},
            {"role": "user", "content": config['prompt_template'].format(
                question=question,
                article_text=article_text
            )}
        ]
        
        logger.info(f"Making Groq API call for PMID: {pmid}")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=config['temperature'],
            max_completion_tokens=config['max_tokens']
        )
        summary = response.choices[0].message.content.strip()
        
        logger.info(f"Successfully summarized PMID: {pmid}")
        return summary
        
    except Exception as e:
        pmid = extract_pmid_from_text(article_text) or "Unknown"
        logger.error(f"Error summarizing PMID {pmid}: {str(e)}", exc_info=True)
        raise

def extract_pmid_from_text(text):
    """Extract PMID from article text or summary"""
    match = re.search(r"PMID:(\d+)", text)
    return match.group(1) if match else "Unknown" 