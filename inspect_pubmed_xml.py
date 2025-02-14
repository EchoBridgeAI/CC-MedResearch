import sys
import os
import ssl
import certifi

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import Bio
    print(f"Biopython version: {Bio.__version__}")
except ImportError as e:
    print(f"Failed to import Bio: {e}")

from Bio import Entrez
import json
from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_xml_structure(pmid):
    """
    Fetch and inspect the XML structure of a PubMed article.
    """
    Entrez.email = "echobridge.ai@gmail.com"
    
    try:
        # Fetch the article with SSL context
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
        article_data = Entrez.read(handle)
        
        # Debug print raw data
        print("\n=== Raw Article Data ===")
        print(f"Type: {type(article_data)}")
        print(f"Keys: {article_data.keys() if hasattr(article_data, 'keys') else 'No keys'}")
        
        if not article_data:
            logger.error("No data received from PubMed")
            return None
            
        # Log the top-level keys
        logger.info("Top level keys in response:")
        logger.info(list(article_data.keys()))
        
        if 'PubmedArticle' not in article_data:
            logger.error("PubmedArticle not found in response")
            return None
            
        # Get the first article
        article = article_data['PubmedArticle'][0]
        logger.info("Article keys:")
        logger.info(list(article.keys()))
        
        # Print full structure
        print("\n=== Full Article Structure ===")
        pprint(article)
        
        return article
        
    except Exception as e:
        logger.error(f"Error inspecting article: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return None

def debug_search_results(article_ids):
    """
    Debug function to inspect a batch of search results
    """
    try:
        # Fetch articles in batch
        handle = Entrez.efetch(db="pubmed", id=article_ids, rettype="xml")
        data = Entrez.read(handle)
        
        logger.info(f"Number of articles retrieved: {len(data.get('PubmedArticle', []))}")
        
        # Examine each article
        for i, article in enumerate(data.get('PubmedArticle', [])):
            print(f"\n=== Article {i+1} ===")
            print(f"Keys present: {list(article.keys())}")
            
            if 'MedlineCitation' not in article:
                print(f"WARNING: MedlineCitation missing for article {i+1}")
                print("Article structure:")
                pprint(article)
            else:
                print("MedlineCitation found")
                print(f"MedlineCitation keys: {list(article['MedlineCitation'].keys())}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error in debug_search_results: {str(e)}")
        return None

if __name__ == "__main__":
    # First install certifi if not present
    try:
        import certifi
    except ImportError:
        print("Installing certifi...")
        os.system("pip install certifi")
        import certifi
    
    # Test specific PMID
    pmid = "29867979"
    print("\nInspecting single article...")
    article = inspect_xml_structure(pmid)
    
    if article:
        print("\nArticle successfully retrieved!")
        if 'MedlineCitation' in article:
            print("MedlineCitation found in article")
            print(f"MedlineCitation keys: {list(article['MedlineCitation'].keys())}")
        else:
            print("MedlineCitation not found in article")
    
    # Test batch with same PMID
    print("\nInspecting as part of batch...")
    article_ids = [pmid]
    results = debug_search_results(article_ids) 