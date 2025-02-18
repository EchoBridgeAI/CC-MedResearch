import logging
from groq import Groq
import os
from dotenv import load_dotenv
import re
from logger_config import setup_logging

# Load environment variables
load_dotenv()

# Get logger from root configuration
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def check_relevance(question, article_text, config):
    """
    Check if an article is relevant to the clinical question.
    
    Args:
        question: The clinical question being researched
        article_text: The article abstract text
        config: Configuration dict with model, temperature, etc.
    
    Returns:
        bool: True if article is relevant, False otherwise
    """
    try:
        pmid = extract_pmid_from_text(article_text)
        logger.info(f"Starting relevance check for PMID: {pmid}")
        
        # Log the input data
        logger.info(f"Question length: {len(question)}")
        logger.info(f"Article text length: {len(article_text)}")
        logger.info(f"Config: {config}")
        
        messages = [
            {"role": "system", "content": config['system_role']},
            {"role": "user", "content": config['prompt_template'].format(
                question=question,
                article_text=article_text
            )}
        ]

        logger.info(f"Making Groq API call for PMID: {pmid}")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use Groq's model
            messages=messages,
            max_tokens=config['max_tokens']  # Groq uses max_tokens
        )
        answer = response.choices[0].message.content.strip().lower()

        logger.info(f"PMID {pmid} relevance: {answer}")
        return answer == 'yes'

    except Exception as e:
        pmid = extract_pmid_from_text(article_text) or "Unknown"
        logger.error(f"Error checking relevance for PMID {pmid}:")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Question: {question[:100]}...")
        logger.error(f"Article text: {article_text[:100]}...")
        raise

def extract_pmid_from_text(text):
    """Extract PMID from article text or summary"""
    match = re.search(r"PMID:(\d+)", text)
    return match.group(1) if match else "Unknown"

def extract_pmid_from_text(text):
    """Extract PMID from article text or summary"""
    match = re.search(r"PMID:(\d+)", text)
    return match.group(1) if match else "Unknown" 