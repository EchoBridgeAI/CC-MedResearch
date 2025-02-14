import logging
import openai
from openai import OpenAI

logger = logging.getLogger('worker')
logger.propagate = False  # Prevent logs from propagating to root logger

# Initialize OpenAI client
client = OpenAI()

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
        logger.info("Starting relevance check")
        messages = [
            {"role": "system", "content": config['system_role']},
            {"role": "user", "content": config['prompt_template'].format(
                question=question,
                article_text=article_text
            )}
        ]

        logger.info("Making OpenAI API call for relevance check")
        response = client.chat.completions.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
        answer = response.choices[0].message.content.strip().lower()

        logger.info(f"Relevance check result: {answer}")
        return answer == 'yes'

    except Exception as e:
        logger.error(f"Error checking article relevance: {str(e)}\nQuestion: {question[:100]}...\nAbstract: {article_text[:100]}...", exc_info=True)
        raise 