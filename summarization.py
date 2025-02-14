import logging
import openai
from openai import OpenAI

logger = logging.getLogger('worker')
logger.propagate = False  # Prevent logs from propagating to root logger

# Initialize OpenAI client
client = OpenAI()

def summarize_article(question, article_text, config):
    """
    Summarize an article abstract in relation to the clinical question.
    
    Args:
        question: The clinical question being researched
        article_text: The article abstract text
        config: Configuration dict with model, temperature, etc.
    
    Returns:
        str: A focused summary of the article
    """
    try:
        messages = [
            {"role": "system", "content": config['system_role']},
            {"role": "user", "content": config['prompt_template'].format(
                question=question,
                article_text=article_text
            )}
        ]
        
        response = client.chat.completions.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
        summary = response.choices[0].message.content.strip()
        
        logger.info("Successfully summarized article")
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing article: {str(e)}", exc_info=True)
        raise 