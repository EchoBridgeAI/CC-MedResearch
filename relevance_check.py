import openai
from openai import OpenAI

client = OpenAI()

def check_relevance(question, article_text, config):
    """
    Check if the article's abstract is relevant to the clinical question.
    Returns True if relevant, False otherwise.
    """
    prompt = config['prompt_template'].format(question=question, article_text=article_text)
    messages = [
        {"role": "system", "content": config['system_role']},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=config['model'],
            messages=messages,
            temperature=config.get('temperature', 0.3),
            max_tokens=config.get('max_tokens', 100)
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == "yes"
    except Exception as e:
        raise RuntimeError(f"Failed to perform relevance check: {str(e)}") 