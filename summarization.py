import openai
from openai import OpenAI

client = OpenAI()

def summarize_article(question, article_text, config):
    """
    Summarize the given article abstract relative to the clinical question.
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
            temperature=config.get('temperature', 0.5),
            max_tokens=config.get('max_tokens', 500)
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        raise RuntimeError(f"Failed to summarize article: {str(e)}") 