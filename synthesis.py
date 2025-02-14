import openai
from openai import OpenAI

client = OpenAI()

def synthesize_summaries(question, article_summaries, config):
    """
    Synthesize multiple article summaries to answer the clinical question.
    """
    summaries_combined = "\n\n".join(article_summaries)
    prompt = config['prompt_template'].format(question=question, article_summaries=summaries_combined)
    messages = [
        {"role": "system", "content": config['system_role']},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=config['model'],
            messages=messages,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 4096)
        )
        synthesis = response.choices[0].message.content.strip()
        return synthesis
    except Exception as e:
        raise RuntimeError(f"Failed to synthesize summaries: {str(e)}") 