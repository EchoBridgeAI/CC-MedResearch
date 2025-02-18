import openai
from openai import OpenAI

client = OpenAI()

def synthesize_summaries(question, article_summaries, config):
    """
    Synthesize multiple article summaries to answer the clinical question.
    """
    try:
        messages = [
            {"role": "system", "content": config['system_role']},
            {"role": "user", "content": config['prompt_template'].format(
                question=question,
                article_summaries="\n\n".join(article_summaries)
            )}
        ]
        
        response = client.chat.completions.create(
            model=config['model'],
            messages=messages,
            max_completion_tokens=config['max_tokens'],
            reasoning_effort=config.get('reasoning_effort', 'medium')
        )
        
        synthesis = response.choices[0].message.content.strip()
        return synthesis
    except Exception as e:
        raise RuntimeError(f"Failed to synthesize summaries: {str(e)}") 