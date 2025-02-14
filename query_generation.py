import openai
import re
from openai import OpenAI

# Instantiate a client using the new v1 interface
client = OpenAI()

def generate_query(question, config):
    """
    Generate a PubMed query using an LLM.
    The response is expected to be enclosed in triple backticks.
    """
    prompt = config['prompt_template'].format(question=question)
    messages = [
        {"role": "system", "content": config['system_role']},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=config['model'],
            messages=messages,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 1024)
        )
        content = response.choices[0].message.content.strip()
        # Extract text within triple backticks if present
        match = re.search(r"```(.*?)```", content, re.DOTALL)
        if match:
            query = match.group(1).strip()
        else:
            query = content
        return query
    except Exception as e:
        raise RuntimeError(f"Failed to generate query: {str(e)}") 