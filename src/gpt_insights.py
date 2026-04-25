from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_gpt_insights(context_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a retail business analyst."},
                {"role": "user", "content": f"Give insights and recommendations:\n{context_text}"}
            ]
        )
        return response.choices[0].message.content
    except:
        return "GPT insights not available."