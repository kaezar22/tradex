from openai import OpenAI
from typing import List
from utils.scraper import get_news_from_company_name  # updated import

def analyze_sentiment_with_deepseek(company_name: str) -> str:
    # Step 1: Scrape headlines using company name
    headlines = get_news_from_company_name(company_name)

    if not headlines or "no ticker" in headlines[0].lower() or "no news" in headlines[0].lower():
        return f"❗ {headlines[0]}" if headlines else "❗ No headlines found."

    # Step 2: Compose the news text for the prompt
    news_text = "\n".join(f"- {headline}" for headline in headlines)

    # Step 3: Define the prompt
    prompt = f"""
You are a financial analyst. Based on the following recent news headlines about {company_name}, do the following:

1. Summarize the current themes or events related to the company.
2. Determine the overall sentiment as one of the following:
   - Bullish: mostly positive outlook or upward movement.
   - Bearish: mostly negative outlook or downward movement.
   - Volatile: significant mix of up/down with uncertainty or instability.
   - Neutral: no strong tendency detected.
3. Keep it concise and insightful.

News headlines:
{news_text}

Respond with a clear and structured analysis.
"""

    messages = [
        {"role": "system", "content": "You are a helpful financial market assistant."},
        {"role": "user", "content": prompt}
    ]

    # Step 4: Call DeepSeek
    try:
        client = OpenAI(
            api_key="sk-900f90f072b349d8ba65e95e1eabb2ff",  # replace with your actual key
            base_url="https://api.deepseek.com/v1"
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.4,
            max_tokens=500
        )

        return "📊 Sentiment Result:\n\n" + response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error from DeepSeek API: {e}"
