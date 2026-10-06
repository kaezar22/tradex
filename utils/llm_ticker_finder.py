import re

import streamlit as st
from openai import OpenAI

def find_tickers_with_deepseek(query: str) -> str:
    """
    Use DeepSeek to return tickers and company names, plus a comma-separated list of tickers.
    """
    prompt = f"""
You are a financial assistant. 
The user will describe the kind of tickers they want.
Return a clean list of ticker symbols and their company names.
Format strictly as:

TICKER - Company Name

Example:
AAPL - Apple Inc
TSLA - Tesla Inc
JNJ - Johnson & Johnson

Do NOT add commentary. Just return the list.
---
User request: {query}
"""

    try:
        client = OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1"
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful stock market assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        raw_text = response.choices[0].message.content.strip()

        # Extract tickers using regex
        tickers = re.findall(r"^[A-Z\.]+", raw_text, flags=re.MULTILINE)
        ticker_list = ", ".join(tickers) if tickers else "❌ No tickers found"

        return raw_text, ticker_list

    except Exception as e:
        return f"❌ Error fetching tickers: {e}", ""

