import requests
from typing import List

NEWS_API_KEY = "fc98b45a53614c63af46f3d369217507"  # Replace with your actual key or load from env

def get_newsapi_headlines(company_name: str, max_results: int = 10) -> List[str]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name,
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "language": "en",
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get("status") != "ok":
            return [f"NewsAPI error: {data.get('message', 'Unknown error')}"]

        articles = data.get("articles", [])
        headlines = [article["title"] for article in articles if article.get("title")]

        return headlines if headlines else [f"No news found for '{company_name}'."]
    except Exception as e:
        return [f"❌ Error fetching news: {e}"]
