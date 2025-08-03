import requests
from bs4 import BeautifulSoup
from typing import List

def get_news_from_company_name(company_name: str, max_headlines: int = 10) -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # Step 1: Look up the ticker for the company
    search_query = company_name.replace(" ", "+")
    lookup_url = f"https://finance.yahoo.com/lookup?s={search_query}"

    try:
        response = requests.get(lookup_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        table = soup.select_one('table tbody')
        first_row = table.select_one('tr') if table else None
        ticker_cell = first_row.select_one('td') if first_row else None

        if not ticker_cell:
            return [f"No ticker symbol found for company: {company_name}"]

        ticker = ticker_cell.text.strip()

        # Step 2: Fetch company-specific news headlines
        news_url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
        news_response = requests.get(news_url, headers=headers)
        news_soup = BeautifulSoup(news_response.content, "html.parser")

        # Look for article headlines in the news tab
        headline_tags = news_soup.select('h3 a[href^="/news/"]')
        headlines = [tag.get_text(strip=True) for tag in headline_tags if tag.get_text(strip=True)]

        return headlines[:max_headlines] if headlines else [f"No news found for '{ticker}'"]
    except Exception as e:
        return [f"Error scraping news: {e}"]


# Optional alias for compatibility
def get_yahoo_finance_headlines(company_name: str, max_results: int = 10) -> List[str]:
    return get_news_from_company_name(company_name, max_results)
