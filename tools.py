import os
import requests


def web_search_tavily(query, num_results=3):
    # Retrieve your API Key from system environment variables
    tavily_api_key = os.getenv('TAVILY_API_KEY')

    if not tavily_api_key:
        print("[Warning] No TAVILY_API_KEY found in environment variables. Using offline stub.")
        return [f"[stub] No TAVILY_API_KEY set. Would search for: {query}"]

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {"api_key": tavily_api_key, "query": query, "num_results": num_results}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        data = resp.json()
        results = []
        for r in data.get('results', []):
            title = r.get('title') or r.get('heading') or ''
            snippet = r.get('content') or r.get('snippet') or ''
            link = r.get('link') or r.get('url') or ''
            results.append({'title': title, 'snippet': snippet, 'link': link})
        return results
    except Exception as e:
        print(f"Error connecting to Tavily: {e}")
        return []