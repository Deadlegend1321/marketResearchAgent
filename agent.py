import json
import os
import time
from typing import List, Dict
from openai import OpenAI
from database import rag_search
from tools import web_search_tavily

# Automatically loads .env file keys into os.environ
from dotenv import load_dotenv

load_dotenv()


class JSONMemory:
    def __init__(self, path='agent_memory_rag_tavily.json'):
        self.path = path
        if not os.path.exists(self.path):
            self.write({'notes': [], 'drafts': [], 'citations': []})

    def load(self) -> dict:
        with open(self.path, 'r') as f:
            return json.load(f)

    def write(self, data: dict):
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_note(self, note: str):
        d = self.load()
        d['notes'].append({'ts': time.time(), 'note': note})
        self.write(d)

    def add_citation(self, src: str):
        d = self.load()
        if src not in d['citations']:
            d['citations'].append(src)
            self.write(d)

    def add_draft(self, draft: str):
        d = self.load()
        d['drafts'].append({'ts': time.time(), 'draft': draft})
        self.write(d)


class RealLLM:
    def __init__(self):
        # OpenAI reads OPENAI_API_KEY from environment variables loaded via dotenv
        self.client = OpenAI()

    def draft(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a professional market research writer. Synthesize the provided context into an executive brief with clear markdown bullet points and source citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    def critique(self, draft: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a rigorous chief editor. Review this market brief draft. Suggest factual, structural, or clarity improvements in 2-3 sentences."},
                {"role": "user", "content": draft}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    def revise(self, draft: str, critique: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a detailed market researcher. Revise the original draft incorporating the feedback from the critique. Retain citations and ensure high accuracy."},
                {"role": "user", "content": f"Original Draft:\n{draft}\n\nCritique:\n{critique}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content


real_llm = RealLLM()


class MarketResearchAgent:
    def __init__(self, memory: JSONMemory):
        self.memory = memory

    def planner(self, query: str) -> List[str]:
        plan = ["Run RAG (internal docs)", "Run Tavily web search (freshness)", "Draft brief", "Critique & Revise"]
        self.memory.add_note('planner created')
        return plan

    def gather(self, query: str, top_k_rag=3, top_k_web=2) -> List[Dict]:
        evidence = []

        rag_hits = rag_search(query, top_k=top_k_rag)
        for i, h in enumerate(rag_hits):
            evidence.append({'type': 'rag', 'id': i, 'text': h})
            self.memory.add_citation(f'rag:{i}')

        web_hits = web_search_tavily(query, num_results=top_k_web)
        for i, h in enumerate(web_hits):
            text = h.get('snippet', h) if isinstance(h, dict) else str(h)
            link = h.get('link') if isinstance(h, dict) else None
            evidence.append({'type': 'web', 'id': i, 'text': text, 'link': link})
            self.memory.add_citation(f'web:{i}')

        self.memory.add_note('gathered evidence')
        return evidence

    def draft(self, query: str, evidence: List[Dict]) -> str:
        context = '\n'.join([f"[{e['type']}:{e['id']}] {e['text']}" for e in evidence])
        prompt = f"Draft a concise market brief for: {query}\nUse the evidence below:\n{context}"
        draft = real_llm.draft(prompt)
        self.memory.add_draft(draft)
        return draft

    def critique_and_revise(self, draft: str) -> str:
        critique = real_llm.critique(draft)
        revised = real_llm.revise(draft, critique)
        self.memory.add_draft(revised)
        return revised