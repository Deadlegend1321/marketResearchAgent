import os
from dotenv import load_dotenv
from database import initialize_rag_index
from agent import JSONMemory, MarketResearchAgent

# Load environmental variables from .env
load_dotenv()


def main():
    company_csv = os.path.join("data", "company_profiles_rag_detailed.csv")
    trend_csv = os.path.join("data", "market_trends_rag_detailed.csv")

    if not os.path.exists(company_csv) or not os.path.exists(trend_csv):
        print(
            "Error: Place company_profiles_rag_detailed.csv and market_trends_rag_detailed.csv in the project folder.")
        return

    # Init DB
    initialize_rag_index(company_csv, trend_csv)

    # Init Agent
    mem = JSONMemory()
    agent = MarketResearchAgent(mem)

    query = 'Healthcare AI market outlook 2025'

    print('\n--- Running Planner Stage ---')
    print('Plan:', agent.planner(query))

    print('\n--- Running Gather Stage ---')
    evidence = agent.gather(query, top_k_rag=3, top_k_web=2)
    print(f'Retrieved {len(evidence)} dynamic references.')

    print('\n--- Querying LLM (Initial Draft) ---')
    draft = agent.draft(query, evidence)
    print(f'Draft Output:\n{draft}')

    print('\n--- Querying LLM (Self-Reflection & Revision) ---')
    final = agent.critique_and_revise(draft)
    print(f'Final Revised Brief:\n{final}')


if __name__ == "__main__":
    main()