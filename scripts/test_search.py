import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.hybrid_search import hybrid_search

if __name__ == '__main__':
    try:
        res = hybrid_search("work time")
        print(f"Got {len(res)} chunks")
        for c in res:
            print(c.text[:100])
    except Exception as e:
        print(f"ERROR: {e}")
