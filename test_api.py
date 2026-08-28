import urllib.request, json, urllib.error
req = urllib.request.Request(
    'https://agentic-rag-jtvr.onrender.com/api/chat',
    data=json.dumps({'query': 'what is the latest news today', 'search_mode': 'auto'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
try:
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read().decode('utf-8'))
    print("ANSWER:", data.get('answer', '')[:200])
    print("CITATIONS:", data.get('citations', []))
except urllib.error.HTTPError as e:
    print("ERROR:", e.read().decode('utf-8'))
except Exception as e:
    print("EXCEPTION:", str(e))
