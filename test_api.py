import urllib.request, json, urllib.error
req = urllib.request.Request('https://agentic-rag-jtvr.onrender.com/api/chat', data=json.dumps({'query': 'education', 'search_mode': 'local'}).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    print(urllib.request.urlopen(req).read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(e.read().decode('utf-8'))
