import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from qdrant_client import QdrantClient
from embeddings import load_model, search

SCORE_THRESHOLD = 0.25
QDRANT_URL = 'http://localhost:6333'

load_dotenv()
bot_token = os.getenv('SLACK_BOT_TOKEN')
app_token = os.getenv('SLACK_APP_TOKEN')

app = App(token=bot_token)
model = load_model()
client = QdrantClient(url=QDRANT_URL)

# message handler goes here
@app.event("message")
def handle_message(event, say):
    user_query = event["text"]
    result = search(client, model, user_query, 3)

    scored_results = [r for r in result if r["score"] >= SCORE_THRESHOLD]

    if(not scored_results):
       say("No results found")
    else:
        for s in scored_results:
            say(f"*Match (score: {s['score']})*\n*Q:* {s['question']}\n*A:* {s['answer']}")


if __name__ == "__main__":
    handler = SocketModeHandler(app, app_token)
    handler.start()