# NLP Slack Bot — RFP Assistant

## Goal

This project demonstrates **semantic search via a Slack integration** — specifically, using natural language processing to help sales engineers find answers to RFP (Request for Proposal) questions faster.

The original concept was developed in 2015–2016, when Slack was becoming ubiquitous in company tech stacks. The problem: sales engineers were spending 8–10 hours per RFP responding to repetitive questions from enterprise prospects. The vision was a Slack bot that a sales engineer could query by pasting in an RFP question, and the system would surface the closest matching answer from a library of previously answered questions — shortening time-to-answer and reducing dependence on tribal knowledge.

This repository contains both the original 2016 implementation (on `main`) and a fully working 2026 rebuild (on `modern-rebuild`), demonstrating how the same problem is now solvable with far less infrastructure complexity.

---

## Architecture Evolution

### Original Vision (2015–2016) — `main` branch

| Layer | Technology |
|-------|-----------|
| Slack bot front-end | Howdy + Botkit (Node.js) |
| Cross-language bridge | ZeroRPC over TCP |
| NLP / matching | Seq2Seq model (TensorFlow 1.x, Python 2) |
| Data store | Redis (key-value Q&A pairs) |

The Seq2Seq model was intended to find the closest matching question in the Redis database and return the stored answer.

### Working Implementation (2026) — `modern-rebuild` branch

| Layer | Technology |
|-------|-----------|
| Slack bot front-end | Slack Bolt for Python (Socket Mode) |
| Cross-language bridge | None — single Python service |
| NLP / matching | `sentence-transformers` (`multi-qa-MiniLM-L6-cos-v1`) |
| Vector store | Qdrant (Docker) |
| Data store | Qdrant payload (vectors + Q&A metadata) |

Key architectural changes:
- **Single language**: Moving to Python for everything eliminated the need for ZeroRPC
- **Pre-trained embeddings replace training**: A model pre-trained on 215M question-answer pairs provides semantic understanding out of the box — no training data or GPU required
- **Vector database replaces key-value store**: Qdrant stores both the embeddings and the Q&A payload, enabling cosine similarity search directly against stored questions
- **Socket Mode**: The bot connects to Slack over a persistent WebSocket — no public URL or ngrok tunnel required

---

## How It Works

1. A sales engineer DMs the bot with an RFP question (e.g. *"Do you have any security certifications?"*)
2. `rfp_bot.py` receives the message via Slack Bolt
3. The question is converted to a 384-dimensional vector using `sentence-transformers`
4. Qdrant performs a cosine similarity search against all stored question vectors
5. Results above a confidence threshold are returned to Slack as individual messages, each showing the matched question, stored answer, and similarity score
6. The sales engineer reviews the results, selects the best match, and edits as needed

---

## Project Structure

```
rfp_bot.py          — Slack Bolt app (message handler, search, response formatting)
embeddings.py       — Model loading, vector generation, Qdrant search
db_init.py          — Populates Qdrant from CSV (run once before starting the bot)
RFP_Questions.csv   — 30 sample RFP Q&A pairs
docker-compose.yml  — Runs Qdrant vector database
requirements.txt    — Python dependencies
```

---

## Setup

**Prerequisites**: Python 3.9+, Docker

**1. Start Qdrant:**
```bash
docker compose up -d
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure environment:**

Create a `.env` file in the project root:
```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

**4. Load Q&A data into Qdrant:**
```bash
python db_init.py
```

**5. Start the bot:**
```bash
python rfp_bot.py
```

---

## Slack App Configuration

Required OAuth scopes (Bot Token): `chat:write`, `im:history`, `im:read`

Required event subscription: `message.im`

Socket Mode must be enabled, with an App-Level Token scoped to `connections:write`.

---

## Technologies

- [Slack Bolt for Python](https://slack.dev/bolt-python/) — Slack bot framework
- [sentence-transformers](https://www.sbert.net/) — Pre-trained semantic embedding models
- [Qdrant](https://qdrant.tech/) — Vector database for similarity search
- [Docker](https://www.docker.com/) — Qdrant container runtime
- [python-dotenv](https://pypi.org/project/python-dotenv/) — Environment variable management
