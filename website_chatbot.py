"""
===================================================================
  Website Chatbot — Web Scraping + Local Transformers (text-generation)
  Target: https://botpenguin.com/
===================================================================

HISTORY OF FIXES
-----------------
  v1  Mistral raw HTTP           -> 410 Gone (endpoint deprecated)
  v2  LLaMA InferenceClient      -> 404      (gated/paid model)
  v3  RoBERTa serverless QA      -> 410 Gone (serverless API shut down)
  v4  local pipeline("question-answering")
                                 -> KeyError  (task removed in transformers>=4.50)
  v5  (THIS FILE)
      - Task  : "text-generation"   (still supported in all versions)
      - Model : TinyLlama/TinyLlama-1.1B-Chat-v1.0
        * ~2.2 GB, runs on CPU, no GPU needed
        * Ungated — downloads without any HF account
        * Chat-style prompt -> generates a full answer sentence

STEP-BY-STEP PROCESS
----------------------
Step 1: Environment Setup (inside a venv — avoids system conflicts)
  python -m venv chatbot_env
  chatbot_env\\Scripts\\activate        # Windows
  pip install -r requirements.txt
  python website_chatbot.py
  
Step 2: Web Scraping  (WebScraper)
  requests + BeautifulSoup crawl the target site and internal links.
  Scripts/styles/nav/footer stripped; clean text split into chunks.

Step 3: Retrieval  (Retriever)
  Inverted keyword index (TF-IDF-lite) ranks chunks by query overlap.
  Top-K chunks are merged into a context string for the model.

Step 4: Local Text-Generation  (LocalLLMClient)
  pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  Applies the ChatML prompt template the model was fine-tuned on.
  Runs entirely on CPU — no API key, no internet after first download.

Step 5: Console REPL  (WebsiteChatbot)
  Interactive loop. Commands: scrape | url | quit
===================================================================
"""

import os
import re
import sys
import time
import textwrap
from urllib.parse import urljoin, urlparse
from collections import defaultdict

# ── dependency check ──────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Run:  pip install requests beautifulsoup4 transformers torch")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    sys.exit("Run:  pip install transformers torch")


# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DEFAULT_URL   = "https://botpenguin.com/"
MAX_PAGES     = 20        # pages to crawl
CHUNK_SIZE    = 300       # words per chunk
TOP_K         = 4         # chunks merged as context
WRAP          = 88        # console wrap width
REQ_TIMEOUT   = 15        # HTTP timeout (s)

# TinyLlama: ~2.2 GB, CPU-friendly, ungated, supports chat template
MODEL_ID      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOK   = 256       # tokens to generate per answer
CONTEXT_CHARS = 1800      # max chars of scraped context sent to model


# ─────────────────────────────────────────────────────────────────
# STEP 2 — WEB SCRAPER
# ─────────────────────────────────────────────────────────────────
class WebScraper:
    def __init__(self, base_url, max_pages=MAX_PAGES):
        self.base_url  = base_url.rstrip("/")
        self.domain    = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visited   = set()
        self.chunks    = []
        self.session   = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; WebChatBot/1.0)"
        )

    def scrape(self):
        print(f"\n[Scraper] Crawling: {self.base_url}")
        self._crawl(self.base_url)
        print(
            f"[Scraper] Done — {len(self.visited)} page(s), "
            f"{len(self.chunks)} chunk(s).\n"
        )
        return self.chunks

    def _crawl(self, url):
        if len(self.visited) >= self.max_pages or url in self.visited:
            return
        self.visited.add(url)
        print(f"  -> [{len(self.visited)}/{self.max_pages}] {url}")

        try:
            r = self.session.get(url, timeout=REQ_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            print(f"     skip: {e}")
            return

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "img",
                          "video", "audio", "iframe", "nav", "footer", "header"]):
            tag.decompose()

        raw = soup.get_text(" ")
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

        words = raw.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i: i + CHUNK_SIZE])
            if len(chunk) > 60:
                self.chunks.append(chunk)

        for a in soup.find_all("a", href=True):
            abs_url = urljoin(url, a["href"]).split("#")[0].split("?")[0]
            if (
                urlparse(abs_url).netloc == self.domain
                and abs_url not in self.visited
                and abs_url.startswith("http")
            ):
                time.sleep(0.25)
                self._crawl(abs_url)


# ─────────────────────────────────────────────────────────────────
# STEP 3 — RETRIEVER (keyword TF-IDF-lite)
# ─────────────────────────────────────────────────────────────────
class Retriever:
    STOP = {
        "a","an","the","is","it","in","on","at","to","of","and","or",
        "for","with","this","that","are","was","were","be","been","has",
        "have","had","do","does","did","but","not","i","you","we","they",
        "he","she","what","how","when","where","which","who","will","can",
        "could","should","would","my","me","your","our","their","its","about"
    }

    def __init__(self, chunks):
        self.chunks = chunks
        self._idx   = self._build()

    def _tok(self, text):
        return {t for t in re.findall(r"[a-z]+", text.lower())
                if t not in self.STOP}

    def _build(self):
        idx = defaultdict(list)
        for i, c in enumerate(self.chunks):
            for w in self._tok(c):
                idx[w].append(i)
        return idx

    def top(self, query, k=TOP_K):
        sc = defaultdict(int)
        for w in self._tok(query):
            for i in self._idx.get(w, []):
                sc[i] += 1
        if not sc:
            return self.chunks[:k]
        ranked = sorted(sc, key=sc.__getitem__, reverse=True)
        return [self.chunks[i] for i in ranked[:k]]


# ─────────────────────────────────────────────────────────────────
# STEP 4 — LOCAL LLM  (text-generation, no API needed)
# ─────────────────────────────────────────────────────────────────
class LocalLLMClient:
    """
    Runs TinyLlama-1.1B-Chat locally via transformers pipeline.

    Why TinyLlama?
      - Uses the 'text-generation' task (supported in ALL transformers versions)
      - ~2.2 GB download, runs on CPU (no GPU required)
      - Completely ungated — no HF account or token needed
      - Fine-tuned with ChatML format for instruction following
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions "
        "only based on the website content provided. "
        "Keep your answer short and factual. "
        "If the answer is not in the content, say: "
        "'I couldn't find that on the website.'"
    )

    def __init__(self):
        print(f"[Model] Loading '{MODEL_ID}' …")
        print("        (First run: ~2.2 GB download. Subsequent runs are instant.)\n")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float32,   # CPU-safe (use float16 if you have GPU)
            device_map="auto",
        )
        print("[Model] Ready.\n")

    def _build_prompt(self, context: str, question: str) -> str:
        """Build ChatML prompt that TinyLlama was trained on."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Website content:\n{context[:CONTEXT_CHARS]}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        # apply_chat_template adds <|im_start|> / <|im_end|> tokens
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def ask(self, question: str, context: str) -> str:
        prompt = self._build_prompt(context, question)
        try:
            out = self.pipe(
                prompt,
                max_new_tokens=MAX_NEW_TOK,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # The pipeline returns the full string including the prompt.
            # Extract only the newly generated part.
            generated = out[0]["generated_text"]
            # Everything after the last <|im_start|>assistant marker
            marker = "<|im_start|>assistant\n"
            if marker in generated:
                answer = generated.split(marker)[-1].strip()
                # strip trailing end token if present
                answer = answer.replace("<|im_end|>", "").strip()
            else:
                answer = generated[len(prompt):].strip()
            return answer if answer else "I couldn't find that on the website."
        except Exception as e:
            return f"[Error] {e}"


# ─────────────────────────────────────────────────────────────────
# STEP 5 — CONSOLE CHATBOT (REPL)
# ─────────────────────────────────────────────────────────────────
class WebsiteChatbot:

    BANNER = """
+--------------------------------------------------------------+
|  Website Chatbot  (TinyLlama local LLM + BeautifulSoup)      |
|  No API key required — runs entirely on your machine         |
+--------------------------------------------------------------+
  Commands:  scrape | url | quit
"""

    def __init__(self):
        self.model     = None
        self.retriever = None
        self.url       = None

    def _get_url(self):
        print(f"Default URL: {DEFAULT_URL}")
        u = input("Enter website URL (or press Enter for default): ").strip()
        return u or DEFAULT_URL

    def _scrape(self, url):
        chunks = WebScraper(url).scrape()
        if not chunks:
            print("[!] Nothing scraped — check the URL.")
            return False
        self.retriever = Retriever(chunks)
        self.url = url
        return True

    def run(self):
        print(self.BANNER)

        # Load model first (one-time download)
        self.model = LocalLLMClient()

        url = self._get_url()
        self._scrape(url)
        print(f"[Ready] Ask anything about: {self.url}\n")

        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not q:
                continue
            cmd = q.lower()

            if cmd in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            elif cmd == "scrape":
                self._scrape(self.url)
                print("[Done] Re-scraped.\n")
                continue
            elif cmd == "url":
                nu = input("New URL: ").strip()
                if nu:
                    self._scrape(nu)
                    print(f"[Done] Now using: {self.url}\n")
                continue

            if not self.retriever:
                print("Bot: No knowledge base. Type 'scrape'.\n")
                continue

            chunks  = self.retriever.top(q)
            context = " ".join(chunks)

            print("Bot: [Thinking — this may take 20-60s on CPU...]\n")
            answer = self.model.ask(q, context)
            print("Bot:\n" + textwrap.fill(answer, WRAP) + "\n")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WebsiteChatbot().run()