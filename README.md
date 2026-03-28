# 🌐 Website Chatbot (Offline - Transformers)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![Status](https://img.shields.io/badge/Project-Complete-green)

---

## 📌 Project Overview

This project is a **console-based chatbot** that answers questions based on the content of a website.

It works by:

* Scraping website data
* Storing the extracted content
* Using a **local Hugging Face transformer model**
* Generating answers based on the website context

👉 The chatbot runs **fully offline after the initial model download**.

---

## 🚀 Features

* ✅ Website content extraction
* ✅ Context-based question answering
* ✅ Uses **text-generation (TinyLlama model)**
* ✅ Works on CPU (no GPU required)
* ✅ No API key required
* ✅ Fully offline after first run

---

## ⚠️ Important Update

Earlier versions of Transformers supported:

```text
"question-answering"
```

However, in **Transformers >= 4.50**, this task was removed.

### ✅ Fix Applied

* Switched to:

```text
"text-generation"
```

* Used model:

```text
TinyLlama-1.1B-Chat
```

---

## 🧠 Why TinyLlama?

* Small model (~2.2 GB)
* No Hugging Face authentication required
* Runs efficiently on CPU
* Supports instruction-based responses
* Uses chat-style prompting for better answers

---

## ⚙️ Setup Instructions

### 1️⃣ Activate Virtual Environment

```bash
chatbot_env\Scripts\activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Chatbot

```bash
python website_chatbot.py
```

---

## 💬 Example Output

```
You: what is this website about?

Bot: [Thinking — this may take 20-60s on CPU...]

Bot:
BotPenguin is an AI-powered chatbot builder that lets businesses
create chatbots for WhatsApp, Facebook, websites, and more.
```

---

## ⚠️ Performance Notes

* First run will **download the model (~2.2 GB)**
* Response time:

  * ⏳ 20–60 seconds on CPU
* Performance depends on your system specifications

---

## 🛠️ Tech Stack

* Python
* Hugging Face Transformers
* BeautifulSoup (Web Scraping)
* Pickle (Data Storage)

---

## 🔍 How It Works

```text
Website URL
    ↓
Web Scraping (BeautifulSoup)
    ↓
Cleaned Text (Context)
    ↓
Transformer Model (TinyLlama)
    ↓
Generated Answer
```

---

## 🔥 Challenges & Fixes

### ❌ Issue

`question-answering` task is not supported in newer versions of Transformers.

### ✅ Solution

* Switched to `text-generation`
* Implemented prompt-based answering using a chat model

---

## 🎯 Key Learning Outcomes

* Built a complete web scraping pipeline
* Learned how transformer-based text generation works
* Developed an offline AI chatbot system
* Handled breaking changes in libraries effectively

---

## 📌 Limitations

* Not optimized for very large websites
* No semantic search (uses full context)
* Slower response time on CPU

---

## 🚀 Future Improvements

* Add FAISS for semantic search
* Build Streamlit UI
* Integrate Gemini/OpenAI for faster responses
* Implement multi-page crawling

---

## 👨‍💻 Author

**Tharun Pasam**

---

## ⭐ One-line Summary

A local AI chatbot that reads website content and answers questions using a transformer-based text generation model.
