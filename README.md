# Hektor: Vector Database

> To note: This was build in a 1 day hackathon sprint, so assume rough edges!

Dead-simple in-memory **vector** database that supports **hybrid (vector + BM25)** retrieval.

- 📝 **Index any texts** with a single call.
- ⚡ **Fast vector similarity** search (cosine/IP/L2).
- 🔎 **Hybrid mode** blends dense and sparse scores (BM25).
- 💾 **Persist & reload** indices to/from disk.
- 🌐 **Built-in vector generation** powered by SentenceTransformers / HuggingFace

---

## ✨ Features

- Plug-and-play with HuggingFace / Sentence-Transformers models.
- Simple `.from_texts()`, `.search()`, `.save()`, `.load()` API.
- Customizable:
  - Metric (`cosine`, `ip`, `l2`),
  - Normalization,
  - Hybrid blending `alpha`,
  - Tokenizer (built-in or custom).

---

## 🚀 Installation

> I only published to TestPyPI for now.

```bash
pip install hektor --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/
```

> Requirements: Python ≥ 3.8
> Embeddings: Uses SentenceTransformers models (for example: intfloat/multilingual-e5-small) to generate vector embeddings.
> Tip: On machines with limited RAM, choose a smaller model for faster, lighter inference.

---

## 🏁 Quickstart

```python
from hektor import HektorSearch

# Initialize index
db_index = HektorSearch(
    model="intfloat/multilingual-e5-small",  # Multilingual embedding model
    hybrid=False                             # Enable hybrid search if desired (BM25 + dense)
)

# add sample documents in Finnish, Portuguese, English
docs = [
    # cat docs
    "The cat sat on the mat.",
    "Cats are curious and independent.",
    "The cat chased a mouse under the table.",
    "O gato dorme no tapete.",
    "Os gatos adoram brincar no jardim.",
    "O gato está olhando pela janela.",
    "Kissat tykkäävät nukkua auringossa.",
    "Kissa juoksee pihalla.",
    "Meidän kissa katsoo ikkunasta ulos",

    # dog docs
    "Koira on ystävä.",
    "Koirat rakastavat juosta rannalla.",
    "Dogs are loyal companions.",
    "The dog loves to fetch the ball.",
    "O cão está correndo no parque.",
    "Os cãos gostam de nadar na praia."
]

# Add docs to index
db_index.from_texts(docs)

# Do a search (<search query>, k=<max documents to return>)
search_results = db_index.search("cat watches outside", k=5)

# Display the similarity score between 0 and 1, where 1 indicates perfect similarity
for idx, score in search_results[0]:
    print(f"Similarity: {score:.4f}, Text: {docs[int(idx)]}")

   # Prints:
      # Similarity: 0.8782, Doc: O gato está olhando pela janela.
      # Similarity: 0.8664, Doc: Cats are curious and independent.
      # Similarity: 0.8393, Doc: The cat sat on the mat.
      # Similarity: 0.8309, Doc: Meidän kissa katsoo ikkunasta ulos
      # Similarity: 0.8292, Doc: O gato dorme no tapete.

# Save & reload index
db_index.save("demo.idx")
saved_db = HektorSearch.load("demo.idx")
print(saved_db.search(["cats"], k=2))
```

---

## ⚙️ Configuration

### Constructor parameters

| Parameter   | Default                          | Description                                  |
| ----------- | -------------------------------- | -------------------------------------------- |
| `model`     | `intfloat/multilingual-e5-small` | HuggingFace / Sentence-Transformers model id |
| `embedding` | `None`                           | Extra kwargs passed to embedder factory      |
| `dim`       | `None` (auto-infer)              | Manually override embedding dimension        |
| `metric`    | `"cosine"`                       | `"cosine"`, `"ip"`, or `"l2"`                |
| `normalize` | `True`                           | L2 normalize embeddings                      |
| `hybrid`    | `True`                           | Enable BM25 hybrid scoring                   |
| `alpha`     | `0.5`                            | Blend weight: dense vs BM25                  |
| `tokenizer` | `"simple"`                       | Built-in or custom callable                  |

### Public methods

| Method               | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `.from_texts(texts)` | Build a fresh index from texts                            |
| `.add(texts, ids)`   | Append to an existing index                               |
| `.search(query, k)`  | Search (single string or list) → `[[ (id, score), …], …]` |
| `.save(path)`        | Save index to `path` + `path.meta.json`                   |
| `.load(path)`        | Classmethod: reload index and meta                        |
| `.close()`           | Free resources                                            |

---

## 🗂️ Project Structure

```
hektor/
├── embeddings/        # Embedding backends (HuggingFace, custom)
├── api.py             # Main HektorSearch class
├── index.py           # Vector index implementation
├── hybrid.py         # BM25 + hybrid scoring logic
├── tokenizer.py       # Tokenizers (simple, custom)
```

---

## 🛡️ License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Sentence-Transformers](https://www.sbert.net/)
- [rank-bm25](https://pypi.org/project/rank-bm25/)

---
