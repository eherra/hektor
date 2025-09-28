# 🧠 Hektor

Dead-simple in-memory **vector** library that supports **hybrid (vector + BM25)** retrieval.

- 📝 **Index any texts** with a single call.
- ⚡ **Fast vector similarity** search (cosine/IP/L2).
- 🔎 **Hybrid mode** blends dense and sparse scores (BM25).
- 💾 **Persist & reload** indices to/from disk.
- 🪶 **Lightweight API** built on familiar Python tools.

---

## ✨ Features

- 🧩 Plug-and-play with HuggingFace / Sentence-Transformers models.
- 🌐 Works with multilingual models (default: `intfloat/multilingual-e5-small`).
- 🛠️ Simple `.from_texts()`, `.search()`, `.save()`, `.load()` API.
- 🔑 Customizable:

  - Metric (`cosine`, `ip`, `l2`),
  - Normalization,
  - Hybrid blending `alpha`,
  - Tokenizer (built-in or custom).

- 📦 Zero external server dependencies.

---

## 🚀 Installation

```bash
pip install hektor
```

> Requires Python 3.8+ and, for HuggingFace mode, `sentence-transformers`:
>
> ```bash
> pip install sentence-transformers
> ```

---

## 🏁 Quickstart

```python
from hektor import HektorSearch

searcher = HektorSearch(
    model="sentence-transformers/all-MiniLM-L6-v2",  # choose any model
    hybrid=True                                      # enable dense + BM25
)

# Add data to the index
searcher.from_texts([
    "The cat sat on the mat.",
    "Dogs are great companions.",
    "Cats like naps and cozy corners.",
])

# Single-query search
print(searcher.search("cats and mats", k=2))

# Save & reload index
searcher.save("demo.idx")
t = HektorSearch.load("demo.idx")
print(t.search(["cats", "dogs"], k=2))
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

## 🤝 Contributing

Pull requests, bug reports, and feature suggestions are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for our guidelines.

---

## 🙏 Acknowledgements

- [Sentence-Transformers](https://www.sbert.net/)
- [rank-bm25](https://pypi.org/project/rank-bm25/)

---

## ⭐️ Support

## If you like Hektor, give it a ⭐ on GitHub!
