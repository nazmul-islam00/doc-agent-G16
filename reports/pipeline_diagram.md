* [ ] 

# Knowledge-base pipeline diagram (A2)

```text
Scanned Bangla page images (data/raw/krishipath, 403 pages)
                         |
                         v
   Chapter-page selection (378 chapter pages; auxiliary pages excluded)
                         |
                         v
Preprocess: EXIF correction -> deskew -> page crop -> border trim ->
            denoise -> background normalisation -> optional binarisation
                         |
                         v
      Surya layout detection (text / heading / table / figure regions)
                         |
                         v
      Surya multilingual OCR (Bangla + Latin scientific terminology)
                         |
                         v
Unicode NFC and OCR-artifact normalisation -> tokenizer windows
                     (256 tokens, 32-token overlap)
                         |
                         v
SentenceTransformer embeddings (all-MiniLM-L6-v2, 384 dimensions)
                         |
                         v
FAISS HNSW inner-product index
  data/index/index.faiss + chunks.jsonl + metadata.json
```

The raw scans are never overwritten. Preprocessed pages are derived artifacts in
`data/interim/preprocessed/`; the persisted index is the hand-off artifact for A3 retrieval.
