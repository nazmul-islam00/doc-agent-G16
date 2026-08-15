# Corpus provenance

- Source (URL): [শস্যের রোগ by Hasan Ashrafuzzaman](https://old.bansdoc.gov.bd/iaeabooks/IAEA%20Books/543%20%E0%A6%B6%E0%A6%B8%E0%A7%8D%E0%A6%AF%E0%A7%87%E0%A6%B0%20%E0%A7%8D%E0%A6%B0%E0%A7%8B%E0%A6%97-%E0%A6%B9%E0%A6%BE%E0%A6%B8%E0%A6%BE%E0%A6%A8%20%E0%A6%86%E0%A6%B6%E0%A6%B0%E0%A6%BE%E0%A6%89%E0%A6%9C%E0%A7%8D%E0%A6%9C%E0%A6%BE%E0%A6%AE%E0%A6%BE%E0%A6%A8.pdf)
- Licence / usage rights: Open access through the Government Digital Library Repository (BANSDOC); verify redistribution terms before publishing the scans.
- Pages: 403   Words: 116,539   Size on disk: 60.13 MB
- Scan/script difficulty notes: Dense printed Bangla with conjuncts and vowel marks, mixed with English and Latin scientific terms. Scans also contain fading, bleed-through, gutter shadows, uneven contrast, and occasional layout elements such as headings, lists, tables, and captions.
- Split policy: The corpus contains a single source volume, so an independent-document split is not possible. We therefore use a fixed, non-overlapping **chapter-grouped split** to reduce within-chapter page leakage:

| Split      | Chapters              | Chapter pages |
| ---------- | --------------------- | ------------: |
| Train      | Remaining 23 chapters |           264 |
| Validation | 1, 12, 17, 25, 32     |            57 |
| Test       | 6, 8, 14, 23, 30      |            57 |

The remaining 25 non-chapter pages (cover, contents, bibliography, and similar
auxiliary material) are excluded from the knowledge-base corpus. The ingestion
pipeline OCR-processes, chunks, embeds, and indexes the **378 chapter pages**.
The source-corpus count remains 403 rendered scans.

OCR fitting and OCR/preprocessing selection use only the training chapters; an internal development subset may be drawn from the training split when required. Validation-chapter questions are used to tune RAG/agent parameters such as retrieval, reranking, re-search, and abstention thresholds. After these choices are frozen, final QA evaluation uses questions derived from the test chapters.

A **stratified manually transcribed subset of pages from the test chapters** is stored under `grading_kit/heldout_pages/`, with corresponding ground-truth records in `grading_kit/labels.jsonl`. These held-out labels are reserved for final OCR evaluation and are never used for OCR training, preprocessing selection, or hyperparameter tuning.

Because the corpus contains a single source volume, a document-level split is not feasible. The chapter-grouped split is therefore used as a leakage-reduction strategy, with complete chapters kept exclusive to train, validation, or test. This constraint will be considered when interpreting the evaluation results.
