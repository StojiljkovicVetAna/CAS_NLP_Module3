#!/usr/bin/env python3
"""
Test script for the MCP server
"""

import json
from pathlib import Path

# Test loading corpus
corpus_path = Path("/storage/homefs/as23z124/NLP_CAS_M3_project/goodreads_corpus")
corpus_file = corpus_path / "corpus_table.with_topics.jsonl"

if not corpus_file.exists():
    corpus_file = corpus_path / "corpus_table.jsonl"

print(f"Testing corpus file: {corpus_file}")
print(f"Exists: {corpus_file.exists()}")

if corpus_file.exists():
    # Load first few books
    books = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            books.append(json.loads(line))
    
    print(f"\nLoaded {len(books)} books")
    
    for book in books:
        print(f"\n- {book.get('title', 'No title')}")
        print(f"  ID: {book.get('book_id', 'No ID')}")
        print(f"  Author: {book.get('author', 'No author')}")
        print(f"  Topics: {len(book.get('topics', []))}")
        
    print("\n✅ Corpus loading test passed!")
else:
    print("\n❌ Corpus file not found!")
    print(f"Expected at: {corpus_file}")
