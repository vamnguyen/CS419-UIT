#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Analysis of News Dataset for Search Engine
"""

import json
import re
from collections import Counter

# Load dataset
with open('news_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("📊 DATASET OVERVIEW")
print("=" * 80)
print(f"Total documents: {len(data)}")
print(f"Columns: {list(data[0].keys()) if data else []}")
print()

# Check for missing values
print("=" * 80)
print("⚠️ MISSING VALUES ANALYSIS (CRITICAL for Search)")
print("=" * 80)
fields = ['id', 'title', 'content', 'topic', 'source', 'author', 'url']
for field in fields:
    null_count = sum(1 for d in data if d.get(field) is None)
    empty_count = sum(1 for d in data if d.get(field) == '')
    total = null_count + empty_count
    pct = (total / len(data)) * 100
    print(f"{field}: {total} missing/empty ({pct:.2f}%)")

# Topics distribution
print()
print("=" * 80)
print("📁 TOPICS DISTRIBUTION")
print("=" * 80)
topics = [d.get('topic', '') for d in data]
topic_counts = Counter(topics)
for topic, count in topic_counts.most_common(20):
    display = topic if topic else "(EMPTY)"
    print(f"  {display}: {count}")

# Sources distribution
print()
print("=" * 80)
print("📰 SOURCES DISTRIBUTION")
print("=" * 80)
sources = [d.get('source', '') for d in data]
source_counts = Counter(sources)
for source, count in source_counts.most_common(15):
    display = source if source else "(EMPTY)"
    print(f"  {display}: {count}")

# Content length analysis
print()
print("=" * 80)
print("📏 CONTENT LENGTH ANALYSIS")
print("=" * 80)
content_lengths = [len(d.get('content', '')) for d in data]
print(f"Min content length: {min(content_lengths)} chars")
print(f"Max content length: {max(content_lengths)} chars")
print(f"Average content length: {sum(content_lengths)/len(content_lengths):.0f} chars")
print(f"Documents with empty content: {sum(1 for l in content_lengths if l == 0)}")
print(f"Documents with very short content (<100 chars): {sum(1 for l in content_lengths if l < 100)}")
print(f"Documents with very long content (>10000 chars): {sum(1 for l in content_lengths if l > 10000)}")

# Title length analysis
print()
print("=" * 80)
print("📝 TITLE LENGTH ANALYSIS")
print("=" * 80)
title_lengths = [len(d.get('title', '') or '') for d in data]
print(f"Min title length: {min(title_lengths)} chars")
print(f"Max title length: {max(title_lengths)} chars")
print(f"Average title length: {sum(title_lengths)/len(title_lengths):.0f} chars")
print(f"Documents with empty title: {sum(1 for l in title_lengths if l == 0)}")

# Encoding issues detection
print()
print("=" * 80)
print("🔤 ENCODING ISSUES DETECTION")
print("=" * 80)

# Check for mojibake patterns (corrupted UTF-8)
mojibake_patterns = [
    r'Ã¢|Ã©|Ã¨|Ãº|Ã¹|Ã´|Ã³|Ã²|Ã¬|Ã­|Ãª|á»|áº',  # Common mojibake
    r'â€™|â€œ|â€|â€¢|â€"',  # Smart quotes mojibake
]

encoding_issues = 0
examples = []
for i, doc in enumerate(data[:1000]):  # Check first 1000
    content = doc.get('content', '')
    title = doc.get('title', '')
    text = content + ' ' + title
    for pattern in mojibake_patterns:
        if re.search(pattern, text):
            encoding_issues += 1
            if len(examples) < 3:
                examples.append({
                    'id': doc.get('id'),
                    'title_sample': title[:100],
                })
            break

print(f"Documents with potential encoding issues (in first 1000): {encoding_issues}")
print(f"Estimated total with encoding issues: {(encoding_issues/1000)*len(data):.0f}")
print()
print("Example titles with encoding issues:")
for ex in examples:
    print(f"  ID {ex['id']}: {ex['title_sample']}")

# Duplicate detection
print()
print("=" * 80)
print("🔁 DUPLICATE DETECTION")
print("=" * 80)
title_set = set()
duplicate_titles = 0
for doc in data:
    title = doc.get('title', '')
    if title in title_set:
        duplicate_titles += 1
    title_set.add(title)
print(f"Duplicate titles: {duplicate_titles}")

content_hashes = set()
duplicate_content = 0
for doc in data:
    content = doc.get('content', '')[:500]  # First 500 chars
    if content in content_hashes:
        duplicate_content += 1
    content_hashes.add(content)
print(f"Approximate duplicate content (by first 500 chars): {duplicate_content}")

# Special characters and noise analysis
print()
print("=" * 80)
print("🔊 NOISE DETECTION IN CONTENT")
print("=" * 80)

noise_patterns = {
    'Advertisements/Banners': r'adsbygoogle|window\.adsbygoogle|push\({}\)',
    'HTML tags remaining': r'<[^>]+>',
    'URLs in text': r'https?://\S+',
    'Email addresses': r'\S+@\S+\.\S+',
    'Phone numbers': r'\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
    'JavaScript code': r'function\s*\(|var\s+\w+\s*=|\.push\(',
    'Copyright/Footer text': r'Bản quyền|Giấy phép|Chịu trách nhiệm|Email:|ĐT:',
}

for noise_name, pattern in noise_patterns.items():
    count = sum(1 for d in data if re.search(pattern, d.get('content', '')))
    pct = (count / len(data)) * 100
    print(f"  {noise_name}: {count} docs ({pct:.1f}%)")

# Query difficulty analysis
print()
print("=" * 80)
print("🎯 SEARCH QUERY CHALLENGE ANALYSIS")
print("=" * 80)

# Check for topic overlap (ambiguous terms)
print("\n1. AMBIGUOUS TOPICS (same title could belong to multiple topics):")
all_title_words = []
for doc in data:
    title = doc.get('title', '') or ''
    words = title.lower().split()
    all_title_words.extend(words)

common_title_words = Counter(all_title_words).most_common(20)
print("   Most common title words (may be too generic):")
for word, count in common_title_words:
    if len(word) > 2:
        print(f"     '{word}': {count} occurrences")

# Check for multi-topic documents
print("\n2. DOCUMENTS WITH EMPTY/VAGUE TOPICS:")
vague_topics = sum(1 for d in data if not d.get('topic') or d.get('topic', '').strip() == '')
print(f"   Documents without topic: {vague_topics}")

# Sample problematic documents
print()
print("=" * 80)
print("📋 SAMPLE PROBLEMATIC DOCUMENTS")
print("=" * 80)

print("\n1. Documents with EMPTY content:")
empty_content_docs = [d for d in data if not d.get('content', '').strip()][:3]
for doc in empty_content_docs:
    print(f"   ID: {doc.get('id')}, Title: {doc.get('title', '')[:60]}...")

print("\n2. Documents with VERY SHORT content (<50 chars):")
short_docs = [d for d in data if 0 < len(d.get('content', '')) < 50][:3]
for doc in short_docs:
    print(f"   ID: {doc.get('id')}, Content: '{doc.get('content', '')}'")

print("\n3. Documents with ENCODING issues (sample):")
for i, doc in enumerate(data[:500]):
    content = doc.get('content', '')
    if 'Ã' in content or 'á»' in content[:50]:
        print(f"   ID: {doc.get('id')}")
        print(f"   Title: {doc.get('title', '')[:80]}")
        print(f"   Content sample: {content[:150]}...")
        break
