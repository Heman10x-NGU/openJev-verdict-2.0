#!/usr/bin/env python3
"""
Scrape all TypeSafe AI cookbooks and architectural patterns into 'RLCD Cookbook/'.
Uses concurrent HTTP fetching with fallback mechanisms and generates an INDEX.md.
"""

import os
import sys
import re
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
COOKBOOK_DIR = WORKSPACE_DIR / "cookbooks"

COOKBOOK_TARGETS = [
    # Core Cookbooks
    {
        "slug": "semantic_find",
        "url": "https://docs.typesafe.ai/cookbooks/semantic_find.md",
        "title": "Line-by-Line Search",
        "category": "Information Retrieval & Search",
        "primitives": "Choice + Noul",
        "summary": "Build semantic search for long documents (e.g. GitHub Terms of Service). Score 218 line IDs with a Choice question and check answer existence with a Noul question."
    },
    {
        "slug": "autoformat",
        "url": "https://docs.typesafe.ai/cookbooks/autoformat.md",
        "title": "Structure Recovery",
        "category": "Document Understanding",
        "primitives": "Choice + Score",
        "summary": "Reconstruct Markdown from corrupted plain text: line stitching and block classification (heading, list, code, callout) in two requests."
    },
    {
        "slug": "function_calling",
        "url": "https://docs.typesafe.ai/cookbooks/function_calling.md",
        "title": "Function Calling",
        "category": "Tool Use & Agents",
        "primitives": "Choice",
        "summary": "Map natural language commands to typed functions and closed-set arguments with calibrated confidence scores."
    },
    {
        "slug": "skill_suggestion",
        "url": "https://docs.typesafe.ai/cookbooks/skill_suggestion.md",
        "title": "Skill Suggestion",
        "category": "Tool Use & Agents",
        "primitives": "Choice + Noul",
        "summary": "Subagent/skill routing across large tool registries (e.g. Nous Hermes 182 skills) with two-stage rank and verify passes."
    },
    {
        "slug": "entity_alignment",
        "url": "https://docs.typesafe.ai/cookbooks/entity_alignment.md",
        "title": "Knowledge Graph Entity Alignment",
        "category": "Data Cleaning & Graphs",
        "primitives": "Score (3 levels)",
        "summary": "Decide merge, unlinked, or curator review for candidate entity pairs using an ordinal Score primitive."
    },
    {
        "slug": "classifying_rag_passages",
        "url": "https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md",
        "title": "Classifying RAG Passages",
        "category": "RAG & Knowledge Retrieval",
        "primitives": "Choice + Noul",
        "summary": "Pre-screen retrieved RAG passages to detect prompt injections, contradictory statements, and low-relevance noise before generation."
    },
    {
        "slug": "citation_check",
        "url": "https://docs.typesafe.ai/cookbooks/citation_check.md",
        "title": "Double-Checking Citations",
        "category": "Evaluation & Factuality",
        "primitives": "Choice",
        "summary": "Detect hallucinated citations by verifying quotes against source text with confidence-gated human-in-the-loop review."
    },
    {
        "slug": "llm_guardrails",
        "url": "https://docs.typesafe.ai/cookbooks/llm_guardrails.md",
        "title": "Guardrails for LLMs",
        "category": "Safety & Moderation",
        "primitives": "Noul + Score",
        "summary": "Screen inputs and outputs in a single forward pass: evaluate multi-hazard presence (jailbreak, PII) and harm severity."
    },
    {
        "slug": "sde_cascade",
        "url": "https://docs.typesafe.ai/cookbooks/sde_cascade.md",
        "title": "Structured Data Extraction (SDE) Cascade",
        "category": "Data Extraction",
        "primitives": "Choice + Score",
        "summary": "Multi-tier extraction cascade (mini -> verify -> reasoning model fallback) maximizing quality while minimizing inference cost."
    },
    {
        "slug": "date_extraction_cookbook",
        "url": "https://docs.typesafe.ai/cookbooks/date_extraction_cookbook.md",
        "title": "Date Extraction",
        "category": "Data Extraction",
        "primitives": "Choice + Noul",
        "summary": "Extract absolute and relative temporal phrases, resolving and validating candidate date tokens in deterministic Python code."
    },
    {
        "slug": "pre_parsed_value_extraction_cookbook",
        "url": "https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md",
        "title": "Pre-Parsed Value Extraction",
        "category": "Data Extraction",
        "primitives": "Choice",
        "summary": "Regex pre-parsing of emails, phone numbers, and currency values combined with calibrated semantic selection."
    },
    {
        "slug": "hierarchical_classification",
        "url": "https://docs.typesafe.ai/cookbooks/hierarchical_classification.md",
        "title": "Hierarchical Classification",
        "category": "Taxonomy & Routing",
        "primitives": "Choice (Tree Search)",
        "summary": "Classify documents across deep taxonomies (patents, product catalogs, biomedical terms) using parallel beam search."
    },
    {
        "slug": "autoresearch_feature_discovery",
        "url": "https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery.md",
        "title": "Autoresearch Feature Discovery",
        "category": "Machine Learning Integration",
        "primitives": "Score + Choice",
        "summary": "Iterative feature engineering loop converting unstructured text into dense calibrated numerical signals for CatBoost/XGBoost."
    },
    {
        "slug": "classification_using_confidence",
        "url": "https://docs.typesafe.ai/cookbooks/classification_using_confidence.md",
        "title": "Classification Using Confidence",
        "category": "Taxonomy & Routing",
        "primitives": "Choice + Confidence Gating",
        "summary": "Classify SEC reports into 75 industry groups with fallback to higher-level divisions whenever confidence falls below threshold."
    },
    {
        "slug": "consistency_noul_cookbook",
        "url": "https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook.md",
        "title": "Self-Consistency: Nouls",
        "category": "Calibration & Reliability",
        "primitives": "Noul",
        "summary": "Detect out-of-distribution inputs and route uncertain binary decisions to human review while keeping probability visible."
    },
    {
        "slug": "consistency_choice_cookbook",
        "url": "https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md",
        "title": "Self-Consistency: Choices",
        "category": "Calibration & Reliability",
        "primitives": "Choice",
        "summary": "Moderation workflow incorporating uncertain outcomes to guarantee calibrated autonomous vs. manual split."
    },
    {
        "slug": "parallel_questions",
        "url": "https://docs.typesafe.ai/cookbooks/parallel_questions.md",
        "title": "Parallel Questions",
        "category": "Performance & Throughput",
        "primitives": "Choice + Score + Noul",
        "summary": "Execute 13 regulatory briefing questions concurrently over a long document in a single non-autoregressive request (12.2x cheaper, 10x faster)."
    },
    {
        "slug": "rerank_typesafe",
        "url": "https://docs.typesafe.ai/cookbooks/rerank_typesafe.md",
        "title": "Re-ranking Candidates",
        "category": "Information Retrieval & Search",
        "primitives": "Choice",
        "summary": "Re-rank 30 BM25 candidate passages for complex legal queries using calibrated pairwise relevance questions, boosting top-1 and top-10 accuracy."
    },
    # Supplementary Architectural Patterns & Demos
    {
        "slug": "patterns_composite_scoring",
        "url": "https://docs.typesafe.ai/patterns/composite-scoring.md",
        "title": "Pattern: Composite Scoring",
        "category": "Architectural Patterns",
        "primitives": "Score",
        "summary": "Decompose complex multifaceted qualitative judgments into atomic Score primitives combined with deterministic software weights."
    },
    {
        "slug": "patterns_confidence_routing",
        "url": "https://docs.typesafe.ai/patterns/confidence-routing.md",
        "title": "Pattern: Confidence-Gated Routing",
        "category": "Architectural Patterns",
        "primitives": "Choice / Score / Noul + Confidence",
        "summary": "Use confidence as an orthogonal decision axis: probabilities determine what to execute, confidence determines whether to act autonomously."
    },
    {
        "slug": "patterns_fan_out",
        "url": "https://docs.typesafe.ai/patterns/fan-out.md",
        "title": "Pattern: Speculative Fan-Out",
        "category": "Architectural Patterns",
        "primitives": "Multi-Question Single Pass",
        "summary": "Send multiple speculative questions in one forward pass and let deterministic application logic decide relevance downstream."
    },
    {
        "slug": "patterns_intent_routing",
        "url": "https://docs.typesafe.ai/patterns/intent-routing.md",
        "title": "Pattern: Intent Routing",
        "category": "Architectural Patterns",
        "primitives": "Choice",
        "summary": "Classify incoming user requests into deterministic rule dispatch, specialized fine-tuned LLM execution, or human escalation."
    },
    {
        "slug": "demos_smart_home",
        "url": "https://docs.typesafe.ai/demos/smart-home.md",
        "title": "Demo: Smart Home Assistant",
        "category": "Demos & Systems",
        "primitives": "Choice + Noul + Score",
        "summary": "Full implementation of a low-latency smart home command interpreter with ambient device state and intent parsing."
    }
]


def clean_markdown(raw_content: str, title: str, source_url: str) -> str:
    """Strip Mintlify index headers and ensure clean frontmatter/structure."""
    lines = raw_content.splitlines()
    
    cleaned_lines = []
    for line in lines:
        if line.startswith("> ## Documentation Index") or "Fetch the complete documentation index at:" in line:
            continue
        if line.startswith("> Use this file to discover all available pages"):
            continue
        cleaned_lines.append(line)
        
    cleaned_text = "\n".join(cleaned_lines).strip()
    
    header = f"---\ntitle: {title}\nsource: {source_url}\nscraped_at: 2026-09-17\n---\n\n"
    return header + cleaned_text + "\n"


def fetch_and_save(target: dict) -> tuple[str, bool, str]:
    slug = target["slug"]
    url = target["url"]
    out_file = COOKBOOK_DIR / f"{slug}.md"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "text/markdown, text/plain, */*"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
            
        formatted_content = clean_markdown(content, target["title"], url)
        out_file.write_text(formatted_content, encoding="utf-8")
        return (slug, True, f"Saved {out_file.name} ({len(formatted_content)} bytes)")
    except Exception as e:
        # Fallback using cloakbrowser if available
        try:
            cloak_path = os.environ.get("CLOAK_BROWSER_PATH")
            if cloak_path and Path(cloak_path).exists():
                sys.path.insert(0, cloak_path)
            from cloakbrowser import launch
            browser = launch(headless=True, humanize=True)
            page = browser.new_page()
            html_url = url.replace(".md", "")
            page.goto(html_url, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            text_content = page.inner_text("body")
            browser.close()
            
            formatted_content = clean_markdown(text_content, target["title"], url)
            out_file.write_text(formatted_content, encoding="utf-8")
            return (slug, True, f"Saved {out_file.name} via CloakBrowser fallback ({len(formatted_content)} bytes)")
        except Exception as e2:
            return (slug, False, f"Failed {slug}: {e} (Fallback error: {e2})")


def generate_index_md(targets: list[dict]):
    index_path = COOKBOOK_DIR / "INDEX.md"
    
    lines = [
        "# RLCD & System 1 Decision Cookbooks",
        "",
        "> Comprehensive collection of 18 non-autoregressive decision recipes, architectural patterns, and production demos extracted from TypeSafe AI.",
        "",
        "## Overview",
        "",
        "Generative language models waste compute producing free-form strings that deterministic code must parse and validate. These cookbooks demonstrate **non-autoregressive System 1 decision engineering**: single forward-pass inference returning calibrated choices, ordinal scores, and binary probabilities directly into software workflows.",
        "",
        "---",
        "",
        "## Cookbooks by Category",
        ""
    ]
    
    # Group by category
    categories: dict[str, list[dict]] = {}
    for t in targets:
        cat = t["category"]
        categories.setdefault(cat, []).append(t)
        
    for cat, items in categories.items():
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Recipe | Primitives | Description |")
        lines.append("|---|---|---|")
        for item in items:
            link = f"[{item['title']}]({item['slug']}.md)"
            lines.append(f"| {link} | `{item['primitives']}` | {item['summary']} |")
        lines.append("")
        
    lines.extend([
        "---",
        "",
        "## Decision Primitives Reference",
        "",
        "- **Choice**: Selects one option from a defined set of categorical labels with calibrated probabilities summing to 1.0.",
        "- **Score**: Evaluates content along an ordered ordinal rubric with descriptive levels, returning expected score and distribution.",
        "- **Noul**: Evaluates binary yes/no propositions with independent calibrated certainty.",
        "- **Confidence**: Reports certainty on a [0.0, 1.0] scale orthogonal to probability, enabling automated confidence-gated policy routing.",
        "",
        "---",
        "*Scraped and cataloged into `cookbooks/` on 2026-09-17.*"
    ])
    
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {index_path.name}")


def main():
    COOKBOOK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting concurrent scrape of {len(COOKBOOK_TARGETS)} cookbooks into: {COOKBOOK_DIR}")
    
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_and_save, target): target for target in COOKBOOK_TARGETS}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"[{'OK' if res[1] else 'ERR'}] {res[2]}")
            
    success_count = sum(1 for r in results if r[1])
    print(f"\nCompleted: {success_count}/{len(COOKBOOK_TARGETS)} successfully scraped.")
    
    generate_index_md(COOKBOOK_TARGETS)


if __name__ == "__main__":
    main()
