---
title: Laya Demo
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fast System 1 decisions with calibrated probabilities
---

# Laya Demo

Laya is a fast System 1 decision engine: send a **state** and **typed questions**, get typed answers with probabilities and a
confidence score. It never generates text, so there is nothing to parse and nothing to hallucinate.

| type | question | answer |
|---|---|---|
| `choice` | which of these options? | the option, a probability per option, confidence |
| `score` | where on this rubric? | a position along your levels, probabilities, confidence |
| `noul` | is this true? | the probability that it is |

The tabs are the patterns people use most: support triage, email and phishing, LLM guardrails, RAG passage filtering,
moderation, model routing, and a free-form playground. Every tab asks all of its questions in one pass and then decides
what to do with plain code: the thresholds live in the app, not in the model.

**This is a preview checkpoint** (421M parameters, trained with pure reinforcement learning against proper scoring rules on
public datasets). It is strong on routing, classification, moderation and guardrails, weaker on rubric scores, and it saw
no email data, so treat the email tab as generalisation rather than a trained skill.

Built by [Convai Innovations](https://huggingface.co/convaiinnovations).
