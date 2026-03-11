# research_loop_v1.py
# Pattern: plan -> retrieve -> extract notes -> (repeat) -> synthesize -> verify
# Assumes: you already have a persistent Chroma collection and an embed_text() like you wrote.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import ollama  # pip install ollama


# -----------------------------
# LLM helper (single interface)
# -----------------------------

def llm_json(model: str, system: str, user: str, *, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Ask for JSON only and parse it. Raises if parsing fails.
    Keep prompts strict and short to make small local models behave.
    """
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    text = resp["message"]["content"].strip()
    # Allow models that wrap JSON in ```json fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.replace("json", "", 1).strip()
    return json.loads(text)


def llm_text(model: str, system: str, user: str, *, temperature: float = 0.2) -> str:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None  # distance or similarity depending on your Chroma config


@dataclass
class Note:
    claim: str
    evidence: str                 # short quote / excerpt
    citations: List[str]          # chunk_ids
    confidence: str = "medium"    # low/medium/high
    tags: List[str] = field(default_factory=list)


@dataclass
class ResearchState:
    question: str
    assumptions: List[str] = field(default_factory=list)
    subquestions: List[str] = field(default_factory=list)
    tried_queries: List[str] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)


# -----------------------------
# Retrieval adapter
# -----------------------------

def retrieve(collection, embed_text_fn, query: str, n_results: int = 6) -> List[RetrievedChunk]:
    """
    Minimal retrieval wrapper around Chroma.
    Returns chunk_id + document + optional metadata + optional distance.
    """
    q_emb = embed_text_fn(query)
    res = collection.query(
        query_embeddings=q_emb,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns lists per-query; we have one query.
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else [{} for _ in ids]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [None for _ in ids]

    chunks: List[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        chunks.append(RetrievedChunk(chunk_id=cid, text=doc, metadata=meta or {}, score=dist))
    return chunks


# -----------------------------
# Prompts (keep these tight)
# -----------------------------

PLAN_SYSTEM = (
    "You are a research planner. Output JSON only. "
    "Do not include any prose outside JSON."
)

PLAN_USER_TEMPLATE = """
Question: {question}

Return JSON with:
- assumptions: array of strings (only if needed)
- subquestions: 3-8 focused sub-questions that can be answered from a private book corpus
- query_variants: array of 6-16 search queries (phrases) to retrieve relevant passages

JSON schema:
{{"assumptions":[...],"subquestions":[...],"query_variants":[...]}}
""".strip()


EXTRACT_SYSTEM = (
    "You extract grounded notes from provided passages. Output JSON only. "
    "Rules: Every note must cite chunk_ids. Evidence must be a short excerpt taken from passages. "
    "No new facts beyond passages."
)

EXTRACT_USER_TEMPLATE = """
Main question: {question}

Search query used: {query}

Passages (each has chunk_id + text):
{passages}

Create up to {max_notes} notes. Each note:
- claim: a specific factual statement supported by a passage
- evidence: short excerpt (<= 40 words)
- citations: list of chunk_ids used
- confidence: low/medium/high
- tags: optional short tags

Also return:
- gaps: what is still missing / unclear that would benefit from another retrieval query

JSON schema:
{{"notes":[{{"claim":"...","evidence":"...","citations":["..."],"confidence":"medium","tags":["..."]}}], "gaps":["..."]}}
""".strip()


SYNTH_SYSTEM = (
    "You write a concise report grounded in notes with citations. "
    "Do not invent facts. If evidence is missing, say so."
)

SYNTH_USER_TEMPLATE = """
Question: {question}

Assumptions:
{assumptions}

Grounded notes (each includes citations):
{notes}

Write a report that:
- answers the question directly
- groups key points logically
- cites sources inline as [chunk_id]
- includes a short 'Uncertainties' section listing gaps

Keep it readable and not overly long.
""".strip()


VERIFY_SYSTEM = (
    "You are a strict verifier. Output JSON only. "
    "Goal: find uncited claims, contradictions, or missing parts of the question."
)

VERIFY_USER_TEMPLATE = """
Question: {question}

Report:
{report}

Notes (ground truth evidence):
{notes}

Return JSON:
{{"issues":[...], "fix_suggestions":[...], "needs_more_retrieval": true/false, "followup_queries":[...]}}
""".strip()


# -----------------------------
# Orchestrator (V1)
# -----------------------------

def format_passages(chunks: List[RetrievedChunk], max_chars_each: int = 1200) -> str:
    parts = []
    for ch in chunks:
        t = ch.text.strip()
        if len(t) > max_chars_each:
            t = t[:max_chars_each] + "…"
        parts.append(f"- chunk_id: {ch.chunk_id}\n  text: {t}")
    return "\n".join(parts)


def format_notes(notes: List[Note]) -> str:
    lines = []
    for n in notes:
        cits = ", ".join(n.citations)
        tags = f" tags={n.tags}" if n.tags else ""
        lines.append(f"- claim: {n.claim}\n  evidence: {n.evidence}\n  citations: [{cits}]  confidence={n.confidence}{tags}")
    return "\n".join(lines)


def research_v1(
    *,
    collection,
    embed_text_fn,
    question: str,
    model: str = "qwen2.5:7b-instruct",  # pick whatever local model you have
    max_iterations: int = 6,
    n_results: int = 6,
    max_notes_per_iter: int = 5,
    time_budget_sec: Optional[int] = None,
) -> Tuple[str, ResearchState]:
    state = ResearchState(question=question)
    start = time.time()

    # 1) Plan
    plan = llm_json(
        model=model,
        system=PLAN_SYSTEM,
        user=PLAN_USER_TEMPLATE.format(question=question),
        temperature=0.2,
    )
    state.assumptions = plan.get("assumptions", []) or []
    state.subquestions = plan.get("subquestions", []) or []
    query_variants = plan.get("query_variants", []) or []

    print(plan)

    # 2) Iterate: retrieve -> extract notes -> refine with gaps
    iter_queries = query_variants[:]
    if not iter_queries:
        iter_queries = [question]

    for _ in range(max_iterations):
        if time_budget_sec is not None and (time.time() - start) > time_budget_sec:
            break
        if not iter_queries:
            break

        query = iter_queries.pop(0)
        if query in state.tried_queries:
            continue
        state.tried_queries.append(query)

        chunks = retrieve(collection, embed_text_fn, query=query, n_results=n_results)
        passages_str = format_passages(chunks)

        extracted = llm_json(
            model=model,
            system=EXTRACT_SYSTEM,
            user=EXTRACT_USER_TEMPLATE.format(
                question=state.question,
                query=query,
                passages=passages_str,
                max_notes=max_notes_per_iter,
            ),
            temperature=0.1,
        )

        # Add notes
        for raw in extracted.get("notes", []) or []:
            try:
                note = Note(
                    claim=str(raw.get("claim", "")).strip(),
                    evidence=str(raw.get("evidence", "")).strip(),
                    citations=list(raw.get("citations", []) or []),
                    confidence=str(raw.get("confidence", "medium")).strip(),
                    tags=list(raw.get("tags", []) or []),
                )
                if note.claim and note.citations:
                    state.notes.append(note)
            except Exception:
                continue

        # Update gaps and optionally expand queries
        gaps = extracted.get("gaps", []) or []
        state.gaps = list(dict.fromkeys([*state.gaps, *[str(g).strip() for g in gaps if str(g).strip()]]))

        # Very simple gap -> followup query expansion (V1)
        # You can replace this later with a smarter "query generator" step.
        for g in gaps[:3]:
            q = str(g).strip()
            if q and q not in state.tried_queries and q not in iter_queries:
                iter_queries.append(q)

    # 3) Synthesize
    report = llm_text(
        model=model,
        system=SYNTH_SYSTEM,
        user=SYNTH_USER_TEMPLATE.format(
            question=state.question,
            assumptions="\n".join(f"- {a}" for a in state.assumptions) or "(none)",
            notes=format_notes(state.notes) or "(no notes collected)",
        ),
        temperature=0.2,
    )

    # 4) Verify (optional, V1)
    # If verifier says "needs_more_retrieval", you can loop again; for V1 we just return issues.
    try:
        verification = llm_json(
            model=model,
            system=VERIFY_SYSTEM,
            user=VERIFY_USER_TEMPLATE.format(
                question=state.question,
                report=report,
                notes=format_notes(state.notes),
            ),
            temperature=0.0,
        )
        issues = verification.get("issues", [])
        if issues:
            report += "\n\nVerifier notes (may require more retrieval):\n"
            report += "\n".join(f"- {i}" for i in issues[:8])
    except Exception:
        pass

    return report, state


# -----------------------------
# Minimal usage example
# -----------------------------
if __name__ == "__main__":
    # You supply these from your project:
    # - collection: your persistent Chroma collection
    # - embed_text_fn: your nomic-embed-text wrapper (ollama.embeddings)
    from chroma import get_collection  # your helper file
    from embedding import embed_text  # or just import your embed_text()

    collection = get_collection()

    q = "Throughout history, what are common human behaviors during times of conflict?"
    report, state = research_v1(
        collection=collection,
        embed_text_fn=embed_text,
        question=q,
        model="granite4:3b",  # change to what you have
        max_iterations=3,
        n_results=6,
        max_notes_per_iter=5,
    )

    print("\n" + "=" * 80)
    print(report)
    print("\n" + "=" * 80)
    print(f"Notes collected: {len(state.notes)} | Queries tried: {len(state.tried_queries)}")