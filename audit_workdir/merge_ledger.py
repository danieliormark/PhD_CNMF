#!/usr/bin/env python3
"""Stage A-merge (mechanical, no agent). Unions all Stage A outputs into:
  - claims_index.json   (all doc claims from claude.json + findings_1/2/3.json)
  - facts_index.json    (all ground-truth facts: chunk13v9_facts, diagnostic_blocks_facts, scripts_1-6)
  - citation_graph.json (every §N cross-reference claim, resolved against real headings)
  - symbol_diff.json    (production symbols with zero doc references -- candidate undocumented mechanisms)
"""
import json
import os
import re

LEDGER_DIR = os.path.dirname(os.path.abspath(__file__)) + "/ledger"
CLAUDE_PATH = "/mnt/hum01-home01/p91688di/PhD_CNMF/CLAUDE.md"
FINDINGS_PATH = "/mnt/hum01-home01/p91688di/PhD_CNMF/FINDINGS.md"

VALID_TICKETS = {2,23,31,32,35,36,43,46,47,52,60,63,64,66,68,69,71,73,74,75,76,77,78,79,80,81,82,84,85,86,87}


def load(name):
    with open(os.path.join(LEDGER_DIR, name)) as f:
        return json.load(f)


def normalize_ticket(t):
    if t in VALID_TICKETS:
        return t
    return None  # invalid/out-of-range values (e.g. 88-92 seen in scripts_3.json) get nulled, not guessed


def build_claims_index():
    claim_files = ["claude.json", "findings_1.json", "findings_2.json", "findings_3.json"]
    all_claims = []
    seen_ids = {}
    dupes = []
    for fname in claim_files:
        for c in load(fname):
            cid = c.get("claim_id")
            if cid in seen_ids:
                dupes.append((cid, seen_ids[cid], fname))
                # keep both but rename the second to avoid silent overwrite
                cid = f"{cid}-DUP-{fname.split('.')[0]}"
                c["claim_id"] = cid
                c["_original_id_collision"] = True
            seen_ids[cid] = fname
            c["_source_ledger_file"] = fname
            all_claims.append(c)
    with open(os.path.join(LEDGER_DIR, "claims_index.json"), "w") as f:
        json.dump({"claims": all_claims, "n": len(all_claims), "id_collisions": dupes}, f, indent=1)
    return all_claims, dupes


def build_facts_index():
    fact_files = [
        "chunk13v9_facts.json", "diagnostic_blocks_facts.json",
        "scripts_1.json", "scripts_2.json", "scripts_3.json",
        "scripts_4.json", "scripts_5.json", "scripts_6.json",
    ]
    all_facts = []
    for fname in fact_files:
        for item in load(fname):
            item["_source_ledger_file"] = fname
            # normalize the odd ticket_relevance values seen in scripts_3.json (88-92 -> invalid)
            if "ticket_relevance" in item:
                tr = item["ticket_relevance"]
                norm = normalize_ticket(tr)
                if norm != tr:
                    item["_ticket_relevance_raw"] = tr
                    item["ticket_relevance"] = norm if norm is not None else 82  # this cluster's default
                    item["_ticket_relevance_normalized_note"] = (
                        f"raw value {tr} is not a real ticket number; "
                        f"reassigned to 82 (domain-balance/E2 exploitability cluster) per merge-time fix"
                    )
            all_facts.append(item)
    with open(os.path.join(LEDGER_DIR, "facts_index.json"), "w") as f:
        json.dump({"facts": all_facts, "n": len(all_facts)}, f, indent=1)
    return all_facts


def get_headings(path):
    """Return dict of {heading_number_str: (line_no, text)} for '## N.' / '### N.M' style headings."""
    headings = {}
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            m = re.match(r"^#{1,4}\s+(\d+(?:\.\d+)?)[.\s]", line)
            if m:
                headings[m.group(1)] = (i, line.strip())
    return headings


def build_citation_graph(all_claims):
    claude_headings = get_headings(CLAUDE_PATH)
    findings_headings = get_headings(FINDINGS_PATH)

    entries = []
    pattern = re.compile(r"§\s*(\d+(?:\.\d+)?)")

    for c in all_claims:
        if c.get("claim_type") != "cross-reference":
            continue
        text = c.get("claim_text", "")
        refs = pattern.findall(text)
        # Per-claim hint (from extraction agent's referenced_files / source doc) -- used only
        # as a tiebreaker for genuinely ambiguous bare refs, never to override the structural
        # fact that CLAUDE.md's own top-level numbering tops out at §11.
        target_files = c.get("referenced_files", [])
        hint_doc = None
        if any("FINDINGS" in t for t in target_files):
            hint_doc = "FINDINGS.md"
        elif any("CLAUDE" in t for t in target_files):
            hint_doc = "CLAUDE.md"
        elif any("SESSION_PROTOCOL" in t for t in target_files):
            hint_doc = "SESSION_PROTOCOL.md"

        for ref in refs:
            # Resolve target_doc PER REFERENCE, not per claim -- a single claim can cite
            # sections in two different files (e.g. "§17 and §4.15").
            if "." in ref:
                # decimal subsection (§4.N) only exists in CLAUDE.md's numbering
                target_doc = "CLAUDE.md"
            else:
                n = int(ref)
                if n > 11:
                    # bare §N > 11 cannot be a CLAUDE.md top-level heading (tops out at §11)
                    target_doc = "FINDINGS.md"
                elif hint_doc in ("CLAUDE.md", "FINDINGS.md"):
                    target_doc = hint_doc
                else:
                    target_doc = "ambiguous"

            if target_doc == "CLAUDE.md":
                resolves = ref in claude_headings
            elif target_doc == "FINDINGS.md":
                resolves = ref in findings_headings
            elif target_doc == "ambiguous":
                resolves = (ref in claude_headings) or (ref in findings_headings)
            else:
                resolves = None

            entries.append({
                "claim_id": c.get("claim_id"),
                "source_doc": c.get("source_doc"),
                "source_location": c.get("source_location"),
                "target_doc": target_doc,
                "target_section": ref,
                "resolves": resolves,
                "claim_text": text,
            })

    broken = [e for e in entries if e["resolves"] is False]
    unresolved_target = [e for e in entries if e["target_doc"] is None]

    with open(os.path.join(LEDGER_DIR, "citation_graph.json"), "w") as f:
        json.dump({
            "entries": entries,
            "n_entries": len(entries),
            "n_broken": len(broken),
            "broken": broken,
            "n_unresolved_target_doc": len(unresolved_target),
            "claude_headings_found": sorted(claude_headings.keys(), key=lambda x: [int(p) for p in x.split(".")]),
            "findings_headings_found": sorted(findings_headings.keys(), key=lambda x: [int(p) for p in x.split(".")]),
        }, f, indent=1)
    return entries, broken


def build_symbol_diff(all_claims, all_facts):
    # production symbols from chunk13v9_facts.json (kind in function/constant/nested_function)
    chunk13v9_facts = [f for f in all_facts if f.get("_source_ledger_file") == "chunk13v9_facts.json"]
    prod_symbols = {
        f["symbol"] for f in chunk13v9_facts
        if f.get("kind") in ("function", "constant", "nested_function") and f.get("symbol")
    }

    referenced = set()
    for c in all_claims:
        for s in c.get("referenced_symbols", []) or []:
            referenced.add(s)
    for f in all_facts:
        for s in f.get("referenced_symbols", []) or []:
            referenced.add(s)
        # scripts facts list imports/reimplementations that count as "referencing" a symbol
        for s in (f.get("imports_from_diagnostic_blocks") or []):
            referenced.add(s)

    undocumented = sorted(prod_symbols - referenced)
    with open(os.path.join(LEDGER_DIR, "symbol_diff.json"), "w") as f:
        json.dump({
            "production_symbols_from_chunk13v9": sorted(prod_symbols),
            "referenced_anywhere_in_ledger": sorted(referenced),
            "undocumented_candidates": undocumented,
            "note": "undocumented_candidates are chunk13v9.py functions/constants with zero references "
                    "anywhere in the claims/facts ledger for this pilot's scope (tickets 82/86/87 material only "
                    "-- a symbol could still be documented elsewhere, e.g. batches 1/2/4/5's sections, not covered here).",
        }, f, indent=1)
    return undocumented


def main():
    all_claims, dupes = build_claims_index()
    all_facts = build_facts_index()
    _, broken = build_citation_graph(all_claims)
    undocumented = build_symbol_diff(all_claims, all_facts)

    print(f"claims_index.json: {len(all_claims)} claims, {len(dupes)} id collisions: {dupes}")
    print(f"facts_index.json: {len(all_facts)} facts")
    print(f"citation_graph.json: broken links: {len(broken)}")
    for b in broken:
        print(f"   BROKEN: {b['source_doc']} {b['source_location']} -> {b['target_doc']} §{b['target_section']}")
    print(f"symbol_diff.json: {len(undocumented)} undocumented candidates: {undocumented}")


if __name__ == "__main__":
    main()
