#!/usr/bin/env python3
"""
Script-first triage (zero LLM tokens). Mechanically cross-checks CLAUDE.md / FINDINGS.md
against chunk13v9.py / diagnostic_blocks.py / diagnostic_scripts/*.py using ast + regex only.

Design principle (per 2026-09-22 redesign): an LLM call is reserved for genuine semantic
judgment on the SMALL subset of items this script flags as ambiguous. Everything pattern-
matchable happens here, for free, deterministically, and covers the FULL scope (all 31
tickets, all 73 scripts) rather than a 3-ticket pilot -- there is no cost reason to limit
scope once nothing here calls a model.

Outputs:
  triage_report.json  -- full structured findings
  triage_report.md     -- human-readable summary, findings ranked by how actionable they are
"""
import ast
import json
import os
import re
from collections import defaultdict

REPO = "/mnt/hum01-home01/p91688di/PhD_CNMF"
STAGING = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
CLAUDE_PATH = f"{REPO}/CLAUDE.md"
FINDINGS_PATH = f"{REPO}/FINDINGS.md"
CHUNK13V9_PATH = f"{STAGING}/chunk13v9.py"
DIAGBLOCKS_PATH = f"{STAGING}/diagnostic_blocks.py"
SCRIPTS_DIR = f"{STAGING}/diagnostic_scripts"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CORRECTION_TAGS = ["SUPERSEDED", "HISTORICAL", "CORRECTED", "CORRECTION", "STALE",
                    "RETRACTED", "WITHDRAWN", "REFUTED", "REVISED", "REVERSED"]

# Phrases that reliably mark "described/decided but not built" status language, mined for
# free via regex -- this is the raw material for the audit's 1b deliverable (unimplemented
# recommendations). Kept deliberately literal/narrow: each phrase is close to self-evident
# status language on its own, so this needs little semantic judgment to use as a candidate
# list, unlike the unresolved-symbol check.
PLANNED_NOT_IMPLEMENTED_PHRASES = [
    "not yet implemented", "not yet built", "not yet done", "not yet started",
    "not yet run", "not yet acted on", "not scheduled", "proposal only",
    "proposed, not run", "proposed but not run", "no mechanism has been designed",
    "no mechanism", "diagnostic-only", "not implemented in code",
    "status: proposal only", "considered and rejected", "not yet coded",
]


# ---------------------------------------------------------------------------
# AST-based symbol inventory (chunk13v9.py / diagnostic_blocks.py)
# ---------------------------------------------------------------------------

def literal_repr(node):
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        return None


def extract_symbols(path):
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src, filename=path)

    symbols = []  # list of dicts: name, kind, lineno, end_lineno, docstring_first_line, value
    name_lines = defaultdict(list)  # for duplicate-detection at module level

    def visit(node, depth=0, parent=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(child, clean=True)
                doc_first = doc.splitlines()[0] if doc else None
                symbols.append({
                    "name": child.name, "kind": "function" if depth == 0 else "nested_function",
                    "lineno": child.lineno, "end_lineno": getattr(child, "end_lineno", None),
                    "docstring_first_line": doc_first, "parent": parent,
                })
                visit(child, depth + 1, parent=child.name)
            elif isinstance(child, ast.ClassDef):
                symbols.append({"name": child.name, "kind": "class", "lineno": child.lineno,
                                 "end_lineno": getattr(child, "end_lineno", None),
                                 "docstring_first_line": None, "parent": parent})
                visit(child, depth + 1, parent=child.name)
            elif depth == 0 and isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        val = literal_repr(child.value)
                        symbols.append({"name": t.id, "kind": "constant", "lineno": child.lineno,
                                        "end_lineno": getattr(child, "end_lineno", None),
                                        "docstring_first_line": None, "parent": None, "value": val})
                        name_lines[t.id].append(child.lineno)
            else:
                visit(child, depth, parent)

    visit(tree)

    duplicates = {name: lines for name, lines in name_lines.items() if len(lines) > 1}

    # also scan for commented-out assignments that look like a constant definition
    # (e.g. "#MAX_MONOPOLY = 0.60") -- these are a real doc-relevant fact (a value that
    # LOOKS live in a comment but isn't) even though ast can't see them.
    commented_assignments = []
    const_names = {s["name"] for s in symbols if s["kind"] == "constant"}
    for i, line in enumerate(src.splitlines(), start=1):
        m = re.match(r"^\s*#\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$", line)
        if m and (m.group(1) in const_names or m.group(1).isupper()):
            commented_assignments.append({"name": m.group(1), "lineno": i, "commented_value": m.group(2).strip()})

    return {
        "path": path, "symbols": symbols, "duplicate_top_level_names": duplicates,
        "commented_out_assignments": commented_assignments,
    }


# ---------------------------------------------------------------------------
# Doc signal extraction (CLAUDE.md / FINDINGS.md) -- pure regex, no ast (not python)
# ---------------------------------------------------------------------------

def extract_doc_signals(path):
    with open(path) as f:
        lines = f.readlines()

    headings = {}  # "N" or "N.M" -> (lineno, text)
    ticket_mentions = []  # {ticket, lineno, line}
    tag_mentions = []  # {tag, lineno, line}
    planned_mentions = []  # {phrase, lineno, line}
    backtick_symbols = defaultdict(list)  # symbol -> [lineno, ...]
    section_refs = []  # {lineno, ref, explicit_target}

    heading_re = re.compile(r"^(#{1,4})\s+(\d+(?:\.\d+)?)[.\s]")
    ticket_re = re.compile(r"[Tt]icket\s+(\d+)")
    backtick_re = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)")
    section_re = re.compile(r"§\s*(\d+(?:\.\d+)?)")

    for i, line in enumerate(lines, start=1):
        hm = heading_re.match(line)
        if hm:
            headings[hm.group(2)] = (i, line.strip())
        for tm in ticket_re.finditer(line):
            ticket_mentions.append({"ticket": int(tm.group(1)), "lineno": i, "line": line.strip()[:200]})
        for tag in CORRECTION_TAGS:
            if tag in line:
                tag_mentions.append({"tag": tag, "lineno": i, "line": line.strip()[:200]})
        low = line.lower()
        for phrase in PLANNED_NOT_IMPLEMENTED_PHRASES:
            if phrase in low:
                planned_mentions.append({"phrase": phrase, "lineno": i, "line": line.strip()[:220]})
        for bm in backtick_re.finditer(line):
            backtick_symbols[bm.group(1)].append(i)
        for sm in section_re.finditer(line):
            window_start = max(0, sm.start() - 12)
            near_window = line[window_start:sm.start()]
            # "M1 §1.7" / "M3 §5.2" / "Module 3 §5.2" are CODE-INTERNAL module/section labels
            # (a convention used in prose to point at a place inside chunk13v9.py's own
            # informal module structure), never a markdown heading in either doc -- exclude.
            if re.search(r"\bM[1-4]\s*$", near_window) or re.search(r"[Mm]odule\s+\d*\s*$", near_window):
                continue
            window_start2 = max(0, sm.start() - 40)
            window = line[window_start2:sm.start()]
            explicit = None
            if "FINDINGS" in window:
                explicit = "FINDINGS.md"
            elif "CLAUDE" in window:
                explicit = "CLAUDE.md"
            section_refs.append({"lineno": i, "ref": sm.group(1), "explicit_target": explicit,
                                  "line": line.strip()[:200]})

    # dedupe planned_mentions to one entry per line (a status sentence can match >1 phrase,
    # e.g. "not yet implemented" also contains "not yet" -- keep the longest/most specific)
    by_line = {}
    for pm in planned_mentions:
        cur = by_line.get(pm["lineno"])
        if cur is None or len(pm["phrase"]) > len(cur["phrase"]):
            by_line[pm["lineno"]] = pm
    planned_mentions = sorted(by_line.values(), key=lambda x: x["lineno"])

    return {
        "path": path, "n_lines": len(lines), "headings": headings,
        "ticket_mentions": ticket_mentions, "tag_mentions": tag_mentions,
        "planned_mentions": planned_mentions,
        "backtick_symbols": dict(backtick_symbols), "section_refs": section_refs,
    }


def build_citation_graph(claude_sig, findings_sig):
    entries = []
    for src_name, sig, own_headings, other_name, other_headings in [
        ("CLAUDE.md", claude_sig, claude_sig["headings"], "FINDINGS.md", findings_sig["headings"]),
        ("FINDINGS.md", findings_sig, findings_sig["headings"], "CLAUDE.md", claude_sig["headings"]),
    ]:
        for r in sig["section_refs"]:
            ref = r["ref"]
            is_decimal = "." in ref
            # CLAUDE.md's only REAL decimal headings are §4.1-4.23 (confirmed directly from
            # the extracted heading set, not assumed) -- any other decimal ("§5.2", "§1.7",
            # "§2.2"...) is chunk13v9.py's own informal "Module N §X.Y" prose convention
            # reusing the same glyph, never a markdown heading in either doc. Skip those
            # entirely rather than scoring them as citations at all.
            if is_decimal and not ref.startswith("4.") and not r["explicit_target"]:
                continue
            if r["explicit_target"]:
                target_doc = r["explicit_target"]
            elif is_decimal:
                target_doc = "CLAUDE.md"  # only CLAUDE.md uses N.M numbering
            else:
                target_doc = src_name  # default: self-reference

            target_headings = own_headings if target_doc == src_name else other_headings
            resolves = ref in target_headings
            if not resolves and not r["explicit_target"] and not is_decimal:
                # fall back to the other doc before declaring it broken
                fallback_headings = other_headings if target_doc == src_name else own_headings
                if ref in fallback_headings:
                    target_doc = other_name if target_doc == src_name else src_name
                    resolves = True

            entries.append({"source_doc": src_name, "lineno": r["lineno"], "target_doc": target_doc,
                             "target_section": ref, "resolves": resolves, "line": r["line"]})

    broken = [e for e in entries if not e["resolves"]]
    return entries, broken


# ---------------------------------------------------------------------------
# Numeric constant cross-check
# ---------------------------------------------------------------------------

def check_numeric_constants(chunk13v9_syms, claude_sig, findings_sig):
    constants = [s for s in chunk13v9_syms["symbols"] if s["kind"] == "constant" and s.get("value")]
    findings_json = []
    for c in constants:
        name = c["name"]
        val_repr = c["value"]
        try:
            actual = float(ast.literal_eval(val_repr))
        except Exception:
            continue  # not a plain number (e.g. a list/string constant) -- skip
        for doc_name, path in [("CLAUDE.md", CLAUDE_PATH), ("FINDINGS.md", FINDINGS_PATH)]:
            with open(path) as f:
                text = f.read()
            cited_values = set()
            # Require an actual assignment-style token ("=" or ":") directly between the name
            # and the number -- not just proximity. Proximity alone (e.g. "MASTER_SEED + 4
            # offsets", "...400 epochs...INNER_EPOCHS...") produces false positives: numbers
            # that are nearby prose, not a claimed value for the constant.
            for m in re.finditer(re.escape(name) + r"\s*[=:]\s*(-?\d+\.\d+|-?\d+)", text):
                try:
                    cited_values.add(float(m.group(1)))
                except ValueError:
                    pass
            mismatches = sorted(v for v in cited_values if abs(v - actual) > 1e-9)
            if mismatches:
                findings_json.append({
                    "constant": name, "actual_value": actual, "actual_line": c["lineno"],
                    "doc": doc_name, "cited_values_not_matching": mismatches,
                    "all_cited_values": sorted(cited_values),
                })
    return findings_json


# ---------------------------------------------------------------------------
# Diagnostic script mechanical facts
# ---------------------------------------------------------------------------

def extract_script_facts(path):
    with open(path) as f:
        lines = f.readlines()
    src = "".join(lines)

    try:
        tree = ast.parse(src, filename=path)
        module_doc = ast.get_docstring(tree, clean=True)
        module_doc_first = module_doc.splitlines()[0] if module_doc else None
        top_level_defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except SyntaxError as e:
        module_doc_first = None
        top_level_defs = []
        tree = None

    imports_from_diagblocks = []
    for m in re.finditer(r"from\s+diagnostic_blocks\s+import\s+\(?([^)\n]+)\)?", src):
        names = [n.strip() for n in m.group(1).replace("\n", " ").split(",") if n.strip()]
        imports_from_diagblocks.extend(names)
    imports_diagblocks_module = bool(re.search(r"^\s*import\s+diagnostic_blocks", src, re.MULTILINE))

    raw_diag_hits = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # comment-only line (e.g. one narrating a past bug) -- not executed code
        diag_match = re.search(r"np\.diag\(([^()]*(?:\([^()]*\)[^()]*)?)\)", line)
        if not (diag_match or re.search(r"\.diagonal\(", line)):
            continue
        arg = diag_match.group(1) if diag_match else ""
        # np.diag(VECTOR) *builds* a diagonal matrix from a 1-D array -- a different numpy
        # idiom entirely from np.diag(MATRIX) *extracting* an existing matrix's diagonal.
        # Only the latter is the "raw Z reading" concern this check exists for. A `/` or
        # `sqrt(` inside the argument is a strong construction signal (e.g. `1.0/np.sqrt(d)`).
        looks_like_construction = bool(re.search(r"/|sqrt\(|^\s*1\.0\s*,|^\s*np\.ones", arg))
        if looks_like_construction:
            continue
        lhs_match = re.match(r"\s*(\w+)\s*=", line)
        self_labeled_raw = bool(lhs_match and re.search(r"raw", lhs_match.group(1), re.IGNORECASE))
        window = "".join(lines[max(0, i - 4):min(len(lines), i + 3)])
        paired_nearby = bool(re.search(r"_raw\b", window, re.IGNORECASE)) and bool(
            re.search(r"corrected|community_mass|hungarian", window, re.IGNORECASE))
        raw_diag_hits.append({
            "lineno": i, "line": stripped[:160],
            "self_labeled_raw": self_labeled_raw,
            "looks_paired_with_corrected": paired_nearby,
            "looks_paired_or_self_labeled": paired_nearby or self_labeled_raw,
        })

    blocks_version_hits = []
    for i, line in enumerate(lines, start=1):
        if "BLOCKS_VERSION" in line:
            gates = bool(re.search(r"(!=|==|<|>)", line)) and ("blocks_version" in line.lower())
            blocks_version_hits.append({"lineno": i, "line": line.strip()[:160], "looks_like_gate": gates})

    header_what = any(l.strip().startswith("# WHAT:") for l in lines[:10])
    header_out = any(l.strip().startswith("# OUT:") for l in lines[:10])

    out_json_guess = os.path.basename(path).replace(".py", ".json")
    out_json_exists = os.path.exists(os.path.join(STAGING, "diagnostic_results", out_json_guess))

    return {
        "script": os.path.basename(path),
        "module_docstring_first_line": module_doc_first,
        "top_level_defs": top_level_defs,
        "imports_from_diagnostic_blocks": imports_from_diagblocks,
        "imports_diagnostic_blocks_module": imports_diagblocks_module,
        "raw_diagonal_usage": raw_diag_hits,
        "blocks_version_usage": blocks_version_hits,
        "has_what_header": header_what,
        "has_out_header": header_out,
        "output_json_guess": out_json_guess,
        "output_json_exists": out_json_exists,
        "parse_error": tree is None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chunk13v9_syms = extract_symbols(CHUNK13V9_PATH)
    diagblocks_syms = extract_symbols(DIAGBLOCKS_PATH)
    claude_sig = extract_doc_signals(CLAUDE_PATH)
    findings_sig = extract_doc_signals(FINDINGS_PATH)

    citation_entries, broken_citations = build_citation_graph(claude_sig, findings_sig)
    numeric_mismatches = check_numeric_constants(chunk13v9_syms, claude_sig, findings_sig)

    script_paths = sorted(
        os.path.join(SCRIPTS_DIR, f) for f in os.listdir(SCRIPTS_DIR) if f.endswith(".py")
    )
    script_facts = [extract_script_facts(p) for p in script_paths]

    # Cross-check: every backtick-quoted symbol in the docs -- does it resolve anywhere?
    chunk13v9_names = {s["name"] for s in chunk13v9_syms["symbols"]}
    diagblocks_names = {s["name"] for s in diagblocks_syms["symbols"]}
    all_script_def_names = set()
    for sf in script_facts:
        all_script_def_names.update(sf["top_level_defs"])
    known_names = chunk13v9_names | diagblocks_names | all_script_def_names

    doc_symbols = set(claude_sig["backtick_symbols"]) | set(findings_sig["backtick_symbols"])
    BUILTIN_EXCEPTIONS = {"KeyError", "TypeError", "SyntaxError", "RuntimeError",
                           "ModuleNotFoundError", "ValueError", "AttributeError", "IndexError",
                           "NotImplementedError", "FileNotFoundError", "AssertionError"}
    NON_CODE_PROSE = {"FINDINGS", "Absolute", "Infinity", "Journal", "Module", "None",
                       "PhD_CNMF", "TOTAL", "UDSR", "SESSION_PROTOCOL", "chunk13_execution",
                       "chunk13_metafac", "diagnostic_blocks", "diagnostic_results",
                       "diagnostic_scripts", "tensor_data_staging", "toy_large", "time_slice_1"}
    # filter out obvious non-symbol backtick content: short generic words, relation-key names
    # (S_/M_-prefixed -- these are DATA keys inside RELATION_MAP, never Python identifiers to
    # look up), builtin exceptions, env var names, and known non-code path/file tokens.
    plausible_symbols = {
        s for s in doc_symbols
        if len(s) > 3
        and ("_" in s or s[0].islower() is False or s.isupper())
        and not re.match(r"^[SM]_[A-Z]", s)          # relation-key names (S_Art_Auth, M_Atom_Child, ...)
        and s not in BUILTIN_EXCEPTIONS
        and s not in NON_CODE_PROSE
        and not re.match(r"^[A-Z_]+_THREADS$", s)     # OMP_NUM_THREADS et al.
        and not re.match(r"^[A-Z]\d{6,}$", s)         # OpenAlex-style IDs (W2981852735)
    }
    unresolved_symbols = sorted(plausible_symbols - known_names)
    unresolved_note = (
        "First-pass, low-precision signal: this bucket still mixes genuine gaps (a symbol the "
        "docs describe as belonging to chunk13v9.py/diagnostic_blocks.py that isn't there) with "
        "local variable names (never captured as top-level symbols, not a real gap), and names "
        "correctly attributed elsewhere (chunk12.py, old chunk13v3/v4.py, toy_large.ipynb -- out "
        "of this script's scan scope, not missing). Needs a sharper filter (e.g. requiring '(' "
        "immediately after the name, or cross-referencing against chunk12.py/the notebook too) "
        "before being treated as an escalation list on its own."
    )

    # Ambiguity buckets that genuinely need LLM judgment (small, by construction)
    ambiguous_raw_diag = [
        {"script": sf["script"], **hit}
        for sf in script_facts for hit in sf["raw_diagonal_usage"]
        if not hit["looks_paired_or_self_labeled"]
    ]
    scripts_not_importing_diagblocks = [
        sf["script"] for sf in script_facts
        if not sf["imports_from_diagnostic_blocks"] and not sf["imports_diagnostic_blocks_module"]
    ]
    scripts_missing_headers = [
        sf["script"] for sf in script_facts if not (sf["has_what_header"] and sf["has_out_header"])
    ]
    scripts_output_missing = [
        sf["script"] for sf in script_facts if not sf["output_json_exists"]
    ]

    report = {
        "chunk13v9_symbols": chunk13v9_syms,
        "diagnostic_blocks_symbols": diagblocks_syms,
        "citation_graph": {"n_entries": len(citation_entries), "broken": broken_citations},
        "numeric_constant_mismatches": numeric_mismatches,
        "script_facts": script_facts,
        "unresolved_backtick_symbols": unresolved_symbols,
        "unresolved_backtick_symbols_note": unresolved_note,
        "escalate_ambiguous_raw_diagonal": ambiguous_raw_diag,
        "scripts_not_importing_diagnostic_blocks": scripts_not_importing_diagblocks,
        "scripts_missing_session_protocol_headers": scripts_missing_headers,
        "scripts_with_no_result_json": scripts_output_missing,
        "ticket_mentions_count": {
            "CLAUDE.md": len(claude_sig["ticket_mentions"]),
            "FINDINGS.md": len(findings_sig["ticket_mentions"]),
        },
        "tag_mentions_count": {
            "CLAUDE.md": len(claude_sig["tag_mentions"]),
            "FINDINGS.md": len(findings_sig["tag_mentions"]),
        },
        "planned_not_implemented_mentions": {
            "CLAUDE.md": claude_sig["planned_mentions"],
            "FINDINGS.md": findings_sig["planned_mentions"],
        },
    }

    with open(os.path.join(OUT_DIR, "triage_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)

    # human-readable summary
    md = []
    md.append("# Script-first triage report (mechanical, zero LLM tokens)\n")
    md.append(f"- chunk13v9.py: {len(chunk13v9_syms['symbols'])} symbols; "
              f"duplicate top-level names: {chunk13v9_syms['duplicate_top_level_names'] or 'none'}; "
              f"commented-out assignments: {len(chunk13v9_syms['commented_out_assignments'])}\n")
    md.append(f"- diagnostic_blocks.py: {len(diagblocks_syms['symbols'])} symbols; "
              f"duplicate top-level names: {diagblocks_syms['duplicate_top_level_names'] or 'none'}\n")
    md.append(f"- Citation graph: {len(citation_entries)} §N references checked, "
              f"**{len(broken_citations)} broken**\n")
    for b in broken_citations:
        md.append(f"  - BROKEN: {b['source_doc']}:{b['lineno']} -> {b['target_doc']} §{b['target_section']} "
                   f"(`{b['line']}`)\n")
    md.append(f"- Numeric constant mismatches: **{len(numeric_mismatches)}**\n")
    for nm in numeric_mismatches:
        md.append(f"  - `{nm['constant']}` = {nm['actual_value']} (code line {nm['actual_line']}) "
                   f"but {nm['doc']} cites {nm['cited_values_not_matching']}\n")
    md.append(f"- Scripts: {len(script_facts)} total\n")
    md.append(f"  - Not importing diagnostic_blocks at all: {len(scripts_not_importing_diagblocks)} "
              f"{scripts_not_importing_diagblocks}\n")
    md.append(f"  - Ambiguous raw-diagonal usage (no paired corrected reading nearby): "
              f"{len(ambiguous_raw_diag)}\n")
    for a in ambiguous_raw_diag:
        md.append(f"    - {a['script']}:{a['lineno']} `{a['line']}`\n")
    md.append(f"  - Missing SESSION_PROTOCOL `# WHAT:`/`# OUT:` header: {len(scripts_missing_headers)}\n")
    md.append(f"  - No matching result JSON on disk: {len(scripts_output_missing)} {scripts_output_missing}\n")
    md.append(f"- Unresolved backtick-quoted symbols (named in docs, not found in chunk13v9.py, "
              f"diagnostic_blocks.py, or any script's top-level defs): {len(unresolved_symbols)}\n")
    md.append(f"  {unresolved_symbols}\n")
    md.append(f"- Ticket mentions: CLAUDE.md={report['ticket_mentions_count']['CLAUDE.md']}, "
              f"FINDINGS.md={report['ticket_mentions_count']['FINDINGS.md']}\n")
    md.append(f"- Correction-tag mentions: CLAUDE.md={report['tag_mentions_count']['CLAUDE.md']}, "
              f"FINDINGS.md={report['tag_mentions_count']['FINDINGS.md']}\n")
    pni = report["planned_not_implemented_mentions"]
    md.append(f"- \"Planned/not-implemented\" status-language hits (raw material for the 1b "
              f"deliverable): CLAUDE.md={len(pni['CLAUDE.md'])}, FINDINGS.md={len(pni['FINDINGS.md'])}\n")
    for doc_name in ("CLAUDE.md", "FINDINGS.md"):
        for pm in pni[doc_name]:
            md.append(f"    - {doc_name}:{pm['lineno']} [\"{pm['phrase']}\"] `{pm['line']}`\n")

    with open(os.path.join(OUT_DIR, "triage_report.md"), "w") as f:
        f.write("".join(md))

    print("".join(md))


if __name__ == "__main__":
    main()
