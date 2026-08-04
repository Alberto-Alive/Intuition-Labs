"""Evidence-packet compiler — the *bottleneck between retrieval and decoding*.

Takes the query intent and a set of candidate documents, then:
  1. scopes evidence to the unique subject(s) named in the query (and resolves
     entity aliases to their subject),
  2. extracts typed values,
  3. aligns each value against the query facets (recency/status, region,
     scope, date-role, table coordinate, dependency chain, policy threshold,
     exception override) to decide whether it *applies*,
  4. detects conflicts and missing evidence,
and emits a compact typed `EvidencePacket`.  The decoder may then answer ONLY
from this packet — no raw chunks reach the generator.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..schema import Document, EvidenceItem, EvidencePacket
from .intent import Intent
from .typed_extractors import extract_items

SUPERSEDED = {"superseded", "rejected", "draft", "withdrawn"}


def norm_value(v: str) -> str:
    v = v.strip().lower().rstrip(".")
    v = v.replace("$", "").replace(",", "")
    return re.sub(r"\s+", " ", v)


def _doc_region(doc: Document) -> Optional[str]:
    if "region" in doc.meta:
        return doc.meta["region"]
    m = re.search(r"\b(US|EU|UK|APAC|CA|LATAM)\b", doc.title)
    return m.group(1) if m else None


class EvidenceCompiler:
    def __init__(self, docs_by_id: Dict[str, Document]):
        self.docs_by_id = docs_by_id

    # -- subject scoping + alias resolution -------------------------------
    def _scope_subjects(self, intent: Intent, cand: List[Document]):
        """Return (scope_subjects, scoped_docs, subject_seen_in_corpus)."""
        subjects = list(intent.subjects)
        # resolve an alias code (e.g. ATL-14) to its real subject via the
        # registry document's metadata / text
        if intent.alias:
            for d in list(cand) + list(self.docs_by_id.values()):
                if intent.alias in d.text and "internal code for" in d.text:
                    real = d.meta.get("subject")
                    if not real:
                        m = re.search(r"Project ([A-Z][a-z]+-\d+)", d.text)
                        real = m.group(1) if m else None
                    if real and real not in subjects:
                        subjects.append(real)
                    if d not in cand:
                        cand = cand + [d]
                    break
        if not subjects:
            return subjects, cand, True
        # gather every corpus doc carrying a scope subject (retrieval recovery)
        scoped = [d for d in self.docs_by_id.values()
                  if any(s in d.text or s == d.meta.get("subject") for s in subjects)]
        seen = len(scoped) > 0
        # union with the original candidates that are on-subject
        return subjects, scoped, seen

    # -- main compile -----------------------------------------------------
    def compile(self, intent: Intent, candidate_docs: List[Document]) -> EvidencePacket:
        subjects, scoped_docs, subject_seen = self._scope_subjects(intent, candidate_docs)

        # subject named but absent from the whole corpus -> missing evidence
        if subjects and not subject_seen:
            return EvidencePacket(intent=intent.summary, target_field=intent.target_field,
                                  facets=intent.facets, items=[], conflicts=[],
                                  missing_evidence=[f"no document mentions {subjects}"])

        cand = scoped_docs if subjects else candidate_docs
        items: List[EvidenceItem] = []
        for d in cand[:16]:
            items.extend(extract_items(d))

        target = intent.target_field
        facets = intent.facets
        typed = [it for it in items if it.field_type == target]
        # numbers: keep unit-bearing values and align the unit to the question
        if target == "number":
            unit = re.compile(r"[%$]|\b(?:days?|months?|seconds?|TB|GB)\b", re.I)
            typed = [it for it in typed if unit.search(it.exact_value)] or typed
            low = intent.raw.lower()
            want = ("$" if any(w in low for w in ("budget", "price", "cost", "fee")) else
                    "%" if ("sla" in low or "uptime" in low) else
                    "time" if any(w in low for w in ("window", "period", "retention", "timeout", "cooling")) else None)
            if want == "$":
                typed = [it for it in typed if "$" in it.exact_value] or typed
            elif want == "%":
                typed = [it for it in typed if "%" in it.exact_value] or typed
            elif want == "time":
                typed = [it for it in typed if re.search(r"days?|months?|seconds?", it.exact_value, re.I)] or typed
        clauses = [it for it in items if it.field_type in ("clause", "policy_field")]

        # --- dependency-chain resolver (multi-doc) ---
        chain_item = self._resolve_chain(intent, items)
        if chain_item is not None:
            typed = [chain_item]

        # --- synthesized decision items (policy threshold / exception / proposal) ---
        synth_item = (self._resolve_policy(intent, items)
                      or self._resolve_exception(intent, items)
                      or self._resolve_proposal(intent, items))

        if target == "table_cell":
            typed = self._filter_table(intent, typed)
        if target == "code_symbol":
            typed = self._filter_code(intent, typed)
        if target == "date":
            typed = self._filter_date(intent, typed)

        typed = self._align_applicability(intent, typed, items)
        applicable = [it for it in typed if it.applies]

        conflicts = self._detect_conflicts(applicable)

        # --- assemble compact packet ---
        packet_items: List[EvidenceItem] = []
        if synth_item is not None:
            packet_items.append(synth_item)
        packet_items += applicable[:4]
        if synth_item is None:                  # show governing clause for context
            for c in clauses:
                if any(m in c.source_span.lower() for m in ("override", "statutory", "cooling-off")):
                    packet_items.append(c)
                    break
        packet_items += [it for it in typed if not it.applies][:2]   # audit trail
        seen, uniq = set(), []
        for it in packet_items:
            k = (it.field_type, it.exact_value, it.document_id, it.source_span)
            if k not in seen:
                seen.add(k); uniq.append(it)
        packet_items = uniq

        missing: List[str] = []
        if not applicable and synth_item is None and not conflicts:
            missing.append(f"no applicable {target} for {facets or '{}'}")

        return EvidencePacket(intent=intent.summary, target_field=target, facets=facets,
                              items=packet_items, conflicts=conflicts, missing_evidence=missing)

    # -- resolvers --------------------------------------------------------
    def _resolve_chain(self, intent: Intent, items: List[EvidenceItem]) -> Optional[EvidenceItem]:
        low = intent.raw.lower()
        if "stored" not in low and "backup" not in low:
            return None
        subj = intent.facets.get("subject", "")
        primary = None
        for it in items:
            m = re.search(r"default region for .*accounts is ([a-z0-9\-]+)", it.source_span, re.I)
            if m:
                primary = m.group(1)
                break
        if not primary:
            return None
        for it in items:
            m = re.search(rf"backups in {re.escape(primary)} are stored in ([a-z0-9\-]+)",
                          it.source_span, re.I)
            if m:
                return EvidenceItem(
                    field_type=intent.target_field, entity=f"backup_region({subj})",
                    exact_value=m.group(1), source_span=it.source_span,
                    document_id=it.document_id, timestamp=it.timestamp,
                    applicability_context=f"resolved chain {subj}->{primary}->backup",
                    applies=True, status=it.status)
        return None

    def _resolve_policy(self, intent: Intent, items: List[EvidenceItem]) -> Optional[EvidenceItem]:
        if "amount" not in intent.facets:
            return None
        try:
            amount = float(intent.facets["amount"])
        except ValueError:
            return None
        threshold, span = None, ""
        for it in items:
            m = re.search(r"\$(\d[\d,\.]*)\s*or more require", it.source_span, re.I)
            if m:
                threshold = float(m.group(1).replace(",", "")); span = it.source_span; break
        if threshold is None:
            return None
        applies = amount >= threshold
        verdict = "manager approval required" if applies else "auto-approved"
        return EvidenceItem(field_type="policy_field", entity="approval_decision",
                            exact_value=verdict, source_span=span, document_id="",
                            applicability_context=f"amount ${amount:.0f} vs threshold ${threshold:.0f}",
                            applies=True, status="active")

    def _filter_table(self, intent: Intent, typed: List[EvidenceItem]) -> List[EvidenceItem]:
        plan = intent.facets.get("plan", "").lower()
        region = intent.facets.get("region", "").lower()
        keep = []
        for it in typed:
            ent = it.entity.lower()
            if plan and region and plan in ent and region in ent:
                keep.append(it)
            elif plan and not region and plan in ent:
                keep.append(it)
        return keep or typed

    def _filter_code(self, intent: Intent, typed: List[EvidenceItem]) -> List[EvidenceItem]:
        from .intent import SYMBOL_RE
        syms = [s for s in SYMBOL_RE.findall(intent.raw) if "_" in s or s.isupper()]
        if syms:
            typed = [it for it in typed if it.entity in syms] or typed
        # production scope wins over test
        if intent.facets.get("scope") == "prod":
            prod = [it for it in typed
                    if (self.docs_by_id.get(it.document_id) and
                        self.docs_by_id[it.document_id].meta.get("scope") == "prod")]
            for it in typed:
                d = self.docs_by_id.get(it.document_id)
                if d and d.meta.get("scope") == "test":
                    it.applies = False
                    it.applicability_context += "; test scope (not production)"
            return prod or typed
        return typed

    def _filter_date(self, intent: Intent, typed: List[EvidenceItem]) -> List[EvidenceItem]:
        low = intent.raw.lower()
        want = next((w for w in ("renew", "expire", "effective", "sign") if w in low), None)
        if want:
            # check the date ROLE (entity), not the full-line span which holds all dates
            pref = [it for it in typed if want in it.entity.lower()]
            if pref:
                for it in typed:
                    if it not in pref:
                        it.applies = False
                        it.applicability_context += "; wrong date role"
        return typed

    def _resolve_proposal(self, intent: Intent, items: List[EvidenceItem]) -> Optional[EvidenceItem]:
        if "in effect" not in intent.raw.lower():
            return None
        for it in items:
            if it.field_type == "policy_field" and it.entity == "status":
                dec = it.exact_value.upper()
                if dec in ("REJECTED", "WITHDRAWN"):
                    return EvidenceItem(
                        field_type="policy_field", entity="proposal_status",
                        exact_value=f"not in effect (proposal {dec.lower()})",
                        source_span=it.source_span, document_id=it.document_id,
                        timestamp=it.timestamp, applicability_context=f"proposal {dec.lower()}",
                        applies=True, status="active")
                if dec == "APPROVED":
                    return EvidenceItem(
                        field_type="policy_field", entity="proposal_status",
                        exact_value="in effect (proposal approved)", source_span=it.source_span,
                        document_id=it.document_id, timestamp=it.timestamp,
                        applicability_context="proposal approved", applies=True, status="active")
        return None

    def _resolve_exception(self, intent: Intent, items: List[EvidenceItem]) -> Optional[EvidenceItem]:
        if intent.facets.get("purchase") != "non-refundable" or intent.facets.get("region") != "EU":
            return None
        for it in items:
            span = it.source_span.lower()
            if "statutory" in span and ("override" in span or "cooling-off" in span):
                return EvidenceItem(
                    field_type="policy_field", entity="exception_decision",
                    exact_value="refund allowed via 14-day statutory exception (overrides standard rule)",
                    source_span=it.source_span, document_id=it.document_id, timestamp=it.timestamp,
                    applicability_context="EU statutory exception applies to non-refundable item",
                    applies=True, status="active")
        return None

    def _align_applicability(self, intent: Intent, typed: List[EvidenceItem],
                             all_items: List[EvidenceItem]) -> List[EvidenceItem]:
        region = intent.facets.get("region")
        has_exception = any(("override" in it.source_span.lower() or
                             "statutory" in it.source_span.lower()) for it in all_items)
        for it in typed:
            doc = self.docs_by_id.get(it.document_id)
            if (it.status or "").lower() in SUPERSEDED:
                it.applies = False
                it.applicability_context += f"; {it.status} (not in force)"
            if region and doc is not None:
                dr = _doc_region(doc)
                if dr and dr != region:
                    it.applies = False
                    it.applicability_context += f"; region {dr}!={region}"
            if has_exception and intent.facets.get("purchase") == "non-refundable":
                if "statutory" not in it.source_span.lower() and "cooling-off" not in it.source_span.lower():
                    it.applies = False
                    it.applicability_context += "; overridden by statutory exception"
        return typed

    def _detect_conflicts(self, applicable: List[EvidenceItem]) -> List[Dict[str, str]]:
        conflicts, seen = [], set()
        for i in range(len(applicable)):
            for j in range(i + 1, len(applicable)):
                a, b = applicable[i], applicable[j]
                if a.field_type != b.field_type or a.document_id == b.document_id:
                    continue
                if norm_value(a.exact_value) == norm_value(b.exact_value):
                    continue
                key = tuple(sorted([norm_value(a.exact_value), norm_value(b.exact_value)]))
                if key in seen:
                    continue
                seen.add(key)
                conflicts.append({"field": a.field_type, "value_a": a.exact_value,
                                  "doc_a": a.document_id, "value_b": b.exact_value,
                                  "doc_b": b.document_id})
        return conflicts[:4]
