"""Synthetic benchmark generator for DIGRAG.

Design goal: build a corpus where a grep / exact-string search can *almost
always* find the literal value asked about, yet the literal it finds is often
the wrong one to use.  The correct answer depends on applicability —
recency, approval status, an exception, an alias, a conflicting source, a
dependency chain, a table coordinate, a date role, a policy condition, the
code scope, or disambiguating a high-frequency keyword.

Every scenario is addressed by a UNIQUE subject token (e.g. ``Atlas-014``) that
appears in both the question and the scenario's documents.  This keeps the gold
answer unique even though high-frequency words (``refund``, ``window``,
``days``, region names, code symbols) are deliberately repeated across hundreds
of background documents so that grep's top-k precision degrades the way it does
on a real knowledge base.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

from ..schema import Document, GoldEvidence, Question

REGIONS = ["US", "EU", "UK", "APAC", "CA", "LATAM"]
PLANS = ["free", "pro", "business", "enterprise", "government"]
TEAMS = ["Platform", "Billing", "Security", "Growth", "Data", "Infra"]
PRODUCTS = ["Atlas", "Nimbus", "Quartz", "Beacon", "Harbor", "Pylon", "Vertex", "Cobalt",
            "Onyx", "Marlin", "Cedar", "Falcon", "Lyra", "Comet", "Drift", "Ember"]
SYMBOLS = ["TIMEOUT_MS", "MAX_RETRIES", "BATCH_SIZE", "CACHE_TTL", "POOL_SIZE", "RATE_LIMIT"]
REGION_CODES = ["eu-west-1", "us-east-1", "ap-south-1", "eu-central-1", "us-east-2",
                "ap-southeast-1", "eu-north-1", "sa-east-1"]


def _id(n: int) -> str:
    return f"D{n:04d}"


class Generator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._n = 0
        self._scn = 0
        self.docs: List[Document] = []
        self.questions: List[Question] = []

    # -- helpers ----------------------------------------------------------
    def add_doc(self, title, text, doc_type="policy", timestamp=None,
                status="active", meta=None) -> str:
        self._n += 1
        doc_id = _id(self._n)
        self.docs.append(Document(doc_id=doc_id, title=title, text=text, doc_type=doc_type,
                                  timestamp=timestamp, status=status, meta=meta or {}))
        return doc_id

    def subject(self) -> str:
        """A globally-unique, question-addressable subject token."""
        self._scn += 1
        return f"{self.rng.choice(PRODUCTS)}-{self._scn:03d}"

    def choice(self, xs):
        return self.rng.choice(xs)

    def sample(self, xs, k):
        return self.rng.sample(xs, k)

    def _qid(self, ct, i):
        return f"{ct}-{i:03d}"

    # ================================================================
    def case_stale_current(self, i) -> Question:
        s = self.subject()
        old, new = self.rng.choice([(45, 30), (60, 30), (90, 60), (14, 30), (60, 45)])
        oy, ny = self.rng.choice([2019, 2020, 2021]), self.rng.choice([2024, 2025])
        a = self.add_doc(f"{s} Refund Policy v1",
                         f"Policy {s}-RP. Refund window for {s}: {old} days.\nThis is version 1.",
                         timestamp=f"{oy}-02-01", status="superseded", meta={"subject": s})
        b = self.add_doc(f"{s} Refund Policy v2 (current)",
                         f"Policy {s}-RP. Refund window for {s}: {new} days.\n"
                         f"Version 2 supersedes version 1.",
                         timestamp=f"{ny}-01-15", status="active", meta={"subject": s})
        q = Question(self._qid("stale_current", i), "stale_current",
                     f"What is the current refund window for {s}?",
                     "VALUE", f"{new} days", f"{new} days", True,
                     [GoldEvidence(b, "number", f"{new} days", f"Refund window for {s}: {new} days.")],
                     distractor_value=f"{old} days", distractor_doc_id=a,
                     gold_conflicts=[[a, b]], facets={"subject": s})
        self.questions.append(q); return q

    def case_approved_rejected(self, i) -> Question:
        s = self.subject()
        pid = f"P-{2000 + self._scn}"
        val, unit = self.rng.choice([(60, "day refund window"), (20, "percent partner discount"),
                                     (5, "TB storage cap"), (90, "day data retention")])
        decision = self.choice(["REJECTED", "REJECTED", "WITHDRAWN"])
        a = self.add_doc(f"Proposal {pid} ({s})",
                         f"Proposal {pid} for {s}: change to a {val} {unit}.\n"
                         f"Status: {decision}.",
                         doc_type="memo", status="rejected", meta={"subject": s})
        q = Question(self._qid("approved_rejected", i), "approved_rejected",
                     f"Is the {val} {unit} from proposal {pid} now in effect for {s}?",
                     "NO",
                     f"No. Proposal {pid} was {decision.lower()}, so the {val} {unit} is not in effect.",
                     str(val), True,
                     [GoldEvidence(a, "policy_field", decision, f"Status: {decision}.")],
                     distractor_value=str(val), distractor_doc_id=a, facets={"subject": s, "proposal": pid})
        self.questions.append(q); return q

    def case_exception(self, i) -> Question:
        s = self.subject()
        std = self.rng.choice([30, 45, 60])
        a = self.add_doc(f"{s} Standard Refund Rule",
                         f"{s} standard rule: refunds are not available after {std} days "
                         f"and not available on non-refundable items.",
                         status="active", meta={"subject": s})
        b = self.add_doc(f"{s} EU Statutory Exception",
                         f"For {s}, EU customers retain a 14-day statutory cooling-off right that "
                         f"overrides the standard rule, including on non-refundable items.",
                         status="active", meta={"subject": s, "region": "EU"})
        q = Question(self._qid("exception", i), "exception",
                     f"An EU customer wants to return a non-refundable {s} item within 10 days. "
                     f"Are they entitled to a refund?",
                     "YES",
                     "Yes. The EU 14-day statutory cooling-off right overrides the standard "
                     "non-refundable rule, and 10 days is within it.",
                     "14-day statutory", True,
                     [GoldEvidence(b, "clause", "14-day statutory",
                                   f"For {s}, EU customers retain a 14-day statutory cooling-off right "
                                   f"that overrides the standard rule, including on non-refundable items.")],
                     distractor_value="non-refundable", distractor_doc_id=a,
                     facets={"subject": s, "region": "EU", "purchase": "non-refundable"})
        self.questions.append(q); return q

    def case_alias(self, i) -> Question:
        s = self.subject()                          # e.g. Atlas-014
        base = s.split("-")[0]
        code = f"{base[:3].upper()}-{self._scn}"      # ATL-14
        full = f"{s} Cloud Platform"
        sla = self.rng.choice(["99.9%", "99.95%", "99.5%", "99.99%"])
        a = self.add_doc(f"{s} SLA",
                         f"Project {s} SLA: {sla} monthly uptime.",
                         status="active", meta={"subject": s})
        b = self.add_doc(f"Codename Registry: {code}",
                         f"{code} is the internal code for {full} (a.k.a. Project {s}).",
                         doc_type="memo", status="active", meta={"subject": s})
        q = Question(self._qid("alias", i), "alias",
                     f"What is the uptime SLA for {code}?",
                     "VALUE", f"{sla} monthly uptime", sla, True,
                     [GoldEvidence(b, "id", code, f"{code} is the internal code for {full}"),
                      GoldEvidence(a, "number", sla, f"Project {s} SLA: {sla} monthly uptime.")],
                     distractor_value=code, distractor_doc_id=b, facets={"subject": s, "alias": code})
        self.questions.append(q); return q

    def case_conflict(self, i) -> Question:
        s = self.subject()
        topic = self.choice(["data retention", "log retention", "session timeout", "backup retention"])
        t1, t2 = self.rng.sample([30, 60, 90, 120, 180, 365], 2)
        team1, team2 = self.rng.sample(TEAMS, 2)
        a = self.add_doc(f"{team1} policy for {s}",
                         f"{team1} policy: {s} {topic} period is {t1} days.",
                         status="active", timestamp="2025-01-01", meta={"subject": s})
        b = self.add_doc(f"{team2} policy for {s}",
                         f"{team2} policy: {s} {topic} period is {t2} days.",
                         status="active", timestamp="2025-01-01", meta={"subject": s})
        q = Question(self._qid("conflict", i), "conflict",
                     f"What is the company-wide {topic} period for {s}?",
                     "INSUFFICIENT",
                     f"The sources conflict: {team1} states {t1} days while {team2} states {t2} days, "
                     f"both active. A single value cannot be determined.",
                     "", False,
                     [GoldEvidence(a, "number", f"{t1} days", f"{s} {topic} period is {t1} days."),
                      GoldEvidence(b, "number", f"{t2} days", f"{s} {topic} period is {t2} days.")],
                     distractor_value=f"{t1} days", distractor_doc_id=a,
                     gold_conflicts=[[a, b]], facets={"subject": s, "topic": topic})
        self.questions.append(q); return q

    def case_multi_doc(self, i) -> Question:
        s = self.subject()
        primary, backup, dprimary, dbackup = self.rng.sample(REGION_CODES, 4)
        a = self.add_doc(f"{s} Region Defaults",
                         f"The default region for {s} accounts is {primary}.",
                         status="active", meta={"subject": s})
        b = self.add_doc(f"{s} Backup Topology",
                         f"For {s}: backups in {primary} are stored in {backup}.\n"
                         f"For {s}: backups in {dprimary} are stored in {dbackup}.",
                         status="active", meta={"subject": s})
        q = Question(self._qid("multi_doc", i), "multi_doc",
                     f"Where are backups physically stored for a {s} account?",
                     "VALUE", backup, backup, True,
                     [GoldEvidence(a, "policy_field", primary, f"The default region for {s} accounts is {primary}."),
                      GoldEvidence(b, "policy_field", backup, f"For {s}: backups in {primary} are stored in {backup}.")],
                     distractor_value=dbackup, distractor_doc_id=b, facets={"subject": s})
        self.questions.append(q); return q

    def case_table(self, i) -> Question:
        s = self.subject()
        plans = self.sample(PLANS, 3)
        regions = self.sample(REGIONS, 3)
        prices = {(p, r): self.rng.choice([9, 19, 29, 49, 99, 199, 299, 499])
                  for p in plans for r in regions}
        tgt_plan, tgt_region = plans[1], regions[1]
        tgt_price = prices[(tgt_plan, tgt_region)]
        collide = (plans[2], regions[0])
        if collide != (tgt_plan, tgt_region):
            prices[collide] = tgt_price                # make the literal ambiguous
        header = "| plan | " + " | ".join(regions) + " |"
        sep = "|" + "---|" * (len(regions) + 1)
        rows = ["| " + p + " | " + " | ".join(f"${prices[(p,r)]}" for r in regions) + " |" for p in plans]
        table = "\n".join([f"{s} monthly price table (USD):", header, sep, *rows])
        a = self.add_doc(f"{s} Pricing Matrix", table, doc_type="table", status="active",
                         meta={"subject": s})
        q = Question(self._qid("table", i), "table",
                     f"In the {s} pricing matrix, what is the monthly price of the {tgt_plan} plan in {tgt_region}?",
                     "VALUE", f"${tgt_price}", f"${tgt_price}", True,
                     [GoldEvidence(a, "table_cell", f"${tgt_price}", rows[plans.index(tgt_plan)])],
                     distractor_value=f"${prices[collide]}", distractor_doc_id=a,
                     facets={"subject": s, "plan": tgt_plan, "region": tgt_region})
        self.questions.append(q); return q

    def case_date(self, i) -> Question:
        cid = f"C-{100 + self._scn}"
        s = cid
        signed = f"2023-{self.rng.randint(1,9):02d}-02"
        effective = f"2024-{self.rng.randint(1,6):02d}-01"
        renewal = f"2025-{self.rng.randint(1,9):02d}-01"
        a = self.add_doc(f"Contract {cid}",
                         f"Contract {cid}. Signed {signed}. Effective {effective}. Auto-renews {renewal}.",
                         doc_type="contract", status="active", meta={"subject": cid})
        q = Question(self._qid("date", i), "date",
                     f"On what date does contract {cid} auto-renew?",
                     "VALUE", renewal, renewal, True,
                     [GoldEvidence(a, "date", renewal, f"Auto-renews {renewal}.")],
                     distractor_value=effective, distractor_doc_id=a, facets={"subject": cid})
        self.questions.append(q); return q

    def case_policy_condition(self, i) -> Question:
        s = self.subject()
        threshold = self.rng.choice([200, 500, 1000])
        amount = threshold + self.rng.choice([100, 250, 500])
        a = self.add_doc(f"{s} Refund Approval Policy",
                         f"For {s}: refunds under ${threshold} are auto-approved.\n"
                         f"For {s}: refunds of ${threshold} or more require manager approval.",
                         status="active", meta={"subject": s})
        q = Question(self._qid("policy_condition", i), "policy_condition",
                     f"Under the {s} policy, can a ${amount} refund be issued without manager approval?",
                     "NO",
                     f"No. ${amount} is at or above the ${threshold} threshold, so manager approval is required.",
                     str(threshold), True,
                     [GoldEvidence(a, "policy_field", f"${threshold} or more",
                                   f"For {s}: refunds of ${threshold} or more require manager approval.")],
                     distractor_value="auto-approved", distractor_doc_id=a,
                     facets={"subject": s, "amount": str(amount), "threshold": str(threshold)})
        self.questions.append(q); return q

    def case_code_symbol(self, i) -> Question:
        s = self.subject()
        svc = s.lower()
        sym = self.choice(SYMBOLS)
        prod_v = self.rng.choice([30000, 60000, 16, 64, 128, 300])
        test_v = self.rng.choice([1, 2, 5, 7])
        a = self.add_doc(f"{svc}/config/prod.py",
                         f"# {s} production configuration\n{sym} = {prod_v}\nDEBUG = False",
                         doc_type="code", status="active", meta={"subject": s, "scope": "prod"})
        b = self.add_doc(f"{svc}/tests/fixtures.py",
                         f"# {s} test fixtures (do not use in prod)\n{sym} = {test_v}\nDEBUG = True",
                         doc_type="code", status="active", meta={"subject": s, "scope": "test"})
        q = Question(self._qid("code_symbol", i), "code_symbol",
                     f"What is the production value of {sym} in the {s} service?",
                     "VALUE", str(prod_v), str(prod_v), True,
                     [GoldEvidence(a, "code_symbol", str(prod_v), f"{sym} = {prod_v}")],
                     distractor_value=str(test_v), distractor_doc_id=b,
                     facets={"subject": s, "symbol": sym, "scope": "prod"})
        self.questions.append(q); return q

    def case_ambiguous_keyword(self, i) -> Question:
        s = self.subject()
        budget = f"${self.rng.choice([80, 120, 150, 200, 250])}k"
        version = f"{self.rng.randint(2,5)}.{self.rng.randint(0,9)}"
        a = self.add_doc(f"{s} Campaign Brief",
                         f"The {s} marketing campaign has an approved budget of {budget} for Q3.",
                         doc_type="memo", status="active", meta={"subject": s, "sense": "campaign"})
        self.add_doc(f"{s} Engine Release Notes",
                     f"The {s} engine ships in version {version} with performance fixes.",
                     doc_type="memo", status="active", meta={"subject": s, "sense": "product"})
        self.add_doc(f"{s} Onboarding FAQ",
                     f"New users often ask about {s}. The {s} dashboard loads in under 2 seconds.",
                     doc_type="faq", status="active", meta={"subject": s, "sense": "product"})
        q = Question(self._qid("ambiguous_keyword", i), "ambiguous_keyword",
                     f"What is the approved budget for the {s} marketing campaign?",
                     "VALUE", budget, budget, True,
                     [GoldEvidence(a, "number", budget,
                                   f"The {s} marketing campaign has an approved budget of {budget} for Q3.")],
                     distractor_value=version, distractor_doc_id="", facets={"subject": s, "sense": "campaign"})
        self.questions.append(q); return q

    def case_unanswerable(self, i) -> Question:
        s = self.subject()                 # subject that has NO matching doc
        present = self.subject()           # a different subject that DOES
        self.add_doc(f"{present} Refund Policy",
                     f"Refund window for {present}: {self.rng.choice([14,30,45])} days.",
                     status="active", meta={"subject": present})
        topic = self.choice(["refund window", "data retention period", "support SLA"])
        q = Question(self._qid("unanswerable", i), "unanswerable",
                     f"What is the {topic} for {s}?",
                     "INSUFFICIENT",
                     f"The corpus does not specify the {topic} for {s}; the evidence is missing.",
                     "", False, [], distractor_value=present, distractor_doc_id="",
                     facets={"subject": s})
        self.questions.append(q); return q

    # ----------------------------------------------------------------
    def add_background_noise(self, n) -> None:
        templates = [
            "Refund requests are processed within {d} business days by the {team} team.",
            "The dashboard refresh interval is {d} seconds for {plan} plans.",
            "{region} customers can contact support about refund policy and retention questions.",
            "Data retention and backup policy reviews happen every {d} months.",
            "The analytics engine version {v} improves campaign and budget reporting.",
            "Session timeout defaults are documented per region; {region} uses standard values.",
            "Proposal review boards meet to discuss refund window and discount changes.",
            "Pricing for the {plan} plan varies by region; see the official matrix.",
            "{sym} should be tuned per environment; consult the runbook before changing it.",
        ]
        for _ in range(n):
            t = self.choice(templates).format(
                d=self.rng.randint(2, 365), team=self.choice(TEAMS), plan=self.choice(PLANS),
                region=self.choice(REGIONS), v=f"{self.rng.randint(1,9)}.{self.rng.randint(0,9)}",
                sym=self.choice(SYMBOLS))
            self.add_doc("Background Note", t, doc_type="memo", status="active")

    def build(self, n_scenarios, distractor_docs) -> Tuple[List[Document], List[Question]]:
        builders = [self.case_stale_current, self.case_approved_rejected, self.case_exception,
                    self.case_alias, self.case_conflict, self.case_multi_doc, self.case_table,
                    self.case_date, self.case_policy_condition, self.case_code_symbol,
                    self.case_ambiguous_keyword, self.case_unanswerable]
        per = max(1, n_scenarios // len(builders))
        counters = {b.__name__: 0 for b in builders}
        for _ in range(per):
            for b in builders:
                counters[b.__name__] += 1
                b(counters[b.__name__])
        self.add_background_noise(distractor_docs)
        return self.docs, self.questions


def generate(out_dir, seed, n_scenarios, distractor_docs) -> Dict:
    g = Generator(seed=seed)
    docs, questions = g.build(n_scenarios, distractor_docs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")
    with open(os.path.join(out_dir, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q)) + "\n")
    stats = {"n_docs": len(docs), "n_questions": len(questions), "by_case_type": {}}
    for q in questions:
        stats["by_case_type"][q.case_type] = stats["by_case_type"].get(q.case_type, 0) + 1
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", type=int, default=220)
    ap.add_argument("--distractors", type=int, default=400)
    a = ap.parse_args()
    print(json.dumps(generate(a.out, a.seed, a.scenarios, a.distractors), indent=2))
