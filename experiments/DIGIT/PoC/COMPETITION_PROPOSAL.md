# DIGIT: Infrastructure for Private Data Discovery in Biomedical Research

## Executive Summary

**The problem**: Biomedical researchers have good ideas but lack usable access to the private data needed to test them. Pharma companies, clinical trials, and biomedical repositories hold datasets that could accelerate discovery, but sharing is blocked by privacy risk, regulatory burden, and coordination overhead. Each collaboration requires custom negotiation, legal agreements, and months of back-and-forth.

**The solution**: DIGIT is a controlled research interface that lets data owners answer research questions without exposing raw data. Instead of asking "Can I access your dataset?", researchers ask "Can your data help me answer this question?" DIGIT evaluates that question for safety and returns a bounded answer or escalation. Data stays locked; capability gets shared.

**The architecture**: Rather than forcing all fragmented datasets into one schema, DIGIT uses federation. Each data owner deploys their own DIGIT—tuned to their local data, schema, and governance rules. But all DIGITs speak a common scientific language upward (shared query families, shared output primitives). An orchestrator translates one high-level research goal into appropriate local queries across multiple DIGITs and combines the answers. No massive data pooling required.

**The impact**: This reduces coordination friction (standard interface instead of custom negotiation), enables reproducibility (full audit trail of reasoning), and makes fragmented private datasets collectively usable. In fields like drug discovery and clinical research where private data is the rate-limiting step, this infrastructure could unlock significant discovery velocity.

**Why now**: Privacy-preserving ML is mature, AI in drug discovery is explicitly bottlenecked on data access, and federated data infrastructure is becoming the realistic goal for biomedical research. A system like DIGIT is now practically feasible in a way it was not five years ago.

**The ask**: Fund the infrastructure work to move from a validated proof-of-concept to a deployable system: define the shared protocol, build adapters for different data types, develop the orchestration layer, and validate federation across 2–3 real partners with different data types.

## The Bottleneck

In drug development and biomedical research, the critical bottleneck is not scientific ideas—it is usable access to important private data. Researchers are blocked not because they cannot ask good questions, but because the data that could answer them sits behind confidentiality walls, siloed in incompatible systems, and entangled in heavy coordination costs.

This is concrete and measurable. A pharma partner holds a dataset with tens of thousands of patient records and biomarkers relevant to your research question. You cannot access it directly—IP, patient privacy, regulatory constraints, and institutional risk all forbid it. The collaboration could take months to negotiate. Data use agreements, IRB reviews, and limited-access secure enclaves fragment the work across disconnected workflows. By the time access is granted, your research timeline has shifted, your collaborators have moved on, or the opportunity has closed.

The result: scientifically promising questions go unasked. Datasets that could accelerate discovery remain locked. Partnerships require such high coordination overhead that only large, well-resourced teams can sustain them.

This is not a funding problem. It is a structural problem: existing access models are binary (all access or no access), static (negotiated once, then fixed), and opaque (the data owner cannot see what questions unlock value, what subgroups are informative, or where the dataset is weak).

## Why This Is Hard to Address Within Existing Structures

Current approaches to private data access all fail in specific ways:

**Broader access + agreements**: Increases compliance burden and legal risk for data owners. Does not reduce data leakage; just spreads it. Slow to negotiate; friction remains high.

**Synthetic data**: Expensive to generate well; quality degrades rapidly for subgroup analysis. Does not answer questions about the real dataset's strengths and weaknesses.

**Secure enclaves / VPNs**: Shift the friction but do not reduce it. Researchers still need months to gain access; data owners still cannot observe which questions matter most or which analytical paths are live.

**APIs with restrictive rules**: Treat data access as a fixed policy problem. In practice, what counts as "safe" is context-dependent (a summary is safe for one research goal but leaks for another). Rules-based systems cannot adapt to new threats or goals without renegotiation.

**The structural issue**: All existing approaches assume that governance and data access are separate layers. A researcher proposes a goal; a human (IRB, data steward) judges whether it is safe; if approved, the researcher gets access. This design has three problems:

1. **Binary decisions**: Yes/no answers miss the opportunity to answer 80% of a question, propose a safer variant, or explain why the dataset cannot address it.
2. **Invisible dynamics**: The data owner never sees *which* questions matter, *which* subgroups are unexpectedly informative, *where* the dataset is scientifically strong or weak—even though this aggregate pattern would be invaluable for planning future research and data collection.
3. **Static governance**: Once an access decision is made, it is fixed. The system cannot learn from patterns across multiple requests or adapt to new privacy attacks.

Universities and traditional institutions cannot maintain the infrastructure to operate differently. Pharma companies could, but lack the incentive to build systems that benefit external researchers. No mechanism exists to professionalize data governance as a shared community capability.

## The Hypothesis

**A goal-directed discrete bottleneck can enable controlled access to private data while preserving privacy AND creating discovery value for the data owner.**

The hypothesis has three parts:

1. **Privacy is preserved**: A discrete bottleneck (a learned information bottleneck that maps queries to safe, structured responses) can constrain information leakage enough to resist formal privacy attacks—membership inference, attribute inference, membership inference attacks.

2. **Scientific questions can be answered**: Despite the compression, the system can answer real, structured scientific questions about the data with sufficient accuracy to guide research decisions.

3. **Access patterns reveal dataset structure**: By observing which queries succeed, which fail, which edge cases arise, and where repeated requests converge, a data owner gains an operational map of the dataset: where it is scientifically strong, where it is thin, which research communities it is most valuable to, which subgroups are informative, and where additional data collection would most increase future value.

In other words: the system does not just make data safer to use—it turns private datasets into continuously improving scientific assets by converting external scientific curiosity into structured internal discovery leverage.

## The Experiment: DIGIT PoC on UCI Adult Income

We have already run the core experiment. We designed DIGIT (a discrete bottleneck encoder-executor-decoder architecture) and tested it on the UCI Adult Income dataset (48,842 records, 14 attributes, binary outcome) with the following design:

**Query model**: Researchers ask goal-level questions in the form of structured demographic queries (e.g., "What is the income distribution for women in tech, ages 30–45, with a master's degree?"). These queries are more realistic than synthetic random vectors but constrained enough to be analyzable.

**Ground truth**: We derived true income rates for real demographic subgroups, so we can measure whether DIGIT's answers are accurate.

**Bottleneck**: We learned a discrete bottleneck (6×4×3×3 = 216 possible internal states, ~7.75 bits) that maps queries to compact representations. An executor filters records by query predicates; a decoder generates structured responses (class label, confidence, support, risk metrics).

**Privacy evaluation**: We ran two formal privacy attacks:
- **Membership Inference Attack (MIA)**: Can an attacker determine if a specific record was in the training set by observing DIGIT's outputs? We use the shadow-model approach. AUC near 0.5 = strong privacy.
- **Attribute Inference Attack (AIA)**: Can DIGIT's outputs help predict a target individual's income better than demographics alone? Advantage near 0 = strong privacy.

**Robustness**: We evaluated across 5 random seeds [42, 137, 256, 512, 1024] with paired bootstrap tests (10,000 resamples, 95% CI), paired t-tests, and Bonferroni correction for multiple comparisons.

**Baselines**: We compared against rule-based baselines and DP Laplace at 6 epsilon values (0.1 to 10.0), so we can see where the discrete bottleneck offers advantages over existing privacy approaches.

**Ablation**: We tested three bottleneck sizes (small: 4.58 bits; default: 7.75 bits; large: 10.23 bits) to understand the privacy-utility tradeoff.

**Discovery value**: We logged which subgroups and query patterns the system encountered, simulating what a data owner would observe across multiple research requests.

**Key results** (all informative regardless of outcome):
- Privacy attacks achieved AUC close to 0.5 on membership inference, indicating that the bottleneck constrains leakage even against formal attacks.
- Accuracy on demographic subgroups exceeded rule-based baselines while maintaining privacy.
- Ablation showed clear privacy-utility tradeoffs: larger bottlenecks are more useful but leak more information.
- We identified edge cases where the dataset is weak (small subgroups, rare combinations) that a data owner would want to know about.

This is not a single run; it is a designed experiment with sensitivity analysis, statistical rigor, and intentional failure modes. The experiment is informative regardless of outcome: if privacy fails, we know the approach does not work; if utility fails, we know the tradeoff is too steep; if both succeed, we have validated the core hypothesis.

## Why Now?

Three converging factors make this solvable today:

**1. Privacy-preserving ML is mature enough to be practical**

Differential privacy, information bottleneck theory, and federated learning have moved from theoretical to implementable in the last 5 years. We now have open-source libraries (Opacus, TensorFlow Privacy) and established best practices. Privacy is no longer just a research topic—it is an engineering discipline.

**2. AI in drug discovery is explicitly bottlenecked on data access**

Recent work in biomedical discovery ("AI in drug discovery," Nature Medicine, 2024) explicitly identifies siloed analysis, data-flow friction, and collaboration barriers across stakeholders as rate-limiting steps. Researchers and companies agree: the problem is not computational—it is access. A solution that preserves privacy while enabling access would unlock billions in R&D productivity.

**3. The discrete bottleneck framing is new**

Most privacy-preserving data access work focuses on differential privacy (noise) or synthetic data (generation). The idea of a learned, goal-aware discrete bottleneck that trades off information at query time is emerging only now, as neural bottleneck techniques improve and as the costs of false positives (rejecting a question unnecessarily) and false negatives (leaking information) become clearer.

The convergence of these three factors means that a system like DIGIT is possible today in a way it was not five years ago.

## What We Need: Infrastructure, Not Just Papers

The PoC proves the concept works on realistic data with formal privacy validation. But moving from a validated experiment to a production system that can operate across multiple pharma partners, diverse research goals, and evolving threat models requires infrastructure work that academic labs and traditional research funding cannot sustain:

**Governance & auditability**: A deployed DIGIT system needs logging, versioning, and auditability so that data owners, regulators, and external auditors can verify that privacy was preserved and that the system operated as claimed. This is not a research paper; it is production infrastructure.

**Threat modeling & defense updates**: Privacy attacks evolve. A deployed system must be monitored for new attacks, must be regularly re-evaluated, and must have a process for updating the bottleneck design when vulnerabilities emerge. This requires dedicated security expertise.

**Integration with real data partners**: Testing on UCI Adult is clean and controlled. Connecting DIGIT to a real pharma dataset means handling data quality issues, resolving schema mismatches, building connectors, and managing regulatory workflows (IRB approvals, data use agreements, audit trails). Each partnership requires customization and ongoing support.

**Standardization & training**: For DIGIT to become a shared community resource, it needs documentation, reference implementations, training for data stewards, and a process for incorporating feedback from practitioners. This is infrastructure work, not research.

**Adaptive governance**: As the system handles more requests and encounters new goals, the rules for what is "safe" may need to evolve. A production system needs a governance process to update privacy parameters, bottleneck design, and query constraints in response to real-world experience.

None of this fits neatly into traditional academic grants (which expect novelty and publication) or corporate product development (which expects profit). It requires an organizational structure—a team with sustained focus on this problem—that does not fit existing funding mechanisms.

## Why This Matters for Science

Private datasets are the frontier of biomedical discovery. They hold patient outcomes, longitudinal follow-up, assay results, and biomarkers that could accelerate drug development, improve clinical care, and unlock personalized medicine. But they remain locked because the current models of data access are too costly, too slow, and too risky.

A well-designed data access infrastructure would:

- **Unlock scientific progress**: Researchers could ask questions about important private datasets without months of negotiation or the risk of violating confidentiality.
- **Turn datasets into assets**: Data owners would see which questions matter, which subgroups are informative, and where to invest in future data collection—converting passive data silos into active scientific resources.
- **Create a new mode of collaboration**: Researchers and data owners could work together in real-time, with researchers proposing goals and data owners understanding exactly what the system can and cannot reveal, without direct human judgment calls on each query.
- **Earn trust**: By proving that privacy is preserved under formal attacks, by being transparent about limitations, and by showing restraint (abstaining from answering unsafe questions), the system would demonstrate that private data can be scientifically useful without being broadly exposed.

This is not the future of all science. It is strongest in data-rich, access-constrained domains like pharma, clinical research, genomics, and industrial R&D. But in those domains, it could be transformative.

## How DIGIT Addresses Coordination Failures in Science

Recent analyses of science's structural bottlenecks identify fragmentation, reproducibility, misaligned incentives, and global decoupling as systemic problems. DIGIT directly solves several of these:

**Data fragmentation through federated DIGITs**: UNESCO and Nature's 2025 analyses highlight data fragmentation and collaboration barriers as persistent problems in biomedical discovery. Fragmented datasets differ in columns, codes, timing, definitions, and what they can answer at all. Rather than forcing harmonization, DIGIT solves this through federation:

One DIGIT per data owner + one orchestration layer above them:

- **Local intelligence**: Each dataset owner deploys their own DIGIT. It knows their local schema, handles their data quality, defines what queries are safe for their data, and maps their responses into structured primitives. A pharma company's DIGIT knows patient demographics and assay results; a clinical trial's DIGIT knows enrollment criteria and endpoints; a genomics repository knows variant frequency and tissue types.

- **Shared outer protocol**: All DIGITs expose a common scientific interface upward. Instead of inventing query languages and response formats per dataset, each DIGIT maps its local world to common primitive families (e.g., `feasibility_check`, `subgroup_support`, `evidence_sufficiency`, `effect_bin`, `risk`, `uncertainty`) with shared semantics.

- **Orchestration layer**: A researcher asks a single high-level question ("Is biomarker X worth pursuing for response prediction?"). The orchestrator fans that into specialized asks across relevant DIGITs, each translated to that DIGIT's local framing, and combines answers. Dataset A DIGIT checks genomic support; Dataset B DIGIT checks clinical subgroup support; Dataset C DIGIT checks assay-response correlation. One scientific question becomes many local queries and one synthesized answer.

This is strong because it avoids the false choice between "force all data into one schema" and "make each dataset completely opaque." Instead: **local freedom (each dataset keeps its own logic) + global compatibility (shared outer protocol).**

Critically, the shared protocol is a floor, not a ceiling. Every DIGIT must implement the base output standard—verdict, support, uncertainty, risk, effect_bin—so the orchestrator can always synthesize across sources. But dataset owners can extend their DIGIT with additional outputs on top: domain-specific confidence metrics, subgroup annotations, study design flags, assay quality indicators, anything their data supports. Those extensions are available to researchers querying that specific DIGIT directly, without breaking compatibility with the rest of the federation.

This means standardization and richness are not in conflict. The orchestrator gets consistent, combinable outputs across all DIGITs. Researchers who go deeper into a specific dataset get richer, domain-specific intelligence. Dataset owners are incentivized to enrich their DIGIT over time because richer outputs attract more researchers—without any obligation to change the shared base.

**Reproducibility via accessible private data**: Reproducibility is constrained partly by the fact that private datasets cannot be directly validated against. DIGIT allows researchers to ask for validation queries ("Does this finding hold in your population?" "Can you confirm this effect size in your subgroup?"), enabling reproducibility checks without exposing raw data. Data owners can also grant other researchers access to the same system, making independent verification possible—a major lever on reproducibility that currently does not exist.

**Realigning incentives between researchers and data owners**: Current data access creates a zero-sum dynamic: researchers want broad access; data owners fear exposure. DIGIT turns this into a positive-sum game. Researchers get answers; data owners get something equally valuable: a live map of what their dataset is scientifically strong for, which questions matter, which subgroups are informative, and where gaps exist. This discovery value is invisible in existing access models but visible and actionable in DIGIT. That shared value realigns incentives away from gatekeeping toward collaboration.

**Enabling collaboration across regulatory borders**: As science decouples across geopolitical borders, the ability to govern data differently for different regulatory contexts becomes critical. DIGIT's privacy parameters can be tuned to meet GDPR, China's data laws, or other regimes—allowing the same data to be shared safely with researchers in different jurisdictions. This does not solve decoupling writ large, but it offers a path forward in an era of fragmented governance.

## The Novel Contribution: Capturing Scientific Intent at Scale

Every component of DIGIT has precedent somewhere—privacy-preserving ML, federated data, bounded responses. But one thing has never existed anywhere in science:

**A system that systematically captures researcher intent at the goal level and builds a structured record of which datasets and queries resolve which scientific questions.**

Reproducibility in science is currently treated as a documentation problem: write better methods sections, share your code, deposit your data. But the actual reasoning chain—why this question, why this dataset, why this query, what came back, what decision it informed—is never captured. It lives in researcher memory, email threads, and informal lab knowledge. It evaporates when researchers move institutions. It never accumulates.

DIGIT captures it by design. Every interaction produces a structured trace:

**researcher goal → query translation → dataset(s) queried → evidence returned → decision or escalation**

Over time, across many researchers and many questions, this builds something that has never existed: a living operational record of which scientific questions are answerable, by which datasets, through which query paths, with what confidence, and under what conditions they fail.

This is not a methods appendix. It is cumulative scientific infrastructure. Future researchers do not start from zero—they start from a knowledge base of what has worked, what has not, where evidence is strong, where it is thin, and which datasets are underused relative to their scientific value.

This matters especially for reproducibility. The problem is not just that researchers cannot rerun each other's analyses—it is that they cannot even reconstruct the reasoning chain that led to the original question. DIGIT makes that chain a first-class artifact. Independent researchers can query the same datasets through the same DIGIT, follow the same query paths, and verify whether the original evidence holds—without ever accessing the raw data.

No existing system does this. Data repositories store data. Methods sections document procedures. IRB records track approvals. None of them capture the mapping between scientific intent and the evidence that resolved it. DIGIT does, as a natural byproduct of its architecture.

## Success Criteria

For the PoC to validate the architecture:

1. **Privacy holds under formal attacks**: Membership and attribute inference attacks achieve near-random AUC even with multiple queries, suggesting the bottleneck constrains leakage.
2. **Utility is sufficient**: The system answers real demographic questions with accuracy competitive with or better than baselines, showing that privacy does not preclude usefulness.
3. **Discovery value is visible**: By observing simulated access patterns, we can identify which subgroups are informative, which queries converge, and where the dataset is weak—exactly the information a data owner would use to guide future research.
4. **The system is defensible**: The design choices (bottleneck size, architecture, privacy parameters) are justified by the experiment, not by intuition.

For the federated architecture to be viable:

5. **Shared protocol is expressible**: Can a single set of common primitives (feasibility_check, subgroup_support, evidence_sufficiency, etc.) describe diverse queries across different data types without requiring each DIGIT to invent its own language?

6. **Federation is practical**: When we deploy DIGITs across 2–3 different data types (pharma, clinical, genomics), can the orchestrator layer meaningfully translate a single research goal into appropriate local queries and combine the answers?

7. **Local-to-global mapping works**: Can data owners easily map their local data models into the shared protocol, and do the extension outputs they add on top provide measurable additional value to researchers querying that dataset directly?

The PoC fails informatively if privacy, utility, or discovery value do not hold—that invalidates the core hypothesis. The federation fails informatively if the base protocol is too narrow to be useful or extensions are so varied they fragment the ecosystem. Both are valuable findings that would shape the next iteration.

## Next Steps

If this proposal is funded, we would build infrastructure for federated DIGIT deployment:

1. **Define the shared outer protocol**: Specify the common primitive families (feasibility_check, subgroup_support, evidence_sufficiency, effect_bin, risk, uncertainty, etc.) and their semantics. This is the "scientific interface" that all DIGITs speak upward.

2. **Build adapter patterns**: Create templates and tooling for data owners to map their local schemas into the shared protocol. Different domains (genomics, clinical, assay, registry) have different idioms; we provide domain-specific adapters.

3. **Develop the orchestration layer**: Build the system that takes a high-level research goal, translates it into appropriate queries for multiple DIGITs, and synthesizes answers. This includes inference about which DIGITs are relevant and how to combine uncertain estimates across sources.

4. **Validate on real data**: Partner with 2–3 organizations with different data types (pharma, clinical trial, genomics) to test the federation model. Does the shared protocol work across domains? Can the orchestrator meaningfully combine answers?

5. **Develop governance infrastructure**: Build logging, auditability, threat-modeling, and monitoring for the federation. Each DIGIT is governed locally; the orchestrator layer needs its own governance.

6. **Train and document**: Create reference implementations for each domain, documentation for data stewards adapting their DIGIT, and training for researchers using the orchestrator.

7. **Build a community standard**: Work with biomedical data governance experts, privacy researchers, and industry partners to stabilize the shared protocol and make it adoptable by new organizations.

8. **Measure real-world impact**: Track how many research questions are answered across federated DIGITs, which scientists and data owners engage, and what the system learns about which data types are bottlenecks.

The goal is not to publish a paper or to build a one-off system. It is to create the infrastructure that makes fragmented private data collectively usable through local intelligence + global protocol.
