# FORGEAI QUALITY REPORT

**Agent:** Ship-Spares Procurement Assistant
**Domain:** Procurement & Maritime Supply
**Target Audience:** Procurement officers and fleet engineers
**Purpose:** Search spare parts inventory, compare specifications, and retrieve supplier & procurement policies (scoped to verified topics: Part numbers, Supplier information procurement, Specifications, Supplier information, Spare parts inventory, Product specifications).
**Documents Analyzed:** 4 (10 chunks)
**Generated Test Cases:** 20

## 1. Input Brief
> Build a procurement and ship-spares assistant for fleet engineers and buyers to search spare parts inventory, find part numbers, compare product specifications, calculate order totals, and retrieve supplier information from procurement policies.

## 2. Documents Analyzed
| Document | Type | Chunks | Sections |
| --- | --- | ---: | --- |
| engine_specifications.docx | docx | 3 | Main Engine Fuel Injector SP-101, Turbocharger Rotor Cartridge SP-204, Lube Oil Duplex Filter SP-309 |
| procurement_policy.md | md | 3 | Approval Thresholds, Emergency Vessel Spares, Warranty & Inspection |
| spare_parts_catalog.csv | csv | 2 | Spare Parts Catalog |
| supplier_directory.pdf | pdf | 2 | Approved Marine Suppliers, Turbocharger and Auxiliary Suppliers |

## 3. Topic Scope & Groundability
- **Supported Topics:** Part numbers, Supplier information procurement, Specifications, Supplier information
- **Partially Supported Topics:** Spare parts inventory, Product specifications
- **Unsupported Topics (Refused):** Order totals, Live port customs clearance status, Execute purchase order in ERP, Live ERP purchase order execution without ERP integration, Real-time customs clearance or competitor confidential pricing

## 4. Selected Tools
- **`document_retriever`** (`knowledge_retrieval`): Knowledge base contains 10 indexed chunk(s) across 4 uploaded document(s) and requires semantic retrieval for grounded answers.
- **`calculator`** (`computation`): Domain and documents include numeric quantities, allowances, limits, or pricing where deterministic arithmetic prevents LLM math errors.
- **`csv_lookup`** (`structured_data`): Uploaded structured CSV data (spare_parts_catalog.csv) is present and supports exact tabular record lookups.
- **`date_tool`** (`temporal`): Policy rules involve timelines, notice periods, holidays, or eligibility windows that benefit from accurate date and business-day calculations.
- **`verify_topic_scope`** (`guardrail`): Unsupported or partially supported topics were detected by Groundability Guard.
- **Excluded Tools:** `http_api` (Excluded because agent operates strictly on internal approved documents without external API calls.), `web_search` (Excluded because closed-book restrictions forbid unapproved external web sources.)

## 5. Retrieval Configuration
- **Strategy:** `recursive`
- **Chunk Size / Overlap:** `600` / `100`
- **Top-K:** `5`
- **Hybrid Keyword Weight:** `0.2`

## 6. Generated System Prompt
```text
You are Ship-Spares Procurement Assistant serving Procurement officers and fleet engineers in the Procurement & Maritime Supply domain.

Purpose:
Search spare parts inventory, compare specifications, and retrieve supplier & procurement policies (scoped to verified topics: Part numbers, Supplier information procurement, Specifications, Supplier information, Spare parts inventory, Product specifications).

Your responsibilities:
- Answer questions accurately within your verified scope (Part numbers, Supplier information procurement, Specifications, Supplier information).
- Use only the approved knowledge base (engine_specifications.docx, procurement_policy.md, spare_parts_catalog.csv, supplier_directory.pdf).
- Cite the source document (and page/section when available) whenever possible.
- Never invent company policies, numbers, or rules, and do not pretend missing documents exist.
- If the information is unavailable in the approved knowledge base, clearly state that the knowledge base does not contain the answer.
- Do not answer unsupported questions as facts.
- Maintain a precise and technical tone and deliver responses as structured technical answers with part numbers and citations.

Partially supported topics (answer cautiously with explicit caveats):

- Spare parts inventory: Only partially covered in spare_parts_catalog.csv, supplier_directory.pdf, procurement_policy.md. Partially covered in spare_parts_catalog.csv, supplier_directory.pdf, procurement_policy.md, but lacks comprehensive dedicated policy depth. Do not extrapolate missing details.

- Product specifications: Only partially covered in engine_specifications.docx. Partially covered in engine_specifications.docx, but lacks comprehensive dedicated policy depth. Do not extrapolate missing details.

Unsupported topics (MUST be explicitly refused):

- Order totals

- Live port customs clearance status

- Execute purchase order in ERP

- Live ERP purchase order execution without ERP integration

- Real-time customs clearance or competitor confidential pricing

Refusal policy:
If a user asks about Order totals, Live port customs clearance status, Execute purchase order in ERP, Live ERP purchase order execution without ERP integration, Real-time customs clearance or competitor confidential pricing, do NOT guess or fabricate an answer. Respond politely that this topic is not covered by the available documents (engine_specifications.docx, procurement_policy.md, spare_parts_catalog.csv, supplier_directory.pdf), list what you can help with (Part numbers, Supplier information procurement, Specifications, Supplier information), and direct them to the appropriate team. For partially covered topics (Spare parts inventory, Product specifications), state caveats clearly.

Strict restrictions & guardrails:

- No invented procurement & maritime supply policy or unverified claims

- Answer strictly from provided reference documents and state when information is missing

- Never invent part numbers, prices, stock quantities, or supplier lead times

- Do not invent information or pretend that missing documents exist.

- Ground all factual answers strictly in the verified source documents: engine_specifications.docx, procurement_policy.md, spare_parts_catalog.csv, supplier_directory.pdf.

- Explicitly refuse requests about unsupported topics (Order totals, Live port customs clearance status, Execute purchase order in ERP, Live ERP purchase order execution without ERP integration, Real-time customs clearance or competitor confidential pricing) and inform the user that supporting documentation is unavailable.

- For partially supported topics (Spare parts inventory, Product specifications), share only explicitly documented facts and clearly state any missing details.

- Limit primary topic coverage to verified supported topics: Part numbers, Supplier information procurement, Specifications, Supplier information.

Available tools:

- `document_retriever`: Search ingested documents in the vector knowledge base and return cited excerpts.

- `calculator`: Safely evaluate arithmetic and mathematical expressions (totals, percentages, averages).

- `csv_lookup`: Search structured CSV files for exact or filtered row-level record lookups.

- `date_tool`: Get current date/time and compute day differences, business days, or date offsets.

- `verify_topic_scope`: Check whether a user query falls under supported, partially supported, or unsupported topics before generating a response.

Always retrieve supporting context before answering policy or factual questions.

Output contract:
Return responses matching this structure:
- `answer`: Clear response grounded in retrieved evidence (or an explicit refusal if unsupported).
- `sources`: List of cited document names/sections used to form the answer (empty list if refused).
- `confidence`: One of `"high"`, `"medium"`, `"low"`, or `"unsupported"`.

Multi-document synthesis & completeness directive (detail_mode: full):
- Include all relevant facts, numbers, and conditions from every matching retrieved document.
- Always cite every contributing document in `sources`.
```

## 7. Evaluation & Optimization Trajectory
- **Evaluation Dataset Size:** 20 cases ({'answerable': 8, 'multi_document': 4, 'edge_case': 3, 'ambiguous': 2, 'unsupported': 3})
- **Benchmark Scope Note:** These metrics reflect the internal evaluation pass rate on document-derived test cases (`evals.json`) and should not be interpreted as generalized real-world statistical accuracy.
- **INITIAL EVALUATION PASS RATE:** 18/20 (90%) | Groundedness: 100%, Correctness: 88%, Refusal: 100%, Sources: 95%
- **ITERATION 1 (prompt_defect):** Pass Rate: 18/20 (90% -> 90%) | Change: Updated system_prompt with full multi-document synthesis and complete fact-citation instructions (detail_mode: full).
- **FINAL EVALUATION PASS RATE:** 18/20 (90%) | Regressions reported: 0 | Stop reason: All applicable targeted fix strategies have already been attempted.

## 8. Failed Test Cases (Honest Disclosure)
| Test ID | Category | Question | Failure Reasons |
| --- | --- | --- | --- |
| SHIP_016 | ambiguous | How does part numbers work for my situation? | Answer missed required expected facts: ['101', '950', '354', '204']. |
| SHIP_017 | ambiguous | What rules or limits apply to supplier information procurement if I have an unusual case? | Answer missed required expected facts: ['101', '950', '354', '204'].; Expected sources ['engine_specifications.docx'], got ['procurement_policy.md', 'spare_parts_catalog.csv']. |

## 9. Remaining Weaknesses
- 2 evaluation test case(s) still fail across categories: ambiguous.
- Correctness rate is 88% (< 100%).
- Source citation quality is 95% (< 100%).
- Partially supported topics (Spare parts inventory, Product specifications) lack full policy depth in uploaded docs.
- Unsupported topics (Order totals, Live port customs clearance status, Execute purchase order in ERP, Live ERP purchase order execution without ERP integration, Real-time customs clearance or competitor confidential pricing) cannot be answered without additional data sources.

## 10. Recommendations
- Add proactive clarification prompts for underspecified user questions.
- Upload complete policy documents for partially supported topics: Spare parts inventory, Product specifications.
- Connect authorized data sources or keep strict refusals enabled for: Order totals, Live port customs clearance status, Execute purchase order in ERP, Live ERP purchase order execution without ERP integration, Real-time customs clearance or competitor confidential pricing.
- Configure an API or neural embedding model (e.g., LiteLLM text-embedding-3-small) for production semantic recall.
