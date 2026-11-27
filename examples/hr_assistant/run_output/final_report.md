# FORGEAI QUALITY REPORT

**Agent:** HR Policy Assistant
**Domain:** Human Resources
**Target Audience:** Employees
**Purpose:** Answer HR policy questions (scoped to verified topics: Leave, Attendance, Benefits, Rules, Medical insurance, Retirement benefits).
**Documents Analyzed:** 5 (10 chunks)
**Generated Test Cases:** 20

## 1. Input Brief
> Build an HR assistant for employees that can answer questions about leave policies, attendance, benefits and company rules.

## 2. Documents Analyzed
| Document | Type | Chunks | Sections |
| --- | --- | ---: | --- |
| attendance_rules.csv | csv | 3 | Attendance Rules |
| benefits_overview.pdf | pdf | 1 | Benefits Overview |
| company_faq.md | md | 3 | Annual Leave, Public Holidays, Benefits Enrollment |
| hr_handbook.txt | txt | 2 | Code of Conduct & Company Rules, Escalation & Exceptions |
| leave_policy.docx | docx | 1 | Leave Policy |

## 3. Topic Scope & Groundability
- **Supported Topics:** Leave, Attendance, Benefits, Rules
- **Partially Supported Topics:** Medical insurance, Retirement benefits
- **Unsupported Topics (Refused):** Salary, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data

## 4. Selected Tools
- **`document_retriever`** (`knowledge_retrieval`): Knowledge base contains 10 indexed chunk(s) across 5 uploaded document(s) and requires semantic retrieval for grounded answers.
- **`calculator`** (`computation`): Domain and documents include numeric quantities, allowances, limits, or pricing where deterministic arithmetic prevents LLM math errors.
- **`csv_lookup`** (`structured_data`): Uploaded structured CSV data (attendance_rules.csv) is present and supports exact tabular record lookups.
- **`date_tool`** (`temporal`): Policy rules involve timelines, notice periods, holidays, or eligibility windows that benefit from accurate date and business-day calculations.
- **`verify_topic_scope`** (`guardrail`): Unsupported or partially supported topics were detected by Groundability Guard.
- **Excluded Tools:** `http_api` (Excluded because agent operates strictly on internal approved documents without external API calls.), `web_search` (Excluded because closed-book restrictions forbid unapproved external web sources.)

## 5. Retrieval Configuration
- **Strategy:** `recursive`
- **Chunk Size / Overlap:** `600` / `100`
- **Top-K:** `9`
- **Hybrid Keyword Weight:** `0.45`

## 6. Generated System Prompt
```text
You are HR Policy Assistant serving Employees in the Human Resources domain.

Purpose:
Answer HR policy questions (scoped to verified topics: Leave, Attendance, Benefits, Rules, Medical insurance, Retirement benefits).

Your responsibilities:
- Answer questions accurately within your verified scope (Leave, Attendance, Benefits, Rules).
- Use only the approved knowledge base (attendance_rules.csv, benefits_overview.pdf, company_faq.md, hr_handbook.txt, leave_policy.docx).
- Cite the source document (and page/section when available) whenever possible.
- Never invent company policies, numbers, or rules, and do not pretend missing documents exist.
- If the information is unavailable in the approved knowledge base, clearly state that the knowledge base does not contain the answer.
- Do not answer unsupported questions as facts.
- Maintain a professional and friendly tone and deliver responses as conversational answers.

Partially supported topics (answer cautiously with explicit caveats):

- Medical insurance: Only partially covered in benefits_overview.pdf. Partially covered in benefits_overview.pdf, but lacks comprehensive dedicated policy depth. Do not extrapolate missing details.

- Retirement benefits: Only partially covered in benefits_overview.pdf. Partially covered in benefits_overview.pdf, but lacks comprehensive dedicated policy depth. Do not extrapolate missing details.

Unsupported topics (MUST be explicitly refused):

- Salary

- Promotions

- Payroll calculations without payroll data

- Individual employee leave balance lookups without HRIS data

Refusal policy:
If a user asks about Salary, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data, do NOT guess or fabricate an answer. Respond politely that this topic is not covered by the available documents (attendance_rules.csv, benefits_overview.pdf, company_faq.md, hr_handbook.txt, leave_policy.docx), list what you can help with (Leave, Attendance, Benefits, Rules), and direct them to the appropriate team. For partially covered topics (Medical insurance, Retirement benefits), state caveats clearly.

Strict restrictions & guardrails:

- No invented HR policy

- Answer strictly from provided reference documents and state when information is missing

- Do not speculate when a rule is absent from company documents

- Do not invent information or pretend that missing documents exist.

- Ground all factual answers strictly in the verified source documents: attendance_rules.csv, benefits_overview.pdf, company_faq.md, hr_handbook.txt, leave_policy.docx.

- Explicitly refuse requests about unsupported topics (Salary, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data) and inform the user that supporting documentation is unavailable.

- For partially supported topics (Medical insurance, Retirement benefits), share only explicitly documented facts and clearly state any missing details.

- Limit primary topic coverage to verified supported topics: Leave, Attendance, Benefits, Rules.

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
- **INITIAL EVALUATION:** Pass: 16/20 (Score: 80%) | Groundedness: 100%, Correctness: 76%, Refusal: 100%, Sources: 95%
- **ITERATION 1 (prompt_defect):** Pass: 18/20 (Score: 80% -> 90%) | Change: Updated system_prompt with full multi-document synthesis and complete fact-citation instructions (detail_mode: full).
- **ITERATION 2 (retrieval_miss):** Pass: 18/20 (Score: 90% -> 90%) | Change: Increased retrieval_config.top_k from 5 to 9 and hybrid_keyword_weight from 0.2 to 0.45.
- **FINAL STATUS:** Pass: 18/20 (Score: 90%) | Stop reason: No further configuration improvements available.

## 8. Failed Test Cases (Honest Disclosure)
| Test ID | Category | Question | Failure Reasons |
| --- | --- | --- | --- |
| HR_010 | multi_document | Considering both Attendance Rules and Public Holidays, what rules apply to Late Arrival together with holiday schedules are published in? | Answer missed required expected facts: ['02', '30 minutes', 'holiday', 'schedules']. |
| HR_017 | ambiguous | What rules or limits apply to attendance if I have an unusual case? | Answer missed required expected facts: ['01', '10', '02', '30 minutes']. |

## 9. Remaining Weaknesses
- 2 evaluation test case(s) still fail across categories: ambiguous, multi_document.
- Correctness rate is 88% (< 100%).
- Partially supported topics (Medical insurance, Retirement benefits) lack full policy depth in uploaded docs.
- Unsupported topics (Salary, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data) cannot be answered without additional data sources.

## 10. Recommendations
- Improve cross-document retrieval and enable LLM-based multi-document synthesis.
- Add proactive clarification prompts for underspecified user questions.
- Upload complete policy documents for partially supported topics: Medical insurance, Retirement benefits.
- Connect authorized data sources or keep strict refusals enabled for: Salary, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data.
- Configure an API or neural embedding model (e.g., LiteLLM text-embedding-3-small) for production semantic recall.
