# FORGEAI QUALITY REPORT

**Agent:** HR Policy Assistant
**Domain:** Human Resources
**Target Audience:** Employees
**Purpose:** Answer HR policy questions (scoped to verified topics: Leave, Attendance, Benefits, Handbook, Hr faq, Salary).
**Documents Analyzed:** 4 (9 chunks)
**Generated Test Cases:** 20

## 1. Input Brief
> Build an HR assistant for employees that can answer questions about leave policies, attendance, benefits, and company rules from the HR FAQ and handbook.

## 2. Documents Analyzed
| Document | Type | Chunks | Sections |
| --- | --- | ---: | --- |
| attendance_policy.txt | txt | 2 | Core Working Hours, Late Arrival & Remote Work |
| employee_handbook.pdf | pdf | 2 | Workplace Conduct and Ethics, Escalation and HR Support |
| hr_faq.md | md | 3 | Annual Leave Requests, Public Holidays, Benefits & Wellness |
| leave_policy.docx | docx | 2 | Annual Leave Policy, Parental Leave |

## 3. Topic Scope & Groundability
- **Supported Topics:** Leave, Attendance, Benefits, Handbook
- **Partially Supported Topics:** Hr faq, Salary
- **Unsupported Topics (Refused):** Personal leave balance, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data

## 4. Selected Tools
- **`document_retriever`** (`knowledge_retrieval`): Knowledge base contains 9 indexed chunk(s) across 4 uploaded document(s) and requires semantic retrieval for grounded answers.
- **`calculator`** (`computation`): Domain and documents include numeric quantities, allowances, limits, or pricing where deterministic arithmetic prevents LLM math errors.
- **`date_tool`** (`temporal`): Policy rules involve timelines, notice periods, holidays, or eligibility windows that benefit from accurate date and business-day calculations.
- **`verify_topic_scope`** (`guardrail`): Unsupported or partially supported topics were detected by Groundability Guard.
- **Excluded Tools:** `csv_lookup` (No CSV tables or catalogs were uploaded in the document set.), `http_api` (Excluded because agent operates strictly on internal approved documents without external API calls.), `web_search` (Excluded because closed-book restrictions forbid unapproved external web sources.)

## 5. Retrieval Configuration
- **Strategy:** `recursive`
- **Chunk Size / Overlap:** `600` / `100`
- **Top-K:** `9`
- **Hybrid Keyword Weight:** `0.45`

## 6. Generated System Prompt
```text
You are HR Policy Assistant serving Employees in the Human Resources domain.

Purpose:
Answer HR policy questions (scoped to verified topics: Leave, Attendance, Benefits, Handbook, Hr faq, Salary).

Your responsibilities:
- Answer questions accurately within your verified scope (Leave, Attendance, Benefits, Handbook).
- Use only the approved knowledge base (attendance_policy.txt, employee_handbook.pdf, hr_faq.md, leave_policy.docx).
- Cite the source document (and page/section when available) whenever possible.
- Never invent company policies, numbers, or rules, and do not pretend missing documents exist.
- If the information is unavailable in the approved knowledge base, clearly state that the knowledge base does not contain the answer.
- Do not answer unsupported questions as facts.
- Maintain a professional and friendly tone and deliver responses as conversational answers.

Partially supported topics (answer cautiously with explicit caveats):

- Hr faq: Only partially covered in employee_handbook.pdf, hr_faq.md. Partially covered in employee_handbook.pdf, hr_faq.md, but lacks comprehensive dedicated policy depth. Do not extrapolate missing details.

- Salary: Only partially covered in employee_handbook.pdf. Partially covered in employee_handbook.pdf, but lacks comprehensive dedicated policy depth. Do not extrapolate missing details.

Unsupported topics (MUST be explicitly refused):

- Personal leave balance

- Promotions

- Payroll calculations without payroll data

- Individual employee leave balance lookups without HRIS data

Refusal policy:
If a user asks about Personal leave balance, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data, do NOT guess or fabricate an answer. Respond politely that this topic is not covered by the available documents (attendance_policy.txt, employee_handbook.pdf, hr_faq.md, leave_policy.docx), list what you can help with (Leave, Attendance, Benefits, Handbook), and direct them to the appropriate team. For partially covered topics (Hr faq, Salary), state caveats clearly.

Strict restrictions & guardrails:

- No invented HR policy

- Answer strictly from provided reference documents and state when information is missing

- Refuse salary, personal leave balance, and individual promotion queries without HRIS data

- Do not invent information or pretend that missing documents exist.

- Ground all factual answers strictly in the verified source documents: attendance_policy.txt, employee_handbook.pdf, hr_faq.md, leave_policy.docx.

- Explicitly refuse requests about unsupported topics (Personal leave balance, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data) and inform the user that supporting documentation is unavailable.

- For partially supported topics (Hr faq, Salary), share only explicitly documented facts and clearly state any missing details.

- Limit primary topic coverage to verified supported topics: Leave, Attendance, Benefits, Handbook.

Available tools:

- `document_retriever`: Search ingested documents in the vector knowledge base and return cited excerpts.

- `calculator`: Safely evaluate arithmetic and mathematical expressions (totals, percentages, averages).

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
- **ITERATION 1 (retrieval_miss):** Pass Rate: 19/20 (90% -> 95%) | Change: Increased retrieval_config.top_k from 5 to 9 and hybrid_keyword_weight from 0.2 to 0.45.
- **ITERATION 2 (prompt_defect):** Pass Rate: 19/20 (95% -> 95%) | Change: Updated system_prompt with full multi-document synthesis and complete fact-citation instructions (detail_mode: full).
- **FINAL EVALUATION PASS RATE:** 19/20 (95%) | Regressions reported: 0 | Stop reason: No further configuration improvements available.

## 8. Failed Test Cases (Honest Disclosure)
| Test ID | Category | Question | Failure Reasons |
| --- | --- | --- | --- |
| HR_016 | ambiguous | How does leave work for my situation? | Answer missed required expected facts: ['5 business days', 'submit', '20 days', '10']. |

## 9. Remaining Weaknesses
- 1 evaluation test case(s) still fail across categories: ambiguous.
- Correctness rate is 94% (< 100%).
- Partially supported topics (Hr faq, Salary) lack full policy depth in uploaded docs.
- Unsupported topics (Personal leave balance, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data) cannot be answered without additional data sources.

## 10. Recommendations
- Add proactive clarification prompts for underspecified user questions.
- Upload complete policy documents for partially supported topics: Hr faq, Salary.
- Connect authorized data sources or keep strict refusals enabled for: Personal leave balance, Promotions, Payroll calculations without payroll data, Individual employee leave balance lookups without HRIS data.
- Configure an API or neural embedding model (e.g., LiteLLM text-embedding-3-small) for production semantic recall.
