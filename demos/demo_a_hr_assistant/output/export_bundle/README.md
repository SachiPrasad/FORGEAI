# HR Policy Assistant

**Domain:** Human Resources

**Purpose:** Answer HR policy questions (scoped to verified topics: Leave, Attendance, Benefits, Handbook, Hr faq, Salary).

## Tools Enabled
- `document_retriever`: Search ingested documents in the vector knowledge base and return cited excerpts.
- `calculator`: Safely evaluate arithmetic and mathematical expressions (totals, percentages, averages).
- `date_tool`: Get current date/time and compute day differences, business days, or date offsets.
- `verify_topic_scope`: Check whether a user query falls under supported, partially supported, or unsupported topics before generating a response.

## Run
```bash
python run_agent.py -m "Ask your question here"
```
