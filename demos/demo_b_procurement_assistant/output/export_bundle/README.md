# Ship-Spares Procurement Assistant

**Domain:** Procurement & Maritime Supply

**Purpose:** Search spare parts inventory, compare specifications, and retrieve supplier & procurement policies (scoped to verified topics: Part numbers, Supplier information procurement, Specifications, Supplier information, Spare parts inventory, Product specifications).

## Tools Enabled
- `document_retriever`: Search ingested documents in the vector knowledge base and return cited excerpts.
- `calculator`: Safely evaluate arithmetic and mathematical expressions (totals, percentages, averages).
- `csv_lookup`: Search structured CSV files for exact or filtered row-level record lookups.
- `date_tool`: Get current date/time and compute day differences, business days, or date offsets.
- `verify_topic_scope`: Check whether a user query falls under supported, partially supported, or unsupported topics before generating a response.

## Run
```bash
python run_agent.py -m "Ask your question here"
```
