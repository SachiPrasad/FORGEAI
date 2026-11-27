"""Visual 7-Tab HTML Evaluation Dashboard generator -> dashboard.html."""

from __future__ import annotations

import html
from pathlib import Path

from forgeai.eval.schemas import EvalDataset, EvaluationReport
from forgeai.generator.schemas import AgentDefinition
from forgeai.groundability.schemas import GroundabilityReport
from forgeai.intake.schemas import AgentRequirements
from forgeai.knowledge.schemas import KnowledgeManifest
from forgeai.observability.tracer import TraceSummary
from forgeai.optimizer.schemas import OptimizationLog


class VisualDashboardGenerator:
    """Render a self-contained 7-tab HTML evaluation dashboard matching ForgeAI's UI structure."""

    def generate(
        self,
        agent_def: AgentDefinition,
        groundability_report: GroundabilityReport,
        initial_report: EvaluationReport,
        final_report: EvaluationReport,
        optimization_log: OptimizationLog,
        requirements: AgentRequirements | None = None,
        manifest: KnowledgeManifest | None = None,
        eval_dataset: EvalDataset | None = None,
        trace_summary: TraceSummary | None = None,
        version_diff: dict | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        sc = final_report.scorecard
        init_sc = initial_report.scorecard

        iter_bars = self._render_iteration_bars(init_sc.pass_rate, optimization_log)
        iter_cards = self._render_iteration_cards(init_sc.pass_rate, init_sc.passed_cases, init_sc.total_cases, optimization_log)
        total_regressions = sum(len(rec.regressed_tests) for rec in optimization_log.iterations)
        what_changed_items = "".join(
            f"<li><strong>Iteration {rec.iteration} ({html.escape(rec.failure_type)}):</strong> {html.escape(rec.change_applied)}</li>"
            for rec in optimization_log.iterations
        ) or "<li>No configuration changes required.</li>"
        diff_block = html.escape(
            (version_diff or {}).get("config_diff", "No configuration parameter diff recorded.")
        )
        case_rows = self._render_case_rows(final_report)
        tool_rows = "".join(
            f"<tr><td><code>{html.escape(t.name)}</code></td><td>{html.escape(t.type)}</td><td>{html.escape(t.selection_reason)}</td></tr>"
            for t in agent_def.tools
        )
        chunk_rows = (
            "".join(
                f"<tr><td><code>{html.escape(c.chunk_id)}</code></td><td>{html.escape(c.document_name)}</td>"
                f"<td>{html.escape(str(c.page_number or '-'))}</td><td>{html.escape(c.section or '-')}</td>"
                f"<td>{html.escape(c.content[:140])}...</td></tr>"
                for c in manifest.chunks[:25]
            )
            if manifest
            else ""
        )
        clarification_items = (
            "".join(f"<li><strong>{html.escape(q.id)}</strong>: {html.escape(q.question)}</li>" for q in requirements.clarification_questions)
            if requirements
            else ""
        )
        cost_html = (
            f"{trace_summary.total_duration_ms:.0f} ms | "
            f"{trace_summary.total_input_tokens + trace_summary.total_output_tokens} tok | "
            f"${trace_summary.total_estimated_cost_usd:.5f}"
            if trace_summary
            else "N/A"
        )
        failed_rows = "".join(
            f"<tr><td><code>{html.escape(r.test_id)}</code></td><td>{html.escape(r.category)}</td>"
            f"<td>{html.escape(r.question)}</td><td>{html.escape('; '.join(r.failure_reasons))}</td></tr>"
            for r in final_report.results
            if not r.passed
        ) or '<tr><td colspan="4">All test cases passed.</td></tr>'

        doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ForgeAI — Autonomous Agent Factory Dashboard</title>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --muted:#94a3b8; --text:#f8fafc; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; --accent:#38bdf8; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); margin:0; padding:24px; }}
  .container {{ max-width:1200px; margin:0 auto; }}
  h1 {{ margin:0 0 4px; font-size:28px; }}
  .sub {{ color:var(--muted); margin-bottom:18px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(4, minmax(160px, 1fr)); gap:14px; margin-bottom:18px; }}
  .card {{ background:var(--card); border-radius:12px; padding:16px; border:1px solid #334155; margin-bottom:16px; }}
  .metric-label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.05em; }}
  .metric-val {{ font-size:28px; font-weight:700; margin-top:6px; }}
  .tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }}
  .tab-btn {{ background:#1e293b; color:#cbd5e1; border:1px solid #334155; padding:8px 12px; border-radius:8px; cursor:pointer; font-size:13px; font-weight:600; }}
  .tab-btn.active {{ background:var(--accent); color:#0f172a; border-color:var(--accent); }}
  .tab-pane {{ display:none; }}
  .tab-pane.active {{ display:block; }}
  .badge {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:12px; font-weight:600; }}
  .badge-ok {{ background:rgba(34,197,94,.16); color:#4ade80; }}
  .badge-fail {{ background:rgba(239,68,68,.18); color:#f87171; }}
  .bar-row {{ display:flex; align-items:center; gap:12px; margin:8px 0; }}
  .bar-label {{ width:170px; font-size:13px; color:#cbd5e1; }}
  .bar-track {{ flex:1; background:#0f172a; height:18px; border-radius:9px; overflow:hidden; border:1px solid #334155; }}
  .bar-fill {{ height:100%; background:linear-gradient(90deg, var(--accent), var(--ok)); }}
  pre {{ background:#0f172a; padding:14px; border-radius:8px; overflow-x:auto; font-size:12px; border:1px solid #334155; white-space:pre-wrap; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:10px; border-bottom:1px solid #334155; vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; }}
</style>
</head>
<body>
<div class="container">
  <h1>ForgeAI &middot; Autonomous Agent Factory</h1>
  <div class="sub">{html.escape(agent_def.agent_name)} ({html.escape(agent_def.domain)}) | Trace: {html.escape(cost_html)}</div>

  <div class="kpi-grid">
    <div class="card"><div class="metric-label">Initial eval pass rate</div><div class="metric-val">{init_sc.pass_rate:.0%}</div><div class="sub">{init_sc.passed_cases}/{init_sc.total_cases} internal cases</div></div>
    <div class="card"><div class="metric-label">Final eval pass rate</div><div class="metric-val" style="color:var(--ok)">{sc.pass_rate:.0%}</div><div class="sub">{sc.passed_cases}/{sc.total_cases} internal cases</div></div>
    <div class="card"><div class="metric-label">Test cases</div><div class="metric-val">{sc.total_cases}</div><div class="sub">Doc-derived evals.json</div></div>
    <div class="card"><div class="metric-label">Optimization rounds</div><div class="metric-val" style="color:var(--accent)">{optimization_log.total_iterations_run} / {optimization_log.max_iterations}</div><div class="sub">Regressions: {total_regressions}</div></div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="showTab(0)">01 &middot; Project Intake</button>
    <button class="tab-btn" onclick="showTab(1)">02 &middot; Knowledge Base</button>
    <button class="tab-btn" onclick="showTab(2)">03 &middot; Agent Configuration</button>
    <button class="tab-btn" onclick="showTab(3)">04 &middot; Evaluation</button>
    <button class="tab-btn" onclick="showTab(4)">05 &middot; Optimization</button>
    <button class="tab-btn" onclick="showTab(5)">06 &middot; Final Report</button>
    <button class="tab-btn" onclick="showTab(6)">07 &middot; Test Your Agent</button>
  </div>

  <div class="tab-pane active card" id="tab-0">
    <h3>01 &middot; Project Intake</h3>
    <p><strong>Agent Name:</strong> {html.escape(agent_def.agent_name)} | <strong>Audience:</strong> {html.escape(agent_def.target_audience)} | <strong>Purpose:</strong> {html.escape(agent_def.purpose)}</p>
    <p><strong>Supported Topics:</strong> {html.escape(', '.join(agent_def.supported_topics))}</p>
    <p><strong>Unsupported Topics:</strong> {html.escape(', '.join(agent_def.unsupported_topics))}</p>
    <h4>Clarification Questions (max 5)</h4>
    <ul>{clarification_items or '<li>None required.</li>'}</ul>
  </div>

  <div class="tab-pane card" id="tab-1">
    <h3>02 &middot; Knowledge Base &amp; Groundability ({groundability_report.groundability_score:.0%})</h3>
    <p><strong>Sources:</strong> {html.escape(', '.join(agent_def.allowed_sources))}</p>
    <table><thead><tr><th>Chunk ID</th><th>Document</th><th>Page</th><th>Section</th><th>Preview</th></tr></thead><tbody>{chunk_rows}</tbody></table>
  </div>

  <div class="tab-pane card" id="tab-2">
    <h3>03 &middot; Agent Configuration</h3>
    <p><strong>Model:</strong> <code>{html.escape(agent_def.model)}</code> | <strong>Retrieval:</strong> strategy={html.escape(agent_def.retrieval_config.strategy)}, chunk_size={agent_def.retrieval_config.chunk_size}, overlap={agent_def.retrieval_config.chunk_overlap}, top_k={agent_def.retrieval_config.top_k}</p>
    <table><thead><tr><th>Tool</th><th>Type</th><th>Selection Reason</th></tr></thead><tbody>{tool_rows}</tbody></table>
    <h4>Generated System Prompt</h4>
    <pre>{html.escape(agent_def.system_prompt)}</pre>
  </div>

  <div class="tab-pane card" id="tab-3">
    <h3>04 &middot; Evaluation ({sc.passed_cases}/{sc.total_cases} Passed)</h3>
    <p>Groundedness: <strong>{sc.groundedness_rate:.0%}</strong> | Correctness: <strong>{sc.correctness_rate:.0%}</strong> | Refusal: <strong>{sc.refusal_accuracy_rate:.0%}</strong> | Source Quality: <strong>{sc.source_quality_rate:.0%}</strong></p>
    <table><thead><tr><th>Test ID</th><th>Category</th><th>Question</th><th>Grounded</th><th>Correct</th><th>Refusal</th><th>Sources</th><th>Result</th></tr></thead><tbody>{case_rows}</tbody></table>
  </div>

  <div class="tab-pane card" id="tab-4">
    <h3>05 &middot; Critic&ndash;Optimizer Results &amp; Config Diff</h3>
    <div class="kpi-grid">{iter_cards}</div>
    {iter_bars}
    <h4>What Changed</h4>
    <ul>
      {what_changed_items}
      <li><strong>Regression count:</strong> {total_regressions} reported</li>
      <li><strong>Final remaining failures:</strong> {sc.total_cases - sc.passed_cases} (disclosed in Final Report)</li>
    </ul>
    <h4>Exact Configuration Diff (v0 &rarr; v{optimization_log.total_iterations_run})</h4>
    <pre>{diff_block}</pre>
  </div>

  <div class="tab-pane card" id="tab-5">
    <h3>06 &middot; Final Report (Remaining Failures &amp; Artifacts)</h3>
    <table><thead><tr><th>Test ID</th><th>Category</th><th>Question</th><th>Failure Reasons</th></tr></thead><tbody>{failed_rows}</tbody></table>
  </div>

  <div class="tab-pane card" id="tab-6">
    <h3>07 &middot; Test Your Agent</h3>
    <p>Launch the interactive live chat UI with: <code>streamlit run app.py</code> or CLI: <code>python -m forgeai.runtime.cli --agent agent.json --no-llm</code></p>
  </div>
</div>
<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
  document.querySelectorAll('.tab-pane').forEach((p, i) => p.classList.toggle('active', i === idx));
}}
</script>
</body>
</html>
"""
        if output_path is not None:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(doc_html, encoding="utf-8")
        return doc_html

    @staticmethod
    def _render_iteration_cards(
        initial_rate: float,
        initial_passed: int,
        total_cases: int,
        opt_log: OptimizationLog,
    ) -> str:
        cards = [
            f'<div class="card"><div class="metric-label">Initial</div>'
            f'<div class="metric-val">{initial_rate:.0%}</div>'
            f'<div class="sub">{initial_passed}/{total_cases} tests</div></div>'
        ]
        for rec in opt_log.iterations:
            cards.append(
                f'<div class="card"><div class="metric-label">Iteration {rec.iteration}</div>'
                f'<div class="metric-val" style="color:var(--ok)">{rec.score_after:.0%}</div>'
                f'<div class="sub">{rec.passed_after}/{total_cases} tests</div></div>'
            )
        return "\n".join(cards)

    @staticmethod
    def _render_iteration_bars(initial_rate: float, opt_log: OptimizationLog) -> str:
        bars = [
            f'<div class="bar-row"><div class="bar-label">Iteration 0 (Initial)</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{initial_rate*100:.0f}%"></div></div>'
            f'<div>{initial_rate:.0%}</div></div>'
        ]
        for rec in opt_log.iterations:
            label = f"Iter {rec.iteration}: {rec.failure_type}"
            bars.append(
                f'<div class="bar-row"><div class="bar-label">{html.escape(label)}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{rec.score_after*100:.0f}%"></div></div>'
                f'<div>{rec.score_after:.0%}</div></div>'
            )
        return "\n".join(bars)

    @staticmethod
    def _render_case_rows(report: EvaluationReport) -> str:
        rows: list[str] = []
        for r in report.results:
            score_cls = "badge-ok" if r.passed else "badge-fail"
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(r.test_id)}</code></td>"
                f"<td>{html.escape(r.category)}</td>"
                f"<td>{html.escape(r.question)}</td>"
                f"<td>{html.escape(r.grounded)}</td>"
                f"<td>{html.escape(r.correct)}</td>"
                f"<td>{html.escape(r.refusal)}</td>"
                f"<td>{html.escape(r.source_quality)}</td>"
                f'<td><span class="badge {score_cls}">{html.escape(r.score_label)}</span></td>'
                "</tr>"
            )
        return "\n".join(rows)
