import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmt } from '../api/client'
import { useAppState } from '../state/AppState'
import {
  CardTitle, ErrorBanner, Loading, PageHeader, RECO_WORDS, StatCard,
} from '../components/ui'

const TONES = [
  ['credit_committee', 'Credit committee'],
  ['risk_memo', 'Risk memo'],
  ['customer_letter', 'Customer letter'],
]

export default function UnderwritingReport() {
  const { applicant, report, setReport } = useAppState()
  const [tone, setTone] = useState('credit_committee')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  const generate = async () => {
    setBusy(true); setError(null)
    try {
      setReport(await api.underwritingReport(applicant, tone))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const copy = () => {
    if (!report) return
    const text = [
      report.executive_summary, '',
      'IN THEIR FAVOUR', ...report.strengths.map((s) => `- ${s}`), '',
      'CONCERNS', ...report.risk_factors.map((s) => `- ${s}`), '',
      'BEFORE ANY MONEY IS RELEASED', ...report.conditions.map((s) => `- ${s}`), '',
      'SIMILAR BORROWERS', report.similar_borrower_insight, '',
      'FULL ASSESSMENT', report.detailed_explanation, '',
      'COMPLIANCE', report.compliance_note,
    ].join('\n')
    navigator.clipboard?.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const d = report?.decision

  return (
    <>
      <PageHeader
        title="Underwriting Report"
        subtitle="The underwriting memo accompanying this decision, ready for a credit committee or an auditor."
        info={<>The LLM explains a decision already made by the model and the policy
          engine — it cannot overturn it, restate a different limit, or cite a figure it was
          not given. Protected attributes (gender, age, marital status) are omitted from its
          context entirely, so it cannot reason from them.</>}
        actions={[
          <select key="t" className="input" style={{ width: 195 }} value={tone}
                  onChange={(e) => setTone(e.target.value)}>
            {TONES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>,
          report && <button key="c" className="btn btn-ghost" onClick={copy}>
            {copied ? 'Copied' : 'Copy memo'}
          </button>,
          <button key="g" className="btn btn-primary" onClick={generate} disabled={busy || !applicant}>
            {busy ? 'Generating…' : report ? 'Regenerate' : 'Generate report'}
          </button>,
        ].filter(Boolean)}
      />

      <ErrorBanner error={error} />

      {!report && !busy && (
        <div className="card">
          <div className="empty">
            <div style={{ marginBottom: 12, fontSize: 15 }}>
              No memo generated for this applicant yet.
            </div>
            <div className="hint" style={{ maxWidth: 540, margin: '0 auto 20px', lineHeight: 1.7 }}>
              With an API key configured the memo is written by Gemini under a grounded,
              schema-constrained prompt. Without one, the same seven sections are produced
              deterministically from the SHAP and cohort evidence.
            </div>
            <button className="btn btn-primary" onClick={generate} disabled={!applicant}>
              Generate report
            </button>
            {!applicant && <div className="hint mt-16">
              <Link to="/apply">Fill in an application first →</Link></div>}
          </div>
        </div>
      )}

      {busy && <Loading label="Generating the underwriting memo…" />}

      {report && !busy && (
        <>
          <div className="grid grid-4">
            <div className="card">
              <div className="stat-label">Decision</div>
              <div className="big-number" style={{
                fontSize: 26, marginTop: 10,
                color: report.recommendation === 'APPROVE' ? 'var(--ok)'
                  : report.recommendation === 'REJECT' ? 'var(--danger)' : 'var(--warn)',
              }}>
                {RECO_WORDS[report.recommendation]}
              </div>
              <div className="stat-foot">
                risk score {d?.risk_score} · band {d?.risk_band}
              </div>
            </div>
            <StatCard label="Recommended limit"
                      value={fmt.money(report.suggested_credit_limit)}
                      foot={`${fmt.money(d?.suggested_monthly_instalment)}/month over ${d?.suggested_term_months} months`}
                      info="DTI-capped affordable capacity, scaled by risk, capped at the requested amount." />
            <StatCard label="Confidence"
                      value={fmt.pct(report.confidence_score, 0)}
                      foot={d?.requires_human_review ? 'human review required' : 'eligible for auto-decision'}
                      info="Below the 0.70 auto-decision floor the case is routed to a human underwriter."
                      term="confidence_score" />
            <StatCard
              label="Generated by"
              value={report.generator.startsWith('gemini') ? 'Gemini' : 'Template'}
              foot={`${report.prompt_version} · ${report.latency_ms} ms`}
              info={<>Provenance is always stated. With an API key, Gemini writes the memo
                under a JSON-schema-constrained prompt. Without one, a deterministic template
                produces the same sections from identical evidence — the figures are the same
                either way.</>}
              term={report.generator}
            />
          </div>

          <div className="card mt-16">
            <CardTitle info="The decision and its single strongest driver, in two or three sentences.">
              Executive summary
            </CardTitle>
            <p className="prose" style={{ fontSize: 15 }}>{report.executive_summary}</p>
          </div>

          <div className="grid grid-2 mt-16">
            <div className="card">
              <CardTitle info="Evidence-backed strengths, each citing a figure supplied in the model context.">
                <span style={{ color: 'var(--ok)' }}>Strengths</span>
              </CardTitle>
              <ul className="list-clean list-ok">
                {report.strengths.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>
            <div className="card">
              <CardTitle info="Factors raising the assessed risk. These are the adverse-action reasons if credit is declined.">
                <span style={{ color: 'var(--danger)' }}>Risk factors</span>
              </CardTitle>
              <ul className="list-clean list-risk">
                {report.risk_factors.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>
          </div>

          {report.conditions?.length > 0 && (
            <div className="card mt-16">
              <CardTitle info="Verifications required before disbursal.">
                <span style={{ color: 'var(--warn)' }}>Conditions before disbursal</span>
              </CardTitle>
              <ul className="list-clean">{report.conditions.map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
          )}

          <div className="grid grid-2 mt-16">
            <div className="card">
              <CardTitle info="Capacity, stability, behaviour, thin-file treatment and a sensitivity note on what would change the outcome.">
                Detailed assessment
              </CardTitle>
              <p className="prose">{report.detailed_explanation}</p>
            </div>
            <div className="grid" style={{ gap: 16, alignContent: 'start' }}>
              <div className="card">
                <CardTitle info="What the retrieved cohort implies, with its repayment rate and similarity.">
                  Similar borrower evidence
                </CardTitle>
                <p className="prose">{report.similar_borrower_insight}</p>
              </div>
              <div className="card">
                <CardTitle
                  term="model governance / fair lending"
                  info={<>The model-governance paragraph: which model and policy version
                    produced the decision, that the cited drivers are the largest SHAP
                    contributions, that no protected attribute was used as a reason, and
                    whether human review is required.</>}
                >
                  Compliance note
                </CardTitle>
                <p className="prose">{report.compliance_note}</p>
              </div>
            </div>
          </div>

          <div className="banner mt-16" style={{ marginTop: 16 }}>
            <strong>Disclaimer.</strong> This memo explains a decision produced by the
            probability-of-default model and the documented lending policy. It is a
            recommendation for a credit officer, not a binding offer of credit.
          </div>
        </>
      )}
    </>
  )
}
