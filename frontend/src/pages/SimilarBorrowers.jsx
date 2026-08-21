import { Fragment, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmt } from '../api/client'
import { useAppState } from '../state/AppState'
import {
  CardTitle, EmptyState, ErrorBanner, InfoTip, Loading, PageHeader, StatCard,
} from '../components/ui'

/** Parse the pipe-delimited borrower profile into displayable key/value pairs. */
function parseProfile(text) {
  return (text ?? '')
    .split('|')
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const idx = part.indexOf(':')
      return idx === -1 ? [part, ''] : [part.slice(0, idx), part.slice(idx + 1).trim()]
    })
}

export default function SimilarBorrowers() {
  const { applicant, assessment } = useAppState()
  const [topK, setTopK] = useState(8)
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(null)

  const load = async (k = topK) => {
    setBusy(true); setError(null)
    try {
      setData(await api.similarBorrowers(applicant, k))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => { load(topK) /* eslint-disable-next-line */ }, [])

  if (!applicant) {
    return (
      <>
        <PageHeader title="Similar Borrowers" />
        <EmptyState title="Score an applicant first."
                    action={<Link className="btn btn-primary" to="/apply">Applicant intake</Link>} />
      </>
    )
  }

  const cohort = data?.cohort
  const rows = data?.similar_borrowers ?? []
  const repaidCount = rows.filter((r) => r.repaid).length

  return (
    <>
      <PageHeader
        title="Similar Borrowers"
        subtitle="Top-K retrieval over 20,000 historical borrowers with known repayment outcomes."
        info={<>The applicant is rendered as a natural-language borrower profile, embedded
          with all-MiniLM-L6-v2 (384-dim) and matched by cosine similarity, then re-ranked on
          financial comparability. Evidence that corroborates the decision — never the sole
          basis for an outcome.</>}
        actions={[
          <select key="k" className="input" style={{ width: 150 }} value={topK}
                  onChange={(e) => { setTopK(Number(e.target.value)); load(Number(e.target.value)) }}>
            {[5, 8, 10, 15, 25].map((n) => <option key={n} value={n}>Top {n}</option>)}
          </select>,
          <button key="r" className="btn btn-ghost" onClick={() => load()} disabled={busy}>
            Refresh
          </button>,
        ]}
      />

      <ErrorBanner error={error}
                   hint="The search index may not be built yet: run `python -m embeddings.build_index`." />

      {busy && !data && <Loading label="Searching past borrowers…" />}

      {data && (
        <>
          {/* ---- headline answer in one sentence ---- */}
          <div className={`card`} style={{ marginBottom: 16 }}>
            <div className="verdict-line" style={{ fontSize: 17 }}>
              Of the <strong>{cohort?.cohort_size ?? 0}</strong> most similar past borrowers,{' '}
              <strong style={{ color: repaidCount === rows.length ? 'var(--ok)' : 'var(--warn)' }}>
                {repaidCount} repaid
              </strong>{' '}
              and <strong style={{ color: rows.length - repaidCount ? 'var(--danger)' : 'var(--text-muted)' }}>
                {rows.length - repaidCount} did not
              </strong>.
              <InfoTip
                align="left"
                term="cohort repayment_success_rate"
                text={<>Realised historical outcomes, not predictions. A one-sided cohort
                  corroborates the score; a split cohort signals genuine uncertainty and
                  lowers the confidence score.</>}
              />
            </div>
          </div>

          <div className="grid grid-4">
            <StatCard
              label="Repayment rate"
              value={fmt.pct(cohort?.repayment_success_rate, 1)}
              foot={`${cohort?.cohort_size ?? 0} nearest borrowers`}
              accent={(cohort?.repayment_success_rate ?? 0) >= 0.8 ? 'var(--ok)' : 'var(--warn)'}
              info="Share of the retrieved cohort that repaid without serious delinquency."
              term="repayment_success_rate"
            />
            <StatCard
              label="Similarity-weighted"
              value={fmt.pct(cohort?.similarity_weighted_repayment_rate, 1)}
              foot="nearer peers weighted higher"
              info="The same rate, weighted by similarity so a close match counts more than a distant one."
              term="similarity_weighted_repayment_rate"
            />
            <StatCard
              label="Mean similarity"
              value={fmt.num(cohort?.mean_similarity, 3)}
              foot={`max ${fmt.num(cohort?.max_similarity, 3)}`}
              info="Blended 55% embedding cosine / 45% financial comparability. Above ~0.75 is a genuinely comparable borrower."
              term="mean_similarity"
            />
            <StatCard
              label="Retrieval latency"
              value={`${data.latency_ms} ms`}
              foot={`${data.backend} · 20,000 vectors`}
              info={<>Exact cosine search over a FAISS IndexFlatIP of 384-dim embeddings,
                followed by a numeric re-rank of the candidate pool.</>}
              term={`${data.encoder} + ${data.backend}`}
            />
          </div>

          {assessment && (
            <div className="banner mt-16" style={{ marginTop: 16 }}>
              <strong>This feeds back into the decision.</strong> Cohort agreement of{' '}
              <strong>{fmt.pct(cohort?.agreement, 0)}</strong> sets the peer-agreement driver
              of the confidence score to{' '}
              <strong>{fmt.pct(assessment.confidence_drivers?.peer_agreement, 0)}</strong>.
              <InfoTip
                align="left"
                term="peer_agreement driver of confidence_score"
                text={<>A cohort splitting 50/50 indicates a genuinely borderline profile,
                  so the engine lowers its own confidence and routes to human review rather
                  than deciding alone.</>}
              />
            </div>
          )}

          {/* ---- the matches ---- */}
          <div className="card mt-16">
            <CardTitle
              info={<>Each row is a real borrower from the Home Credit dataset. Repaid /
                Defaulted are recorded outcomes (TARGET), not estimates. Expand a row for the
                full rendered profile.</>}
            >
              Nearest historical borrowers
            </CardTitle>
            <div className="scroll-x">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Similarity</th>
                    <th>Outcome</th>
                    <th className="right">Income</th>
                    <th className="right">Credit</th>
                    <th className="right">Tenure</th>
                    <th>File</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((b) => (
                    <Fragment key={b.borrower_id}>
                      <tr>
                        <td>
                          <div className="row" style={{ gap: 10 }}>
                            <div className="meter" style={{ width: 60 }}>
                              <span style={{ width: `${Math.max(0, b.similarity_score) * 100}%`,
                                             background: 'var(--brand)' }} />
                            </div>
                            <span className="dim">{fmt.num(b.similarity_score, 3)}</span>
                          </div>
                        </td>
                        <td>
                          <span className={`badge ${b.repaid ? 'badge-approve' : 'badge-reject'}`}>
                            {b.repaid ? 'Repaid' : 'Defaulted'}
                          </span>
                        </td>
                        <td className="right">{fmt.money(b.AMT_INCOME_TOTAL)}</td>
                        <td className="right">{fmt.money(b.AMT_CREDIT)}</td>
                        <td className="right">{fmt.num(b.employment_years)} y</td>
                        <td>
                          <span className="badge badge-neutral">
                            {b.is_ntc ? 'NTC' : 'Thick'}
                          </span>
                        </td>
                        <td className="right">
                          <button className="btn btn-ghost" style={{ padding: '5px 10px', fontSize: 12 }}
                                  onClick={() => setExpanded(expanded === b.borrower_id ? null : b.borrower_id)}>
                            {expanded === b.borrower_id ? 'Hide' : 'Details'}
                          </button>
                        </td>
                      </tr>
                      {expanded === b.borrower_id && (
                        <tr>
                          <td colSpan={7} style={{ background: 'rgba(255,255,255,0.02)' }}>
                            <div className="grid grid-3" style={{ gap: 8 }}>
                              {parseProfile(b.profile_text).map(([k, v]) => (
                                <div key={k} className="between" style={{ fontSize: 12.5 }}>
                                  <span className="dim">{k}</span>
                                  <span>{v}</span>
                                </div>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <details className="card mt-16">
            <summary style={{ cursor: 'pointer', fontSize: 13, color: 'var(--text-muted)' }}>
Methodology, and what this does not claim
            </summary>
            <div className="prose" style={{ marginTop: 14 }}>
The applicant is rendered as a bucketed natural-language profile, encoded with
              all-MiniLM-L6-v2 into a 384-dimension vector, and matched by cosine similarity
              against 20,000 historical borrowers. Because the profiles share a fixed
              template, raw cosine saturates near 1.0 for almost any pair — so the candidate
              pool is re-ranked on financial comparability (income, ticket size, ratios,
              tenure, discipline) before the top K is returned.
              {'\n\n'}
What this does not claim: that this applicant will behave as these borrowers
              did. It is a statistical reference class drawn from history, used to corroborate
              the model and to quantify confidence. It is never the sole basis for a decision.
            </div>
            <div className="hint mt-16">Profile used for the search:</div>
            <div className="mono muted" style={{ lineHeight: 1.8, marginTop: 6 }}>
              {data.query_profile}
            </div>
          </details>
        </>
      )}
    </>
  )
}
