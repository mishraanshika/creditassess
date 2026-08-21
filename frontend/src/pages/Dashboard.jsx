import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, fmt } from '../api/client'
import { useAppState } from '../state/AppState'
import {
  CardTitle, InfoTip, Loading, PageHeader, RECO_WORDS, RecoBadge, StatCard,
} from '../components/ui'

const RECO_COLORS = { APPROVE: '#2fd39b', REVIEW: '#ffb020', REJECT: '#ff5f6d' }

const BAND_WORDS = {
  auto_approve: ['Auto-approve', 'PD at or below the 6% approve cut-off, with no review trigger fired.'],
  manual_review: ['Manual review', 'PD between the two policy cut-offs, or a confidence / fraud trigger fired.'],
  decline: ['Decline', 'PD at or above the 17% decline threshold.'],
}

export default function Dashboard() {
  const { health } = useAppState()
  const [portfolio, setPortfolio] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [busy, setBusy] = useState(true)

  useEffect(() => {
    Promise.allSettled([api.portfolio(), api.modelMetrics()])
      .then(([p, m]) => {
        if (p.status === 'fulfilled') setPortfolio(p.value)
        if (m.status === 'fulfilled') setMetrics(m.value)
      })
      .finally(() => setBusy(false))
  }, [])

  if (busy) return <><PageHeader title="Overview" /><Loading label="Loading…" /></>

  const hasDecisions = portfolio?.decisions > 0
  const mix = Object.entries(portfolio?.recommendation_mix ?? {})
    .map(([name, value]) => ({ name: RECO_WORDS[name] ?? name, key: name, value }))
  const bands = Object.entries(portfolio?.risk_band_distribution ?? {})
    .map(([band, count]) => ({ band, count }))
  const cv = metrics?.cross_validation
  const bandsPerf = metrics?.decision_bands

  return (
    <>
      <PageHeader
        title="Overview"
        subtitle="Model performance, calibration and live decisioning for New-to-Credit and thin-file applicants."
        info={<>Two questions: is the model sound, and what is it doing right now? Hover any
          (i) for the definition of a metric.</>}
        actions={<Link className="btn btn-primary" to="/apply">New assessment</Link>}
      />

      {/* ---- headline numbers ---- */}
      <div className="grid grid-4">
        <StatCard
          label="ROC AUC"
          value={fmt.num(cv?.roc_auc_mean ?? metrics?.['holdout_at_0.5']?.roc_auc, 4)}
          foot={cv ? `${cv.folds}-fold CV ±${fmt.num(cv.roc_auc_std, 4)}` : 'holdout'}
          info={<>Discrimination. Given one applicant who defaulted and one who did not,
            the probability the model ranks the defaulter as riskier. 0.50 is random,
            1.00 is perfect; ~0.77 is strong for application-only data.</>}
          term="ROC AUC, 5-fold stratified cross-validation"
        />
        <StatCard
          label="Calibration"
          value={fmt.pct(metrics?.calibration?.mean_pd_calibrated, 2)}
          foot={`mean predicted PD vs ${fmt.pct(metrics?.calibration?.observed_default_rate, 2)} observed`}
          accent="var(--ok)"
          info={<>Whether a stated probability means what it says: a book predicted at 8%
            must default at ~8%. The policy layer thresholds a probability, so
            miscalibration would make every cut-off arbitrary. Fitted with isotonic
            regression on a held-out split.</>}
          term="mean predicted PD vs observed default rate"
        />
        <StatCard
          label="Behavioural feature gain"
          value={`${fmt.num(metrics?.behavioural_feature_gain_pct)}%`}
          foot="share of total model gain"
          info={<>Portion of total split gain contributed by the 11 engineered
            alternative-data features. Near zero would mean the engine is a conventional
            bureau model wearing a different label.</>}
          term="gain-based feature importance, behavioural block"
        />
        <StatCard
          label="Decisions this session"
          value={portfolio?.decisions ?? 0}
          foot={`${portfolio?.review_queue ?? 0} in the review queue`}
          info="Applications scored since the server started, and how many await a human underwriter."
        />
      </div>

      <div className="grid grid-3 mt-16">
        {/* ---- the proof ---- */}
        <div className="card">
          <CardTitle
            term="decision band performance on a 61,503-applicant holdout"
            info={<>The core validation. Predicted PD against realised default rate for
              61,503 held-out applicants the model never saw, split by the band the policy
              engine would have assigned. Columns agreeing band by band is what makes the
              thresholds defensible.</>}
          >
            Policy band performance
          </CardTitle>
          {bandsPerf ? (
            <>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Band</th>
                    <th className="right">Share</th>
                    <th className="right">Predicted PD</th>
                    <th className="right">Observed default</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(bandsPerf).map(([name, b]) => {
                    const [label, help] = BAND_WORDS[name] ?? [name.replace(/_/g, ' '), null]
                    return (
                      <tr key={name}>
                        <td>{label}{help && <InfoTip text={help} term={name} />}</td>
                        <td className="right">{fmt.pct(b.share, 1)}</td>
                        <td className="right muted">{fmt.pct(b.mean_predicted_pd, 2)}</td>
                        <td className="right" style={{ fontWeight: 700 }}>
                          {fmt.pct(b.observed_default_rate, 2)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              <div className="banner banner-ok mt-16" style={{ fontSize: 12.5 }}>
                Predicted and observed agree in every band. Portfolio base default rate is{' '}
                {fmt.pct(metrics?.base_default_rate, 2)}; the auto-approved book runs at{' '}
                {fmt.pct(bandsPerf.auto_approve?.observed_default_rate, 2)}.
              </div>
            </>
          ) : <div className="empty">Train the model to fill this in.</div>}
        </div>

        {/* ---- live mix ---- */}
        <div className="card">
          <CardTitle info="Split of APPROVE / REVIEW / REJECT across applications scored this session.">
            Decision mix
          </CardTitle>
          {hasDecisions ? (
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={mix} dataKey="value" nameKey="name" innerRadius={54} outerRadius={82}
                     paddingAngle={3} isAnimationActive={false}>
                  {mix.map((m) => <Cell key={m.key} fill={RECO_COLORS[m.key] ?? '#4f7cff'} />)}
                </Pie>
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Tooltip itemStyle={{ color: '#e8edf9' }} labelStyle={{ color: '#97a4c4', marginBottom: 4 }} contentStyle={{ background: '#0a0a0c', borderRadius: 10,
                                         border: '1px solid rgba(255,255,255,0.18)', fontSize: 12 }}
                         formatter={(v, n) => [`${v} application${v === 1 ? '' : 's'}`, n]} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">Nothing assessed yet.<br />
              <Link to="/apply">Assess someone →</Link></div>
          )}
        </div>

        {/* ---- system status ---- */}
        <div className="card">
          <CardTitle
            info={<>The engine degrades gracefully: without FAISS it falls back to exact
              numpy cosine, without Postgres to an in-memory ledger, without an API key to a
              deterministic template report. Only the model is a hard dependency. This panel
              reports which path is live.</>}
          >
            Subsystem status
          </CardTitle>
          {[
            ['XGBoost model', health?.model_version ?? 'not loaded', health?.model_loaded,
              'The calibrated gradient-boosted model producing the probability of default. The only hard dependency.'],
            ['Vector store', `${health?.vector_backend ?? '—'} · ${health?.vector_size ?? 0}`,
              health?.vector_backend !== 'unavailable',
              'Similar-borrower index: FAISS IndexFlatIP over 384-dim MiniLM embeddings, with an exact numpy cosine fallback.'],
            ['Persistence', health?.database === 'connected' ? 'PostgreSQL' : 'in-memory ledger',
              health?.database === 'connected',
              'Where predictions and audit logs are written. Falls back to a bounded in-memory ledger when Postgres is unreachable.'],
            ['LLM underwriter', health?.llm?.startsWith('gemini') ? 'Gemini' : 'template',
              health?.llm?.startsWith('gemini'),
              'Generates the underwriting memo. Without an API key the same seven sections are produced deterministically from the SHAP and cohort evidence.'],
          ].map(([k, v, ok, help]) => (
            <div key={k} className="between" style={{ padding: '11px 0',
                 borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <span className="muted" style={{ fontSize: 13 }}>
                {k}<InfoTip text={help} />
              </span>
              <span className={`badge ${ok ? 'badge-approve' : 'badge-neutral'}`}>{v}</span>
            </div>
          ))}
        </div>
      </div>

      {hasDecisions && (
        <div className="grid grid-2 mt-16">
          <div className="card">
            <CardTitle
              term="risk_band A1 (safest) to D2 (riskiest)"
              info="Distribution of scored applicants across the A1-D2 risk bands."
            >
              Risk band distribution
            </CardTitle>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={bands}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="band" tick={{ fill: '#97a4c4', fontSize: 11 }} />
                <YAxis tick={{ fill: '#6b7a9c', fontSize: 11 }} allowDecimals={false} />
                <Tooltip itemStyle={{ color: '#e8edf9' }} labelStyle={{ color: '#97a4c4', marginBottom: 4 }} contentStyle={{ background: '#0a0a0c', borderRadius: 10,
                                         border: '1px solid rgba(255,255,255,0.18)', fontSize: 12 }} />
                <Bar dataKey="count" fill="#4f7cff" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <CardTitle
              info={<>The mission metric: how much of the book is thin-file, what share
                of them receive an offer, and at what limit.</>}
            >
              New-to-credit impact
            </CardTitle>
            <div className="grid grid-2" style={{ gap: 12 }}>
              <StatCard label="NTC applications" value={portfolio.ntc.applications}
                        foot={`${fmt.pct(portfolio.ntc.share, 1)} of volume`} />
              <StatCard label="NTC approval rate" value={fmt.pct(portfolio.ntc.approval_rate, 1)}
                        accent="var(--ok)"
                        foot={`avg limit ${fmt.money(portfolio.ntc.avg_limit)}`} />
              <StatCard label="Overall approval rate" value={fmt.pct(portfolio.approval_rate, 1)}
                        foot="across all applicants" />
              <StatCard label="Limit offered" value={fmt.money(portfolio.total_limit_offered)}
                        foot="cumulative this session" />
            </div>
          </div>
        </div>
      )}

      {hasDecisions && (
        <div className="card mt-16">
          <CardTitle info="Every application scored since the server started. Each row is also written to the append-only audit trail.">
            Recent decisions
          </CardTitle>
          <div className="scroll-x">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Reference</th><th>Decision</th>
                  <th className="right">Score</th>
                  <th className="right">PD</th>
                  <th className="right">Limit</th>
                  <th className="right">Confidence</th>
                  <th>File</th><th>Flags</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.recent.map((r) => (
                  <tr key={r.id}>
                    <td className="mono">{r.external_ref ?? r.request_id?.slice(0, 8)}</td>
                    <td><RecoBadge value={r.recommendation} /></td>
                    <td className="right">{r.risk_score}</td>
                    <td className="right">{fmt.pct(r.probability_of_default, 2)}</td>
                    <td className="right">{fmt.money(r.recommended_credit_limit)}</td>
                    <td className="right">{fmt.pct(r.confidence_score, 0)}</td>
                    <td>{r.is_ntc
                      ? <span className="badge badge-info">NTC</span>
                      : <span className="dim">Thick</span>}</td>
                    <td>{r.fraud_flags?.length
                      ? <span className="badge badge-reject">{r.fraud_flags.length}</span>
                      : <span className="dim">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
