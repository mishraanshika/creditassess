import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, fmt } from '../api/client'
import { CardTitle, InfoTip, Loading, PageHeader, StatCard } from '../components/ui'

const TABS = [
  ['performance', 'Model performance'],
  ['importance', 'Feature importance'],
  ['fairness', 'Fairness audit'],
  ['policy', 'Decision policy'],
  ['audit', 'Audit trail'],
]

const SLICE_WORDS = {
  gender: 'Gender',
  age_band: 'Age band',
  education: 'Education',
  family_status: 'Family status',
  income_band: 'Income band',
  file_type: 'File type',
}

const GROUP_WORDS = { new_to_credit: 'New to credit', thick_file: 'Thick file' }

export default function Analytics() {
  const [tab, setTab] = useState('performance')
  const [data, setData] = useState({})
  const [busy, setBusy] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      api.modelMetrics(), api.featureImportance(18), api.bias(), api.policy(), api.auditLog(40),
    ]).then(([m, f, b, p, a]) => {
      setData({
        metrics: m.status === 'fulfilled' ? m.value : null,
        importance: f.status === 'fulfilled' ? f.value : null,
        bias: b.status === 'fulfilled' ? b.value : null,
        biasError: b.status === 'rejected' ? b.reason?.message : null,
        policy: p.status === 'fulfilled' ? p.value : null,
        audit: a.status === 'fulfilled' ? a.value : null,
      })
      setBusy(false)
    })
  }, [])

  if (busy) return <><PageHeader title="Analytics & Fairness" /><Loading /></>

  const m = data.metrics

  return (
    <>
      <PageHeader
        title="Analytics & Fairness"
        subtitle="Model governance: performance, what the model keys on, whether it treats groups equitably, the policy in force, and the audit trail."
        info={<>A lending model has to be defensible, not merely accurate. These five tabs are
          what a risk officer or a regulator would ask to see.</>}
        actions={TABS.map(([k, label]) => (
          <button key={k} className={`btn ${tab === k ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => setTab(k)}>{label}</button>
        ))}
      />

      {/* ================= PERFORMANCE ================= */}
      {tab === 'performance' && m && (
        <>
          <div className="grid grid-4">
            <StatCard label="ROC AUC"
                      value={fmt.num(m.cross_validation?.roc_auc_mean, 4)}
                      foot="0.50 random · 1.00 perfect"
                      info={<>Discrimination: given one defaulter and one non-defaulter, the
                        probability the model ranks the defaulter as riskier. Cross-validated
                        over five stratified folds.</>}
                      term="ROC AUC, 5-fold stratified CV" />
            <StatCard label="Fold stability"
                      value={`±${fmt.num(m.cross_validation?.roc_auc_std, 4)}`}
                      foot={`across ${m.cross_validation?.folds} folds`}
                      info="Standard deviation of ROC AUC across folds. A tight spread means the result generalises rather than fitting one lucky split."
                      term="std(ROC AUC) across folds" />
            <StatCard label="Brier score"
                      value={fmt.num(m.calibration?.brier_calibrated, 4)}
                      foot="mean squared error of the probability · lower is better"
                      info={<>Scores the probabilities themselves rather than the ranking.
                        Penalises both miscalibration and poor discrimination, so it is the
                        single number for "are these probabilities any good".</>}
                      term="Brier score" />
            <StatCard label="Training rows"
                      value={fmt.money(m.n_train)}
                      foot={`${fmt.money(m.n_test)} held out · ${m.n_features} features`}
                      info="Stratified split. The holdout is never seen during training, so reported metrics are out-of-sample."
                      term={m.model_version} />
          </div>

          <div className="grid grid-2 mt-16">
            <div className="card">
              <CardTitle
                term="mean predicted PD vs observed default rate"
                info={<>The critical check for a policy that thresholds a probability. If
                  the model says 8%, roughly 8% must actually default — otherwise every
                  cut-off is built on a number that means nothing.</>}
              >
                Calibration
              </CardTitle>
              <div className="grid grid-2" style={{ gap: 12 }}>
                <StatCard label="Mean predicted PD"
                          value={fmt.pct(m.calibration?.mean_pd_calibrated, 2)} />
                <StatCard label="Observed default rate"
                          value={fmt.pct(m.calibration?.observed_default_rate, 2)}
                          accent="var(--ok)" />
              </div>
              <div className="banner banner-ok mt-16" style={{ fontSize: 12.5 }}>
Agreement to within a fraction of a percentage point after isotonic
                calibration on a held-out split.
              </div>
            </div>

            <div className="card">
              <CardTitle term="per-fold ROC AUC"
                         info="Trained and evaluated five times on disjoint stratified folds. Similar bar heights indicate the result generalises.">
                Cross-validation stability
              </CardTitle>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={(m.cross_validation?.per_fold_roc_auc ?? [])
                  .map((v, i) => ({ fold: `Fold ${i + 1}`, auc: v }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="fold" tick={{ fill: '#97a4c4', fontSize: 11 }} />
                  <YAxis domain={[0.7, 0.8]} tick={{ fill: '#6b7a9c', fontSize: 11 }} />
                  <Tooltip itemStyle={{ color: '#e8edf9' }} labelStyle={{ color: '#97a4c4', marginBottom: 4 }} contentStyle={{ background: '#0a0a0c', borderRadius: 10,
                                           border: '1px solid rgba(255,255,255,0.18)', fontSize: 12 }} />
                  <Bar dataKey="auc" fill="#4f7cff" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <details className="card mt-16">
            <summary style={{ cursor: 'pointer', fontSize: 13, color: 'var(--text-muted)' }}>
Classification metrics at fixed thresholds
            </summary>
            <table className="tbl mt-16">
              <thead><tr><th>Threshold</th><th className="right">Accuracy</th>
                <th className="right">Precision</th><th className="right">Recall</th>
                <th className="right">F1</th></tr></thead>
              <tbody>
                {[['0.5 (naive)', m['holdout_at_0.5']],
                  ['best F1', m.holdout_at_best_f1],
                  ['policy decline cut-off', m.holdout_at_policy_cutoff]].map(([label, v]) => v && (
                  <tr key={label}>
                    <td>{label} <span className="dim">({v.threshold})</span></td>
                    <td className="right">{fmt.num(v.accuracy, 3)}</td>
                    <td className="right">{fmt.num(v.precision, 3)}</td>
                    <td className="right">{fmt.num(v.recall, 3)}</td>
                    <td className="right">{fmt.num(v.f1, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="hint mt-16">
On an 8.07% base rate, accuracy at a fixed threshold is close to meaningless —
              a model approving everyone scores 92%. Discrimination (ROC AUC) and calibration
              (Brier) are what the policy layer actually consumes.
            </div>
          </details>
        </>
      )}

      {/* ================= IMPORTANCE ================= */}
      {tab === 'importance' && data.importance && (
        <div className="card">
          <CardTitle
            term="gain-based feature importance"
            info={<>Total split gain attributable to each feature. Green bars are the
              engineered behavioural / alternative-data block — the signals that make a
              thin-file assessment possible.</>}
          >
            Global feature importance
          </CardTitle>
          <div className="legend" style={{ marginBottom: 14 }}>
            <span><i style={{ background: '#2fd39b' }} />Engineered behavioural features</span>
            <span><i style={{ background: '#4f7cff' }} />Raw application / bureau data</span>
          </div>
          <ResponsiveContainer width="100%" height={520}>
            <BarChart data={[...data.importance.features].reverse()} layout="vertical"
                      margin={{ left: 10, right: 24 }}>
              <XAxis type="number" tick={{ fill: '#6b7a9c', fontSize: 11 }}
                     tickFormatter={(v) => `${v}%`} />
              <YAxis type="category" dataKey="label" width={215}
                     tick={{ fill: '#97a4c4', fontSize: 11 }} />
              <Tooltip itemStyle={{ color: '#e8edf9' }} labelStyle={{ color: '#97a4c4', marginBottom: 4 }} contentStyle={{ background: '#0a0a0c', borderRadius: 10,
                                       border: '1px solid rgba(255,255,255,0.18)', fontSize: 12 }}
                       formatter={(v) => [`${v}% of total gain`, 'Importance']} />
              <Bar dataKey="gain_pct" radius={[0, 6, 6, 0]}>
                {[...data.importance.features].reverse().map((f, i) => (
                  <Cell key={i} fill={f.feature.includes('score') || f.feature.includes('consistency')
                    ? '#2fd39b' : '#4f7cff'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ================= FAIRNESS ================= */}
      {tab === 'fairness' && (
        data.bias ? (
          <>
            <div className={`banner ${data.bias.overall_pass ? 'banner-ok' : 'banner-warn'}`}>
              <strong>
                {data.bias.overall_pass
                  ? 'All audited groups pass the four-fifths rule.'
                  : `${data.bias.four_fifths_failures.length} group(s) below the four-fifths (80%) threshold.`}
              </strong>
              <InfoTip
                align="left"
                term="four-fifths rule / disparate impact ratio"
                text={<>Standard fair-lending test: a group passes if its selection rate is
                  at least 80% of the best-performing group's. It measures selection-rate
                  parity, not predictive accuracy.</>}
              />
              <div className="hint" style={{ marginTop: 8, lineHeight: 1.7 }}>
Published rather than suppressed. Where a group's observed default rate is
                genuinely higher, a risk-accurate model selects it less often; group AUCs here
                are near-identical, so no group is served by a less accurate model. Resolving
                the parity-vs-accuracy trade-off is a lending-policy and legal decision, not
                one the software should make silently.
              </div>
            </div>

            <div className="grid grid-3 mt-16">
              <StatCard label="Applicants audited" value={fmt.money(data.bias.n_audited)}
                        info="The audit runs over a large sample, not a handful of cases." />
              <StatCard label="NTC share of book"
                        value={fmt.pct(data.bias.ntc_coverage?.ntc_share_of_applicants, 1)}
                        info="Proportion of audited applicants a bureau-dependent model could not properly assess."
                        term="is_ntc" />
              <StatCard label="NTC selection rate"
                        value={fmt.pct(data.bias.ntc_coverage?.ntc_selection_rate, 1)}
                        accent="var(--ok)"
                        foot={`observed default ${fmt.pct(data.bias.ntc_coverage?.ntc_observed_default_rate, 1)}`}
                        info="The mission metric: share of thin-file applicants receiving an automatic approval, next to their realised default rate." />
            </div>

            {Object.entries(data.bias.slices).map(([slice, rows]) => rows.length > 0 && (
              <div className="card mt-16" key={slice}>
                <CardTitle info={`Selection rate, disparate impact, realised default rate and group AUC across ${SLICE_WORDS[slice]?.toLowerCase() ?? slice} groups.`}>
                  {SLICE_WORDS[slice] ?? slice}
                </CardTitle>
                <div className="scroll-x">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Group</th>
                        <th className="right">People</th>
                        <th className="right">
                          Selection rate
                          <InfoTip align="left" term="selection_rate"
                                   text="Share of this group the engine would auto-approve." />
                        </th>
                        <th className="right">
                          Disparate impact
                          <InfoTip align="left" term="disparate_impact_ratio"
                                   text="This group's selection rate divided by the best-performing group's. 0.80 or above passes the four-fifths rule." />
                        </th>
                        <th className="right">
                          Observed default
                          <InfoTip align="left" term="observed_default_rate"
                                   text="Realised default rate for this group in the data — the underlying reason selection rates differ." />
                        </th>
                        <th className="right">
                          Group AUC
                          <InfoTip align="left" term="ROC AUC within group"
                                   text="Discrimination measured within this group alone. Near-identical values across groups mean no group is served by a less accurate model." />
                        </th>
                        <th>4/5 rule</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={r.group}>
                          <td>{GROUP_WORDS[r.group] ?? r.group}</td>
                          <td className="right">{fmt.money(r.n)}</td>
                          <td className="right">{fmt.pct(r.selection_rate, 1)}</td>
                          <td className="right">{fmt.num(r.disparate_impact_ratio, 3)}</td>
                          <td className="right">{fmt.pct(r.observed_default_rate, 2)}</td>
                          <td className="right">{fmt.num(r.roc_auc, 3)}</td>
                          <td>
                            <span className={`badge ${r.passes_four_fifths ? 'badge-approve' : 'badge-reject'}`}>
                              {r.passes_four_fifths ? 'PASS' : 'FAIL'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </>
        ) : (
          <div className="card"><div className="empty">
No fairness report found. Run{' '}
            <span className="mono">python -m ml.bias_check</span>.
            {data.biasError && <div className="hint mt-16">{data.biasError}</div>}
          </div></div>
        )
      )}

      {/* ================= POLICY ================= */}
      {tab === 'policy' && data.policy && (
        <div className="grid grid-2">
          <div className="card">
            <CardTitle
              term={data.policy.policy_version}
              info={<>The model emits a probability; these thresholds turn it into a
                decision. They are versioned and kept in one module so a change to lending
                policy is auditable rather than buried in code.</>}
            >
              Thresholds in force
            </CardTitle>
            <table className="tbl">
              <tbody>
                <tr>
                  <td>Auto-approve when PD is at or below</td>
                  <td className="right" style={{ fontWeight: 700, color: 'var(--ok)' }}>
                    {fmt.pct(data.policy.approve_max_pd, 1)}
                  </td>
                </tr>
                <tr>
                  <td>Decline when PD is at or above</td>
                  <td className="right" style={{ fontWeight: 700, color: 'var(--danger)' }}>
                    {fmt.pct(data.policy.reject_min_pd, 1)}
                  </td>
                </tr>
                <tr>
                  <td>Route to human review when confidence is below</td>
                  <td className="right" style={{ fontWeight: 700, color: 'var(--warn)' }}>
                    {fmt.num(data.policy.min_auto_confidence, 2)}
                  </td>
                </tr>
              </tbody>
            </table>
            <div className="banner mt-16" style={{ fontSize: 12.5 }}>
Anything between the two cut-offs, below the confidence floor, or carrying a
              fraud flag is routed to a human underwriter rather than auto-declined.
            </div>
          </div>
          <div className="card">
            <CardTitle info="Bands A1 (safest) to D2 (riskiest), cut on probability of default and used for portfolio slicing.">
              Risk bands
            </CardTitle>
            <table className="tbl">
              <thead><tr><th>Band</th><th>Tier</th><th className="right">PD below</th></tr></thead>
              <tbody>
                {data.policy.risk_bands.map((b) => (
                  <tr key={b.band}>
                    <td><span className="badge badge-info">{b.band}</span></td>
                    <td>{b.tier}</td>
                    <td className="right">{fmt.pct(Math.min(b.max_pd, 1), 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ================= AUDIT ================= */}
      {tab === 'audit' && (
        <div className="card">
          <CardTitle
            term="audit_logs"
            info={<>Append-only. Every request records the event, the model and policy
              version that answered it, latency, and a SHA-256 of the request body — which
              lets an auditor prove what was submitted without the log holding raw PII.</>}
          >
            Audit trail
          </CardTitle>
          <div className="scroll-x">
            <table className="tbl">
              <thead><tr>
                <th>Timestamp</th><th>Event</th>
                <th className="right">Latency</th><th>Model version</th>
                <th>
                  Payload hash
                  <InfoTip align="left" term="SHA-256 of the request body"
                           text="One-way hash of the submitted application. Identical input always yields the same digest, so tampering is detectable, but the digest cannot be reversed into personal data." />
                </th>
              </tr></thead>
              <tbody>
                {(data.audit?.entries ?? []).map((e, i) => (
                  <tr key={i}>
                    <td className="dim">{fmt.date(e.created_at)}</td>
                    <td>
                      <span className="badge badge-neutral">{e.event_type}</span>
                      <span className="mono dim" style={{ marginLeft: 8 }}>{e.endpoint}</span>
                    </td>
                    <td className="right">{e.latency_ms ?? '—'} ms</td>
                    <td className="mono dim">{e.model_version ?? '—'}</td>
                    <td className="mono dim">{e.payload_hash?.slice(0, 12) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(data.audit?.entries ?? []).length === 0 && (
            <div className="empty">No API activity recorded yet.</div>
          )}
        </div>
      )}
    </>
  )
}
