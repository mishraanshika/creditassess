import { Link, useNavigate } from 'react-router-dom'
import {
  Bar, BarChart, Cell, PolarAngleAxis, PolarGrid,
  Radar, RadarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { fmt } from '../api/client'
import { useAppState } from '../state/AppState'
import {
  CardTitle, ConfidenceBar, EmptyState, InfoTip, PageHeader, RECO_MEANING,
  RECO_WORDS, RiskGauge, ScoreMeter, StatCard, scoreWord,
} from '../components/ui'

const RADAR_KEYS = [
  ['payment_consistency_score', 'Payment'],
  ['income_stability_score', 'Income'],
  ['spending_stability_score', 'Spending'],
  ['credit_utilization_score', 'Headroom'],
  ['monthly_cashflow_consistency', 'Cash-flow'],
  ['utility_payment_consistency', 'Utility'],
  ['mobile_recharge_consistency', 'Recharge'],
  ['digital_trust_score', 'Digital'],
]

// Plain-English meaning for every behavioural score, shown on hover.
const SCORE_HELP = {
  payment_consistency_score: ['Payment consistency',
    'Proxy for meeting scheduled obligations: instalment affordability, implied term, defaults in the declared social circle, identity trail and document hygiene.'],
  utility_payment_consistency: ['Utility payment consistency',
    'Household bill discipline, derived from realty ownership, address stability, length of registration history and a landline on file.'],
  mobile_recharge_consistency: ['Mobile recharge consistency',
    'Telco analogue of an uninterrupted recharge record: handset age, reachable channels, always-on mobile and an employer phone.'],
  digital_trust_score: ['Digital trust',
    'Breadth and stability of the digital footprint: email on file, contactable channels, handset stability, no address contradictions.'],
  spending_stability_score: ['Spending stability',
    'Whether the credit requested tracks the financed asset. A large cash-out gap over the goods price is overreach and a known fraud tell.'],
  income_stability_score: ['Income stability',
    'Tenure dominates, plus career share, income per household member, a formally recognised employer and guaranteed inflows.'],
  credit_utilization_score: ['Credit utilisation headroom',
    'Capacity remaining after this facility: leverage, instalment burden and recent enquiry intensity. Higher means more headroom.'],
  monthly_cashflow_consistency: ['Monthly cash-flow consistency',
    'Post-instalment surplus as a share of income, per household member, penalised for dependants.'],
  transaction_volatility: ['Transaction volatility',
    'Cash-flow dispersion proxy: cash-out gap, handset churn, enquiry hunting, address contradictions and off-hours application timing. LOWER IS BETTER.'],
  financial_discipline_score: ['Financial discipline',
    'Weighted composite of payment consistency, utilisation headroom, cash-flow consistency, spending stability, utility record and (inverted) volatility.'],
  thin_file_score: ['Thin-file indicator',
    'How little bureau history exists. High means almost no traditional record — an absence of evidence, not evidence of risk.'],
}

export default function RiskAssessment() {
  const { assessment } = useAppState()
  const navigate = useNavigate()

  if (!assessment) {
    return (
      <>
        <PageHeader title="Risk Assessment" />
        <EmptyState
          title="No applicant scored in this session yet."
          action={<button className="btn btn-primary" onClick={() => navigate('/apply')}>
            Go to applicant intake
          </button>}
        />
      </>
    )
  }

  const a = assessment
  const beh = a.behavioural_features ?? {}
  const chart = (a.explanation?.contribution_chart ?? [])
    .slice(0, 10)
    .map((f) => ({ ...f, name: f.label }))
    .sort((x, y) => x.pd_impact_pp - y.pd_impact_pp)

  const radar = RADAR_KEYS.map(([k, label]) => ({ metric: label, value: beh[k] ?? 0 }))

  const headline = {
    APPROVE: 'Clears policy — straight-through approval.',
    REVIEW: 'Referred to a human underwriter.',
    REJECT: 'Declined on probability of default.',
  }[a.recommendation]

  return (
    <>
      <PageHeader
        title="Risk Assessment"
        subtitle={headline}
        info={<>One applicant, one assessment: probability of default from the calibrated
          XGBoost model, then the policy engine for band, limit, confidence and review
          triggers, then TreeSHAP attribution. Hover any (i) for a definition.</>}
        actions={[
          <Link key="s" to="/similar" className="btn btn-ghost">Similar borrowers</Link>,
          <Link key="r" to="/report" className="btn btn-primary">Generate report</Link>,
        ]}
      />

      {a.requires_human_review && (
        <div className="banner banner-warn" style={{ marginBottom: 16 }}>
          <strong>Human review required — this is not a decline.</strong> Triggers:
          <ul className="list-clean" style={{ marginTop: 6 }}>
            {a.review_reasons.map((r) => <li key={r}>{r}</li>)}
          </ul>
        </div>
      )}
      {a.fraud_flags?.length > 0 && (
        <div className="banner banner-danger" style={{ marginBottom: 16 }}>
          <strong>{a.fraud_flags.length} fraud / anomaly signal(s)</strong>
          <InfoTip
            term="fraud_flags"
            text={<>Rule-based anomaly tells, not accusations. They never auto-decline —
              they route the case to human review and tell the reviewer what to verify.</>}
          />
          <ul className="list-clean list-risk" style={{ marginTop: 8 }}>
            {a.fraud_flags.map((f) => <li key={f}>{f}</li>)}
          </ul>
        </div>
      )}

      {/* ---- the four numbers that matter ---- */}
      <div className="grid grid-4">
        <div className="card">
          <RiskGauge score={a.risk_score} band={a.risk_band}
                     tier={a.risk_tier} pd={a.probability_of_default} />
        </div>

        <div className="card">
          <div className="stat-label">
            Recommendation
            <InfoTip term={a.recommendation} text={RECO_MEANING[a.recommendation]} />
          </div>
          <div className="big-number" style={{
            marginTop: 10,
            color: a.recommendation === 'APPROVE' ? 'var(--ok)'
              : a.recommendation === 'REJECT' ? 'var(--danger)' : 'var(--warn)',
          }}>
            {RECO_WORDS[a.recommendation]}
          </div>
          <div className="stat-foot">
            {a.recommendation === 'REJECT'
              ? 'No offer extended'
              : `Limit ${fmt.money(a.recommended_credit_limit)}`}
          </div>
        </div>

        <div className="card">
          <div className="stat-label">
            Recommended limit
            <InfoTip
              term="recommended_credit_limit"
              text={<>Affordable capacity (DTI-capped monthly budget x term), scaled by a
                risk multiplier and a behavioural adjustment, then capped at the requested
                amount and rounded down.</>}
            />
          </div>
          <div className="big-number" style={{ marginTop: 10, fontSize: 30 }}>
            {fmt.money(a.recommended_credit_limit)}
          </div>
          <div className="stat-foot">
Requested {fmt.money(a.requested_amount)}
          </div>
          <div className="hint" style={{ marginTop: 10 }}>
            {fmt.money(a.suggested_monthly_instalment)}/month over {a.suggested_term_months} months
            <InfoTip align="left" term="max_affordable_limit"
                     text={<>Declared income supports up to{' '}
                       <strong>{fmt.money(a.max_affordable_limit)}</strong> at the applicable
                       DTI cap (35% for NTC, 45% otherwise). The offer sits below capacity as
                       risk rises.</>} />
          </div>
        </div>

        <div className="card">
          <ConfidenceBar score={a.confidence_score} drivers={a.confidence_drivers}
                         requiresReview={a.requires_human_review} />
        </div>
      </div>

      {/* ---- credit history status ---- */}
      <div className="card mt-16">
        <div className="between wrap" style={{ gap: 16 }}>
          <div>
            <div className="stat-label">
              Credit file
              <InfoTip
                term="is_ntc / thin_file_score"
                text={<>New-to-credit means two or more bureau scores are absent with no
                  recent enquiries. Conventional models impute a median or decline outright;
                  here the missing value is kept as <strong>NaN</strong> so the model applies
                  a learned default split direction, and the behavioural block carries the
                  assessment.</>}
              />
            </div>
            <div className="verdict-line" style={{ marginTop: 8 }}>
              {a.is_ntc
                ? 'No usable bureau record — assessed on behavioural and alternative data.'
                : 'Bureau record available and used alongside behavioural data.'}
            </div>
          </div>
          <span className={`badge ${a.is_ntc ? 'badge-info' : 'badge-neutral'}`}>
            {a.is_ntc ? 'NEW TO CREDIT' : 'THICK FILE'}
          </span>
        </div>
      </div>

      {/* ---- why ---- */}
      <div className="grid grid-2 mt-16">
        <div className="card">
          <CardTitle
            term="TreeSHAP contributions, expressed in percentage points of PD"
            info={<>Exact TreeSHAP. Each bar is one feature's marginal effect on the
              probability of default, in percentage points. Contributions are additive, so
              the base rate plus every bar reconstructs the final PD.</>}
          >
            SHAP attribution
          </CardTitle>
          <div className="legend" style={{ marginBottom: 12 }}>
            <span><i style={{ background: '#2fd39b' }} />Reduces PD</span>
            <span><i style={{ background: '#ff5f6d' }} />Increases PD</span>
          </div>
          <ResponsiveContainer width="100%" height={Math.max(280, chart.length * 34)}>
            <BarChart data={chart} layout="vertical"
                      margin={{ left: 10, right: 24, top: 4, bottom: 4 }}>
              <XAxis type="number" tick={{ fill: '#6b7a9c', fontSize: 11 }}
                     tickFormatter={(v) => (v > 0 ? `+${v}` : `${v}`)} />
              <YAxis type="category" dataKey="name" width={185}
                     tick={{ fill: '#97a4c4', fontSize: 11 }} />
              <ReferenceLine x={0} stroke="rgba(255,255,255,0.25)" />
              <Tooltip itemStyle={{ color: '#e8edf9' }} labelStyle={{ color: '#97a4c4', marginBottom: 4 }}
                cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                contentStyle={{ background: '#0a0a0c', border: '1px solid rgba(255,255,255,0.18)',
                                borderRadius: 10, fontSize: 12 }}
                formatter={(v, _n, p) => [
                  `${v > 0 ? '+' : ''}${v} pp · value ${p.payload.value_display}`,
                  'PD impact']} />
              <Bar dataKey="pd_impact_pp" radius={[4, 4, 4, 4]}>
                {chart.map((d, i) => (
                  <Cell key={i} fill={d.pd_impact_pp > 0 ? '#ff5f6d' : '#2fd39b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid" style={{ gap: 16, alignContent: 'start' }}>
          <div className="card">
            <CardTitle info={<>Each axis is one behavioural score on a 0-100 scale. A
              large, even polygon is a uniformly strong applicant; a spiky one is strong on
              some dimensions and hollow on others.</>}>
              Behavioural profile
            </CardTitle>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radar} outerRadius="72%">
                <PolarGrid stroke="rgba(255,255,255,0.12)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#97a4c4', fontSize: 10.5 }} />
                <Radar dataKey="value" stroke="#4f7cff" fill="#4f7cff" fillOpacity={0.35} />
                <Tooltip itemStyle={{ color: '#e8edf9' }} labelStyle={{ color: '#97a4c4', marginBottom: 4 }} contentStyle={{ background: '#0a0a0c', borderRadius: 10,
                                         border: '1px solid rgba(255,255,255,0.18)', fontSize: 12 }}
                         formatter={(v) => [`${scoreWord(v)} (${Math.round(v)}/100)`, '']} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-2" style={{ gap: 16 }}>
            <div className="card">
              <CardTitle term="negative SHAP contributions"
                         info="Features contributing most to lowering the probability of default.">
                <span style={{ color: 'var(--ok)' }}>Top positive factors</span>
              </CardTitle>
              <ul className="list-clean list-ok">
                {(a.explanation?.top_positive_factors ?? []).slice(0, 4).map((f) => (
                  <li key={f.feature}>
                    <strong>{f.label}</strong>
                    <div className="hint">{f.value_display}</div>
                  </li>
                ))}
              </ul>
            </div>
            <div className="card">
              <CardTitle term="positive SHAP contributions"
                         info="Features contributing most to raising the probability of default.">
                <span style={{ color: 'var(--danger)' }}>Top negative factors</span>
              </CardTitle>
              <ul className="list-clean list-risk">
                {(a.explanation?.top_negative_factors ?? []).slice(0, 4).map((f) => (
                  <li key={f.feature}>
                    <strong>{f.label}</strong>
                    <div className="hint">{f.value_display}</div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* ---- behaviour detail ---- */}
      <div className="grid grid-3 mt-16">
        <div className="card">
          <CardTitle info={<>Behavioural signals derived from the application rather than
            from a bureau. This block is what makes a thin-file assessment possible.</>}>
            Alternative-data scores
          </CardTitle>
          {['payment_consistency_score', 'utility_payment_consistency',
            'mobile_recharge_consistency', 'digital_trust_score'].map((k) => (
            <ScoreMeter key={k} label={SCORE_HELP[k][0]} value={beh[k]}
                        info={SCORE_HELP[k][1]} term={k} />
          ))}
        </div>

        <div className="card">
          <CardTitle info="Affordability and capacity: can this applicant service the instalment?">
            Capacity scores
          </CardTitle>
          {['income_stability_score', 'credit_utilization_score',
            'monthly_cashflow_consistency', 'spending_stability_score'].map((k) => (
            <ScoreMeter key={k} label={SCORE_HELP[k][0]} value={beh[k]}
                        info={SCORE_HELP[k][1]} term={k} />
          ))}
        </div>

        <div className="card">
          <CardTitle info="Composite scores and the deterministic model narrative.">Summary</CardTitle>
          <ScoreMeter label={SCORE_HELP.financial_discipline_score[0]}
                      value={beh.financial_discipline_score}
                      info={SCORE_HELP.financial_discipline_score[1]}
                      term="financial_discipline_score" />
          <ScoreMeter label={SCORE_HELP.transaction_volatility[0]}
                      value={beh.transaction_volatility} invert
                      info={SCORE_HELP.transaction_volatility[1]}
                      term="transaction_volatility" />
          <ScoreMeter label={SCORE_HELP.thin_file_score[0]}
                      value={beh.thin_file_score} invert
                      info={SCORE_HELP.thin_file_score[1]}
                      term="thin_file_score" />
          <div className="hint" style={{ marginTop: 14, lineHeight: 1.65 }}>
            {a.explanation?.narrative}
          </div>
        </div>
      </div>

      {/* ---- peer evidence ---- */}
      <div className="grid grid-4 mt-16">
        <StatCard
          label="Cohort repayment rate"
          value={fmt.pct(a.cohort?.repayment_success_rate, 0)}
          foot={`${a.cohort?.cohort_size ?? 0} nearest borrowers`}
          info={<>Share of the retrieved similar-borrower cohort that repaid. Corroborates
            the model score and feeds the peer-agreement driver of confidence; it never
            overrides the decision.</>}
          term="cohort repayment_success_rate"
        />
        <StatCard
          label="Mean similarity"
          value={fmt.num(a.cohort?.mean_similarity, 3)}
          foot="1.000 would be an identical profile"
          info="Blended score: 55% MiniLM embedding cosine, 45% financial comparability re-rank."
          term="mean_similarity"
        />
        <StatCard
          label="Fraud signals"
          value={a.fraud_flags?.length ?? 0}
          accent={a.fraud_flags?.length ? 'var(--danger)' : 'var(--ok)'}
          foot={a.fraud_flags?.length ? 'verify before any offer' : 'none raised'}
          info={<>Eight rule-based anomaly tells: cash-out gap, address inconsistency,
            recent handset change, credit hunger, fresh identity, thin tenure on a large
            ticket, high volatility, missing documents.</>}
          term="fraud_flags"
        />
        <StatCard
          label="Decision latency"
          value={`${a.latency_ms} ms`}
          foot="end-to-end"
          info="Feature engineering, scoring, calibration, policy, TreeSHAP attribution and vector retrieval, in a single request."
        />
      </div>
    </>
  )
}
