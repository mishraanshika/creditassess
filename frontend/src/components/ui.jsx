import { fmt, toneOfRecommendation } from '../api/client'

/**
 * InfoTip — the one place long text is allowed to live.
 *
 * Everything that used to sit on the page as a paragraph of explanation,
 * a disclaimer, or a definition of a technical term now hides behind a small
 * (i). The page shows a plain-English label; the detail is one hover away.
 *
 * Opens on hover and on keyboard focus (it is a real <button>), so it is not
 * mouse-only. `term` renders the underlying technical name in small caps at the
 * bottom of the bubble — a non-technical reader can ignore it, and a judge or
 * risk officer can still see exactly which metric is being shown.
 */
export function InfoTip({ text, term, align = 'center' }) {
  return (
    <span className="infotip">
      <button type="button" className="infotip-icon" aria-label={typeof text === 'string' ? text : 'More information'}>
        i
      </button>
      <span className={`infotip-bubble at-${align}`} role="tooltip">
        {text}
        {term && <span className="tip-term">Technical name: {term}</span>}
      </span>
    </span>
  )
}

export function PageHeader({ title, subtitle, info, actions }) {
  return (
    <div className="topbar">
      <div>
        <h1 className="page-title">
          {title}
          {info && <InfoTip text={info} align="right" />}
        </h1>
        {subtitle && <p className="page-sub">{subtitle}</p>}
      </div>
      {actions && <div className="row wrap">{actions}</div>}
    </div>
  )
}

export function StatCard({ label, value, foot, accent, info, term }) {
  return (
    <div className="card">
      <div className="stat-label">
        {label}
        {info && <InfoTip text={info} term={term} />}
      </div>
      <div className="stat-value" style={accent ? { color: accent } : undefined}>{value}</div>
      {foot && <div className="stat-foot">{foot}</div>}
    </div>
  )
}

export function CardTitle({ children, info, term }) {
  return (
    <div className="card-title">
      {children}
      {info && <InfoTip text={info} term={term} />}
    </div>
  )
}

/** Plain-English wording for the three outcomes. */
export const RECO_WORDS = {
  APPROVE: 'Approve',
  REVIEW: 'Review',
  REJECT: 'Reject',
}

export const RECO_MEANING = {
  APPROVE: 'Probability of default is below the auto-approve cut-off and no review trigger fired. The offer can be made straight through.',
  REVIEW: 'Not a decline. The probability of default sits between the policy cut-offs, or confidence is below the auto-decision floor, so a human underwriter makes the final call.',
  REJECT: 'Probability of default is above the policy decline threshold. No offer is made.',
}

export function RecoBadge({ value, plain = true, withInfo = false }) {
  return (
    <span className={`badge badge-${toneOfRecommendation(value)}`}>
      {plain ? (RECO_WORDS[value] ?? value) : value}
      {withInfo && <InfoTip text={RECO_MEANING[value]} term={value} />}
    </span>
  )
}

export function Loading({ label = 'Working…' }) {
  return (
    <div className="empty">
      <span className="spinner" /> <span style={{ marginLeft: 10 }}>{label}</span>
    </div>
  )
}

export function ErrorBanner({ error, hint }) {
  if (!error) return null
  return (
    <div className="banner banner-danger mb-8">
      <strong>Something went wrong.</strong> {String(error)}
      {hint && <div className="hint" style={{ marginTop: 6 }}>{hint}</div>}
    </div>
  )
}

export function EmptyState({ title, action }) {
  return (
    <div className="card">
      <div className="empty">
        <div style={{ fontSize: 15, color: 'var(--text-muted)', marginBottom: 14 }}>{title}</div>
        {action}
      </div>
    </div>
  )
}

/** Turns a 0-100 score into words a non-specialist can act on. */
export function scoreWord(value, invert = false) {
  const v = invert ? 100 - (Number(value) || 0) : (Number(value) || 0)
  if (v >= 80) return 'Excellent'
  if (v >= 65) return 'Good'
  if (v >= 45) return 'Fair'
  if (v >= 30) return 'Weak'
  return 'Poor'
}

export function scoreColor(value, invert = false) {
  const v = invert ? 100 - (Number(value) || 0) : (Number(value) || 0)
  return v >= 65 ? 'var(--ok)' : v >= 45 ? 'var(--warn)' : 'var(--danger)'
}

/**
 * Score meter. Shows the word first ("Good") and the number second, because the
 * word is what a non-technical reader needs. `invert` flips the colour scale for
 * metrics where a high number is bad.
 */
export function ScoreMeter({ label, value, invert = false, info, term }) {
  const v = Math.max(0, Math.min(100, Number(value) || 0))
  const color = scoreColor(v, invert)
  return (
    <div style={{ marginBottom: 14 }}>
      <div className="between" style={{ marginBottom: 6 }}>
        <span style={{ fontSize: 13 }}>
          {label}
          {info && <InfoTip text={info} term={term} />}
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color }}>
          {scoreWord(v, invert)} <span className="dim" style={{ fontWeight: 500 }}>{fmt.num(v, 0)}</span>
        </span>
      </div>
      <div className="meter">
        <span style={{ width: `${v}%`, background: color }} />
      </div>
    </div>
  )
}

/** Semi-circular credit-score gauge (300–900). */
export function RiskGauge({ score, band, tier, pd }) {
  const clamped = Math.max(300, Math.min(900, Number(score) || 300))
  const pctOfArc = (clamped - 300) / 600
  const color = clamped >= 660 ? 'var(--ok)' : clamped >= 570 ? 'var(--warn)' : 'var(--danger)'
  const verdict = clamped >= 660 ? 'Above portfolio average' : clamped >= 570 ? 'Portfolio average' : 'Below average'
  const R = 92, CX = 110, CY = 110
  const arcLength = Math.PI * R
  const toXY = (t) => {
    const a = Math.PI * (1 - t)
    return [CX + R * Math.cos(a), CY - R * Math.sin(a)]
  }
  const [sx, sy] = toXY(0)
  const [ex, ey] = toXY(1)

  return (
    <div style={{ display: 'grid', placeItems: 'center' }}>
      <div className="stat-label" style={{ marginBottom: -6 }}>
        Risk score
        <InfoTip
          term="risk_score (300–900, log-odds scaled)"
          text={<>A single number summarising risk, on the familiar 300–900 scale.
            <strong> Higher is safer.</strong> 600 is the average applicant in this
            portfolio. Every 40 points means the odds of repaying double.</>}
        />
      </div>
      <svg width="220" height="132" viewBox="0 0 220 132" role="img"
           aria-label={`Risk score ${clamped} out of 900`}>
        <path d={`M ${sx} ${sy} A ${R} ${R} 0 0 1 ${ex} ${ey}`}
              fill="none" stroke="rgba(255,255,255,0.09)" strokeWidth="15" strokeLinecap="round" />
        <path d={`M ${sx} ${sy} A ${R} ${R} 0 0 1 ${ex} ${ey}`}
              fill="none" stroke={color} strokeWidth="15" strokeLinecap="round"
              strokeDasharray={`${arcLength * pctOfArc} ${arcLength}`} />
        <text x={CX} y={CY - 16} textAnchor="middle" fontSize="36" fontWeight="700" fill="#e8edf9">
          {clamped}
        </text>
        <text x={CX} y={CY + 6} textAnchor="middle" fontSize="10.5" fill={color}>
          {verdict}
        </text>
      </svg>
      <div className="row" style={{ gap: 8, marginTop: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
        <span className="badge badge-neutral">
          PD {fmt.pct(pd, 2)}
          <InfoTip
            align="left"
            term="probability_of_default (PD)"
            text={<>Out of 100 people with this exact profile, this is how many we
              expect to fall seriously behind on repayments. It is an estimate from
              historical data, not a prediction about this individual.</>}
          />
        </span>
        <span className="badge badge-info">
          Band {band}
          <InfoTip align="left" term={`risk_band ${band} / ${tier}`}
                   text={<>Risk grades run A1 (safest) to D2 (riskiest). This applicant
                     falls in the <strong>{tier}</strong> group.</>} />
        </span>
      </div>
    </div>
  )
}

const CONFIDENCE_DRIVER_WORDS = {
  data_sufficiency: ['Data sufficiency', 'How much real evidence backs the prediction: bureau coverage, document completeness, contactable channels and employment record.'],
  decisiveness: ['Decisiveness', 'Distance of the probability of default from the nearest policy cut-off. A PD sitting on a threshold is inherently a lower-confidence call.'],
  peer_agreement: ['Peer agreement', 'How one-sided the outcomes were among the retrieved similar borrowers. A split cohort lowers confidence.'],
}

export function ConfidenceBar({ score, drivers, requiresReview }) {
  const pct = Math.round((Number(score) || 0) * 100)
  const color = pct >= 80 ? 'var(--ok)' : pct >= 70 ? 'var(--warn)' : 'var(--danger)'
  const word = pct >= 80 ? 'High' : pct >= 70 ? 'Adequate' : 'Below auto-decision floor'
  return (
    <div>
      <div className="between mb-8">
        <span className="stat-label">
          Confidence
          <InfoTip
            term="confidence_score = data sufficiency x decisiveness x peer agreement"
            text={<>How much the engine trusts its own answer — separate from the risk
              itself. A low-PD applicant can still score low confidence if there is little
              evidence behind the prediction. Below <strong>0.70</strong> the case is routed
              to a human instead of being auto-decided.</>}
          />
        </span>
        <span style={{ fontWeight: 700, color }}>{pct}%</span>
      </div>
      <div className="meter"><span style={{ width: `${pct}%`, background: color }} /></div>
      <div className="verdict-line" style={{ marginTop: 10, color }}>
        {word}
        <span className="muted" style={{ fontSize: 13 }}>
          {' — '}{requiresReview ? 'routed to human review' : 'eligible for auto-decision'}
        </span>
      </div>
      {drivers && (
        <div className="mt-16">
          {Object.entries(drivers).map(([k, v]) => {
            const [label, explain] = CONFIDENCE_DRIVER_WORDS[k] ?? [k.replace(/_/g, ' '), null]
            return (
              <div key={k} className="between" style={{ fontSize: 12.5, padding: '5px 0' }}>
                <span className="muted">
                  {label}
                  {explain && <InfoTip text={explain} term={k} />}
                </span>
                <span>{fmt.pct(v, 0)}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
