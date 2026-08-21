import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { SAMPLE_APPLICANTS, useAppState } from '../state/AppState'
import { CardTitle, ErrorBanner, InfoTip, PageHeader } from '../components/ui'

const SELECTS = {
  NAME_CONTRACT_TYPE: ['Cash loans', 'Revolving loans'],
  NAME_INCOME_TYPE: ['Working', 'State servant', 'Commercial associate', 'Pensioner',
    'Student', 'Businessman', 'Maternity leave', 'Unemployed'],
  NAME_EDUCATION_TYPE: ['Secondary / secondary special', 'Higher education',
    'Incomplete higher', 'Lower secondary', 'Academic degree'],
  NAME_FAMILY_STATUS: ['Married', 'Single / not married', 'Civil marriage',
    'Separated', 'Widow'],
  NAME_HOUSING_TYPE: ['House / apartment', 'Rented apartment', 'With parents',
    'Municipal apartment', 'Office apartment', 'Co-op apartment'],
  OCCUPATION_TYPE: ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers',
    'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff',
    'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers',
    'Waiters/barmen staff', 'Secretaries', 'Realty agents', 'HR staff', 'IT staff'],
  ORGANIZATION_TYPE: ['Business Entity Type 3', 'Self-employed', 'Government', 'School',
    'Trade: type 7', 'Medicine', 'Construction', 'Transport: type 4', 'Military',
    'Bank', 'Agriculture', 'Industry: type 9', 'XNA'],
  FLAG_OWN_CAR: ['N', 'Y'],
  FLAG_OWN_REALTY: ['Y', 'N'],
}

const TOGGLES = [
  ['FLAG_MOBIL', 'Mobile number'],
  ['FLAG_CONT_MOBILE', 'Phone reachable'],
  ['FLAG_EMP_PHONE', 'Work contact'],
  ['FLAG_WORK_PHONE', 'Second number'],
  ['FLAG_PHONE', 'Landline'],
  ['FLAG_EMAIL', 'Email address'],
]

export default function ApplicantForm() {
  const navigate = useNavigate()
  const { applicant, setApplicant, setAssessment, resetResults } = useAppState()
  const [form, setForm] = useState(applicant)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [showOptional, setShowOptional] = useState(false)

  const set = (key) => (e) => {
    const raw = e.target.value
    const numeric = e.target.type === 'number'
    setForm((f) => ({ ...f, [key]: numeric ? (raw === '' ? undefined : Number(raw)) : raw }))
  }
  const toggle = (key) => () => setForm((f) => ({ ...f, [key]: f[key] ? 0 : 1 }))

  const loadSample = (key) => {
    setForm(SAMPLE_APPLICANTS[key].applicant)
    resetResults()
  }

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(null)
    try {
      const clean = Object.fromEntries(
        Object.entries(form).filter(([, v]) => v !== undefined && v !== '')
      )
      const result = await api.predict(clean)
      setApplicant(clean)
      setAssessment(result)
      navigate('/risk')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const num = (key, label, { info, term, hint } = {}) => (
    <label className="field" key={key}>
      <span className="field-label">
        {label}
        {info && <InfoTip text={info} term={term} />}
      </span>
      <input className="input" type="number" value={form[key] ?? ''} onChange={set(key)} />
      {hint && <div className="hint">{hint}</div>}
    </label>
  )

  const sel = (key, label, info) => (
    <label className="field" key={key}>
      <span className="field-label">
        {label}
        {info && <InfoTip text={info} term={key} />}
      </span>
      <select className="input" value={form[key] ?? ''} onChange={set(key)}>
        <option value="">Not provided</option>
        {SELECTS[key].map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  )

  return (
    <>
      <PageHeader
        title="Applicant Intake"
        subtitle="Two required fields. Everything else improves the assessment but nothing is mandatory."
        info={<>Bureau-dependent systems cannot assess an applicant without a credit file.
          This form inverts that: the bureau block is optional, and the behavioural /
          alternative-data block carries the assessment.</>}
        actions={Object.entries(SAMPLE_APPLICANTS).map(([k, v]) => (
          <button key={k} type="button" className="btn btn-ghost" onClick={() => loadSample(k)}>
            {v.label}
          </button>
        ))}
      />

      <ErrorBanner error={error} hint="Is the server running? Start it with `python -m backend.run`." />

      <form onSubmit={submit}>
        {/* ---- the essentials ---- */}
        <div className="grid grid-3">
          <div className="card">
            <CardTitle info="Reference details used to identify the application and its audit record.">
              Identity
            </CardTitle>
            <label className="field">
              <span className="field-label">Full name</span>
              <input className="input" value={form.full_name ?? ''} onChange={set('full_name')} />
            </label>
            <label className="field">
              <span className="field-label">
                Reference
                <InfoTip text="Your own file or CRM identifier for this application." term="external_ref" />
              </span>
              <input className="input" value={form.external_ref ?? ''} onChange={set('external_ref')} />
            </label>
            {num('age_years', 'Age')}
            {num('CNT_FAM_MEMBERS', 'Household size', {
              info: 'Income is assessed per household member, and the affordability budget takes a dependants haircut.',
              term: 'CNT_FAM_MEMBERS',
            })}
          </div>

          <div className="card">
            <CardTitle info="Annual income and requested credit are the only mandatory inputs.">
              Financials <span className="badge badge-info" style={{ marginLeft: 8 }}>Required</span>
            </CardTitle>
            {num('AMT_INCOME_TOTAL', 'Annual income', {
              info: 'Declared gross annual income. Drives the affordability calculation and income-per-head scores.',
              term: 'AMT_INCOME_TOTAL',
            })}
            {num('AMT_CREDIT', 'Credit requested', {
              info: 'Requested facility amount. The recommended limit is capped at this figure.',
              term: 'AMT_CREDIT',
            })}
            {num('AMT_ANNUITY', 'Annual annuity', {
              info: 'Total repaid per year. Drives the debt-to-income ratio and the implied term. Left blank, a 36-month schedule is assumed.',
              term: 'AMT_ANNUITY',
              hint: 'Optional — defaults to a 36-month schedule',
            })}
            {num('AMT_GOODS_PRICE', 'Goods price', {
              info: 'Price of the financed asset. A large cash-out gap between credit requested and goods price is a known fraud tell and raises a flag.',
              term: 'AMT_GOODS_PRICE',
              hint: 'Optional',
            })}
          </div>

          <div className="card">
            <CardTitle info="Tenure and employer type drive the income-stability score.">
              Employment
            </CardTitle>
            {num('employment_years', 'Years in current employment', {
              info: 'The dominant component of income stability. Converted server-side to DAYS_EMPLOYED.',
              term: 'employment_years -> DAYS_EMPLOYED',
            })}
            {sel('OCCUPATION_TYPE', 'Occupation')}
            {sel('ORGANIZATION_TYPE', 'Employer type',
                 'A formally recognised organisation scores higher than an unregistered one.')}
            {sel('NAME_INCOME_TYPE', 'Income type')}
          </div>
        </div>

        {/* ---- the differentiator ---- */}
        <div className="grid grid-2 mt-16">
          <div className="card">
            <CardTitle
              term="alternative data / behavioural signals"
              info={<>Behavioural signals that make a thin-file assessment possible. Handset
                tenure, contactability and document completeness measure the same underlying
                reliability a bureau score proxies, gathered from a different source.</>}
            >
              Alternative data <span className="badge badge-approve" style={{ marginLeft: 8 }}>
                Replaces the bureau score
              </span>
            </CardTitle>

            <div className="grid grid-2" style={{ gap: 14 }}>
              {num('months_on_current_handset', 'Months on current handset', {
                info: 'Telco analogue of an uninterrupted recharge record. Under 3 months raises a RECENT_DEVICE_CHANGE flag.',
                term: 'months_on_current_handset -> DAYS_LAST_PHONE_CHANGE',
              })}
              {num('documents_submitted', 'Supporting documents', {
                info: 'Count of supporting documents supplied; mapped to the FLAG_DOCUMENT_* slots. Document completeness correlates with repayment.',
                term: 'documents_submitted -> FLAG_DOCUMENT_*',
              })}
            </div>

            <div className="field-label" style={{ marginTop: 6 }}>
              Contactable channels
              <InfoTip term="FLAG_MOBIL, FLAG_CONT_MOBILE, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL"
                       text="Toggle each channel on file. Breadth of contactability feeds the digital trust and mobile recharge scores." />
            </div>
            <div className="grid grid-2" style={{ gap: 8, marginTop: 8 }}>
              {TOGGLES.map(([k, label]) => (
                <button key={k} type="button" onClick={toggle(k)}
                        className={`badge ${form[k] ? 'badge-approve' : 'badge-neutral'}`}
                        style={{ cursor: 'pointer', justifyContent: 'center', padding: '10px' }}>
                  {form[k] ? '✓' : '○'} {label}
                </button>
              ))}
            </div>
          </div>

          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <CardTitle info="The pipeline that runs on submit, in a single request.">
              Run the assessment
            </CardTitle>
            <div className="step"><span className="step-num">1</span>
              <div>Behavioural feature engineering — 11 alternative-data scores.</div></div>
            <div className="step"><span className="step-num">2</span>
              <div>Calibrated XGBoost produces the probability of default.</div></div>
            <div className="step"><span className="step-num">3</span>
              <div>Policy engine sets band, limit, confidence and review triggers.</div></div>
            <div className="step"><span className="step-num">4</span>
              <div>TreeSHAP attribution plus similar-borrower retrieval.</div></div>

            <button className="btn btn-primary" disabled={busy} type="submit"
                    style={{ marginTop: 'auto', width: '100%', padding: '14px' }}>
              {busy ? 'Assessing…' : 'Run risk assessment'}
            </button>
            <div className="hint" style={{ textAlign: 'center', marginTop: 10 }}>
Sub-second end to end. Every decision is written to the audit trail.
            </div>
          </div>
        </div>

        {/* ---- optional, collapsed by default ---- */}
        <div className="card mt-16">
          <div className="between">
            <CardTitle
              term="EXT_SOURCE_1/2/3, AMT_REQ_CREDIT_BUREAU_*"
              info={<>Leave empty for a genuine new-to-credit applicant. Missing values are
                kept as NaN rather than imputed, so the model applies a learned default split
                direction, and the confidence layer lowers data sufficiency to match.</>}
            >
              Bureau block <span className="badge badge-neutral" style={{ marginLeft: 8 }}>
                Optional — leave empty for NTC
              </span>
            </CardTitle>
            <button type="button" className="btn btn-ghost"
                    onClick={() => setShowOptional((s) => !s)}>
              {showOptional ? 'Hide' : 'Show'}
            </button>
          </div>

          {showOptional && (
            <div className="grid grid-3 mt-16">
              {num('EXT_SOURCE_1', 'External score 1', {
                info: 'Normalised bureau rating between 0 and 1; higher is safer.',
                term: 'EXT_SOURCE_1',
              })}
              {num('EXT_SOURCE_2', 'External score 2', { info: 'Second bureau rating, 0 to 1.', term: 'EXT_SOURCE_2' })}
              {num('EXT_SOURCE_3', 'External score 3', { info: 'Third bureau rating, 0 to 1.', term: 'EXT_SOURCE_3' })}
              {num('AMT_REQ_CREDIT_BUREAU_QRT', 'Bureau enquiries, last quarter', {
                info: 'High recent enquiry intensity indicates credit hunting and raises a CREDIT_HUNGRY flag.',
                term: 'AMT_REQ_CREDIT_BUREAU_QRT',
              })}
              {num('AMT_REQ_CREDIT_BUREAU_YEAR', 'Bureau enquiries, last year', {
                term: 'AMT_REQ_CREDIT_BUREAU_YEAR',
                info: 'Same signal over a longer window. Zero enquiries plus missing scores is what defines a new-to-credit applicant.',
              })}
              {sel('NAME_EDUCATION_TYPE', 'Education')}
              {sel('NAME_FAMILY_STATUS', 'Family status',
                   'Retained for fairness measurement. Excluded from the model matrix entirely under --strict-fairness.')}
              {sel('NAME_HOUSING_TYPE', 'Housing type')}
              <div className="grid grid-2" style={{ gap: 12 }}>
                {sel('FLAG_OWN_CAR', 'Owns car')}
                {sel('FLAG_OWN_REALTY', 'Owns realty')}
              </div>
            </div>
          )}
        </div>
      </form>
    </>
  )
}
