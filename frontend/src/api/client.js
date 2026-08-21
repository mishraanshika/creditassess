/**
 * Thin API client.
 *
 * In dev, Vite proxies `/api/*` to the FastAPI server on :8000, so the app is
 * same-origin and there is no CORS surface. In production set VITE_API_BASE to
 * the gateway URL.
 */
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  const text = await res.text()
  let body
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = { detail: text }
  }
  if (!res.ok) {
    const message = body?.detail ?? `Request failed (${res.status})`
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message))
  }
  return body
}

const post = (path, payload) =>
  request(path, { method: 'POST', body: JSON.stringify(payload) })

export const api = {
  health: () => request('/health'),
  predict: (applicant, opts = {}) =>
    post('/predict', { applicant, top_k: 5, include_explanation: true, include_similar: true, ...opts }),
  similarBorrowers: (applicant, topK = 5) =>
    post('/similar-borrowers', { applicant, top_k: topK }),
  underwritingReport: (applicant, tone = 'credit_committee', topK = 5) =>
    post('/underwriting-report', { applicant, tone, top_k: topK }),
  modelMetrics: () => request('/analytics/model-metrics'),
  featureImportance: (topK = 18) => request(`/analytics/feature-importance?top_k=${topK}`),
  bias: () => request('/analytics/bias'),
  policy: () => request('/analytics/policy'),
  portfolio: () => request('/analytics/portfolio'),
  auditLog: (limit = 40) => request(`/analytics/audit-log?limit=${limit}`),
  reviewQueue: () => request('/analytics/review-queue'),
}

/** Formatting helpers shared by every page. */
export const fmt = {
  money: (v) =>
    v === null || v === undefined ? '—' :
      new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(v),
  pct: (v, digits = 1) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`),
  num: (v, digits = 1) => (v === null || v === undefined ? '—' : Number(v).toFixed(digits)),
  date: (v) => (v ? new Date(v).toLocaleString() : '—'),
}

export const toneOfRecommendation = (reco) =>
  reco === 'APPROVE' ? 'approve' : reco === 'REJECT' ? 'reject' : 'review'
