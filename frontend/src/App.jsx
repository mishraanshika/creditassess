import { useEffect } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { api } from './api/client'
import { useAppState } from './state/AppState'
import Dashboard from './pages/Dashboard'
import ApplicantForm from './pages/ApplicantForm'
import RiskAssessment from './pages/RiskAssessment'
import SimilarBorrowers from './pages/SimilarBorrowers'
import UnderwritingReport from './pages/UnderwritingReport'
import Analytics from './pages/Analytics'

const NAV = [
  { section: 'Underwriting' },
  { to: '/', label: 'Overview', icon: '◈', end: true },
  { to: '/apply', label: 'Applicant Intake', icon: '✎' },
  { to: '/risk', label: 'Risk Assessment', icon: '◎' },
  { section: 'Evidence' },
  { to: '/similar', label: 'Similar Borrowers', icon: '⧉' },
  { to: '/report', label: 'Underwriting Report', icon: '✦' },
  { section: 'Governance' },
  { to: '/analytics', label: 'Analytics & Fairness', icon: '▤' },
]

export default function App() {
  const { health, setHealth } = useAppState()

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'unreachable' }))
  }, [setHealth])

  const dot =
    health?.status === 'healthy' ? 'var(--ok)'
      : health?.status === 'degraded' ? 'var(--warn)' : 'var(--danger)'

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">CA</div>
          <div>
            <div className="brand-name">CreditAssess</div>
            <div className="brand-sub">NTC UNDERWRITING ENGINE</div>
          </div>
        </div>

        {NAV.map((item, i) =>
          item.section ? (
            <div className="nav-section" key={`s-${i}`}>{item.section}</div>
          ) : (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          )
        )}

        <div style={{ marginTop: 'auto' }} className="card" >
          <div className="row" style={{ gap: 8 }}>
            <span style={{ width: 8, height: 8, borderRadius: 999, background: dot }} />
            <span style={{ fontSize: 12, fontWeight: 600 }}>
              {health?.status === 'healthy' ? 'System ready'
                : health?.status === 'degraded' ? 'Running with limits'
                : health ? 'Server unreachable' : 'Checking…'}
            </span>
          </div>
          <div className="hint" style={{ marginTop: 8, lineHeight: 1.6 }}>
            {health?.status === 'healthy'
              ? `model ${health?.model_version ?? '—'} · ${health?.vector_size ?? 0} vectors`
              : 'Degraded — see subsystem status on Overview.'}
          </div>
        </div>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/apply" element={<ApplicantForm />} />
          <Route path="/risk" element={<RiskAssessment />} />
          <Route path="/similar" element={<SimilarBorrowers />} />
          <Route path="/report" element={<UnderwritingReport />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
