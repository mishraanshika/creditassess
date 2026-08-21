import { createContext, useContext, useMemo, useState } from 'react'

/**
 * Cross-page state.
 *
 * The Applicant Form produces one `applicant` payload; Risk Assessment,
 * Similar Borrowers and the AI Report all read the SAME assessment object so
 * the four screens can never show three different decisions for one person.
 */
const AppStateContext = createContext(null)

/** A realistic thin-file applicant: salaried, no bureau score, strong telco signals. */
export const SAMPLE_APPLICANTS = {
  ntc_strong: {
    label: 'NTC · strong behaviour',
    applicant: {
      full_name: 'Aarti Deshmukh',
      external_ref: 'NTC-1001',
      AMT_INCOME_TOTAL: 540000,
      AMT_CREDIT: 300000,
      AMT_ANNUITY: 96000,
      AMT_GOODS_PRICE: 285000,
      age_years: 31,
      employment_years: 5.5,
      NAME_CONTRACT_TYPE: 'Cash loans',
      NAME_INCOME_TYPE: 'Working',
      NAME_EDUCATION_TYPE: 'Higher education',
      NAME_FAMILY_STATUS: 'Married',
      NAME_HOUSING_TYPE: 'House / apartment',
      OCCUPATION_TYPE: 'Core staff',
      ORGANIZATION_TYPE: 'Business Entity Type 3',
      FLAG_OWN_CAR: 'N',
      FLAG_OWN_REALTY: 'Y',
      CNT_CHILDREN: 0,
      CNT_FAM_MEMBERS: 2,
      FLAG_MOBIL: 1, FLAG_EMP_PHONE: 1, FLAG_WORK_PHONE: 1,
      FLAG_CONT_MOBILE: 1, FLAG_PHONE: 1, FLAG_EMAIL: 1,
      months_on_current_handset: 42,
      documents_submitted: 4,
    },
  },
  thin_stretched: {
    label: 'Thin file · stretched',
    applicant: {
      full_name: 'Rohit Nair',
      external_ref: 'NTC-1002',
      AMT_INCOME_TOTAL: 180000,
      AMT_CREDIT: 900000,
      AMT_ANNUITY: 78000,
      AMT_GOODS_PRICE: 500000,
      age_years: 24,
      employment_years: 0.4,
      NAME_INCOME_TYPE: 'Working',
      NAME_EDUCATION_TYPE: 'Secondary / secondary special',
      NAME_FAMILY_STATUS: 'Single / not married',
      NAME_HOUSING_TYPE: 'Rented apartment',
      OCCUPATION_TYPE: 'Laborers',
      ORGANIZATION_TYPE: 'Self-employed',
      FLAG_OWN_CAR: 'N',
      FLAG_OWN_REALTY: 'N',
      CNT_CHILDREN: 1,
      CNT_FAM_MEMBERS: 3,
      FLAG_MOBIL: 1, FLAG_EMP_PHONE: 0, FLAG_WORK_PHONE: 0,
      FLAG_CONT_MOBILE: 1, FLAG_PHONE: 0, FLAG_EMAIL: 0,
      months_on_current_handset: 1,
      documents_submitted: 0,
      AMT_REQ_CREDIT_BUREAU_QRT: 3,
      AMT_REQ_CREDIT_BUREAU_YEAR: 9,
    },
  },
  established: {
    label: 'Established file',
    applicant: {
      full_name: 'Priya Raghavan',
      external_ref: 'STD-2001',
      AMT_INCOME_TOTAL: 810000,
      AMT_CREDIT: 640000,
      AMT_ANNUITY: 120000,
      AMT_GOODS_PRICE: 640000,
      age_years: 42,
      employment_years: 12,
      NAME_INCOME_TYPE: 'State servant',
      NAME_EDUCATION_TYPE: 'Higher education',
      NAME_FAMILY_STATUS: 'Married',
      NAME_HOUSING_TYPE: 'House / apartment',
      OCCUPATION_TYPE: 'Managers',
      ORGANIZATION_TYPE: 'School',
      FLAG_OWN_CAR: 'Y',
      FLAG_OWN_REALTY: 'Y',
      CNT_CHILDREN: 2,
      CNT_FAM_MEMBERS: 4,
      FLAG_MOBIL: 1, FLAG_EMP_PHONE: 1, FLAG_WORK_PHONE: 1,
      FLAG_CONT_MOBILE: 1, FLAG_PHONE: 1, FLAG_EMAIL: 1,
      months_on_current_handset: 30,
      documents_submitted: 5,
      EXT_SOURCE_2: 0.71,
      EXT_SOURCE_3: 0.66,
    },
  },
}

export function AppStateProvider({ children }) {
  const [applicant, setApplicant] = useState(SAMPLE_APPLICANTS.ntc_strong.applicant)
  const [assessment, setAssessment] = useState(null)
  const [report, setReport] = useState(null)
  const [health, setHealth] = useState(null)

  const value = useMemo(
    () => ({
      applicant, setApplicant,
      assessment, setAssessment,
      report, setReport,
      health, setHealth,
      // A new applicant invalidates any assessment and report already on screen.
      resetResults: () => { setAssessment(null); setReport(null) },
    }),
    [applicant, assessment, report, health]
  )

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>
}

export function useAppState() {
  const ctx = useContext(AppStateContext)
  if (!ctx) throw new Error('useAppState must be used inside AppStateProvider')
  return ctx
}
