import io
import os
import re
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Resume Matcher", page_icon="🧠", layout="wide")
MODEL_PATH = r"D:\materials\AI track\materials\DEEP LEARNING\NLP\resume(cv)\ResumeMatch-Pro-File-upload-Resume-Job-Matcher-SentenceTransformers-Streamlit-\FIne_tunning_my_sentence_model"  # e.g., r"D:\models\my_sentence_model_final" for local runs
HF_MODEL_ID = "youssefeladl/my-sentence-model"  # Your Hugging Face model id

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# ===================== Role → Skills Dictionary =====================
job_skills: Dict[str, List[str]] = {
    # ===================== Software & Data =====================
    "Software Engineer": [
        "python", "java", "c#", "c++", "go", "javascript", "typescript",
        "object oriented programming", "design patterns", "data structures",
        "algorithms", "rest api", "graphql", "microservices",
        "sql", "nosql", "git", "unit testing", "integration testing",
        "system design", "linux", "docker", "kubernetes", "ci/cd"
    ],
    "Backend Developer": [
        "python", "django", "flask", "fastapi", "nodejs", "express",
        "java", "spring boot", "php", "laravel", "ruby", "rails",
        "rest api", "graphql", "mysql", "postgresql", "mongodb",
        "redis", "celery", "docker", "kubernetes", "testing"
    ],
    "Frontend Developer": [
        "html", "css", "javascript", "typescript", "react", "redux",
        "next.js", "vue", "nuxt", "angular", "webpack",
        "responsive design", "ui/ux", "accessibility", "testing"
    ],
    "Full Stack Developer": [
        "react", "angular", "vue", "next.js", "nodejs", "express",
        "python", "django", "flask", "php", "laravel",
        "mysql", "postgresql", "mongodb", "graphql", "rest api",
        "git", "docker", "ci/cd", "testing"
    ],
    "DevOps Engineer": [
        "linux", "bash", "docker", "kubernetes", "helm",
        "jenkins", "github actions", "gitlab ci", "ci/cd",
        "terraform", "ansible", "aws", "azure", "gcp",
        "prometheus", "grafana", "nginx", "monitoring", "sre"
    ],
    "Cloud Engineer": [
        "aws", "azure", "gcp", "ec2", "s3", "iam", "vpc",
        "cloud networking", "load balancer", "terraform",
        "cloud formation", "kubectl", "docker", "linux", "monitoring"
    ],
    "Cybersecurity Analyst": [
        "network security", "siem", "soc", "ids/ips", "firewall",
        "incident response", "vulnerability assessment", "threat hunting",
        "wireshark", "nmap", "kali linux", "iso 27001", "gdpr"
    ],
    "Data Scientist": [
        "python", "r", "sql", "pandas", "numpy", "scikit-learn",
        "statistics", "probability", "machine learning", "deep learning",
        "natural language processing", "computer vision",
        "matplotlib", "seaborn", "pytorch", "tensorflow", "mlflow"
    ],
    "Data Analyst": [
        "excel", "sql", "tableau", "power bi", "python",
        "pandas", "statistics", "data cleaning", "etl",
        "dashboard", "a/b testing", "reporting"
    ],
    "Machine Learning Engineer": [
        "python", "pytorch", "tensorflow", "scikit-learn",
        "feature engineering", "model serving", "fastapi",
        "docker", "kubernetes", "mlops", "airflow", "mlflow",
        "monitoring", "experiment tracking"
    ],
    "QA / Test Engineer": [
        "manual testing", "automation testing", "selenium", "cypress",
        "pytest", "junit", "api testing", "postman",
        "test cases", "bug tracking", "jira", "performance testing"
    ],
    "Android Developer": [
        "kotlin", "java", "android sdk", "jetpack", "room",
        "retrofit", "material design", "firebase", "rest api"
    ],
    "iOS Developer": [
        "swift", "objective-c", "xcode", "cocoa touch",
        "uikit", "swiftui", "core data", "firebase", "rest api"
    ],
    "Database Administrator": [
        "sql", "mysql", "postgresql", "oracle", "sql server", "db2",
        "backup", "replication", "indexing", "query optimization",
        "normalization", "er diagrams", "high availability"
    ],

    # ===================== Finance & Business =====================
    "Financial Analyst": [
        "excel", "financial modeling", "valuation", "dcf",
        "budgeting", "forecasting", "kpis", "variance analysis",
        "accounting principles", "power bi", "tableau", "sql"
    ],
    "Accountant": [
        "general ledger", "ap", "ar", "reconciliation",
        "ifrs", "gaap", "tax", "vat", "payroll",
        "excel", "sap", "oracle erp", "quickbooks"
    ],
    "Auditor": [
        "audit planning", "internal controls", "risk assessment",
        "ifrs", "gaap", "sampling", "substantive testing",
        "workpapers", "reporting"
    ],
    "Investment Analyst": [
        "equity research", "financial modeling", "valuation",
        "portfolio analysis", "bloomberg", "excel", "powerpoint"
    ],
    "Business Analyst": [
        "requirements gathering", "process mapping", "sql",
        "excel", "stakeholder management", "uml",
        "agile", "jira", "documentation", "kpis"
    ],
    "Project Manager (IT)": [
        "project management", "agile", "scrum", "kanban",
        "planning", "risk management", "budgeting", "stakeholders",
        "jira", "confluence", "reporting"
    ],
    "Product Manager": [
        "product roadmap", "requirements", "user research",
        "wireframing", "analytics", "kpis", "go-to-market",
        "agile", "scrum", "backlog"
    ],
    "Procurement Specialist": [
        "procurement", "vendor management", "rfq", "rfi", "tenders",
        "negotiation", "contracts", "sap", "oracle erp"
    ],
    "Supply Chain Analyst": [
        "demand planning", "forecasting", "inventory management",
        "logistics", "excel", "power bi", "sql", "sap"
    ],
    "Operations Manager": [
        "process improvement", "kpis", "sop", "leadership",
        "capacity planning", "resource allocation", "reporting"
    ],
    "Sales Executive": [
        "lead generation", "crm", "negotiation", "pipeline management",
        "cold calling", "presentation", "closing", "reporting"
    ],
    "Digital Marketing Specialist": [
        "seo", "sem", "content marketing", "social media",
        "google ads", "facebook ads", "ppc", "email marketing",
        "google analytics", "tag manager"
    ],
    "HR Generalist": [
        "recruitment", "onboarding", "employee relations",
        "payroll", "performance management", "hr policies",
        "labor law", "training", "hris"
    ],
    "Recruiter": [
        "sourcing", "screening", "interviewing", " ats ",
        "stakeholder management", "offer negotiation", "reporting"
    ],
    "Customer Support Specialist": [
        "customer service", "ticketing", "sla", "crm",
        "communication", "troubleshooting", "knowledge base"
    ],

    # ===================== Engineering (Non-Software) =====================
    "Civil Engineer": [
        "autocad", "etabs", "sap2000", "staad", "revit",
        "structural design", "quantity surveying", "shop drawings",
        "site supervision", "project management", "health and safety", "qaqc"
    ],
    "Mechanical Engineer": [
        "solidworks", "autocad", "ansys", "hvac", "piping",
        "manufacturing", "maintenance", "thermodynamics",
        "root cause analysis", "preventive maintenance"
    ],
    "Electrical Engineer": [
        "autocad", "revit", "power systems", "mv/lv",
        "load calculation", "protection", "plc", "scada",
        "single line diagram", "lighting design"
    ],
    "Mechatronics Engineer": [
        "embedded systems", "arduino", "stm32", "c", "c++",
        "plc", "robotics", "sensors", "actuators", "control"
    ],
    "Industrial Engineer": [
        "lean manufacturing", "six sigma", "process improvement",
        "time study", "value stream mapping", "kpis", "sap"
    ],
    "Quality Engineer": [
        "qaqc", "iso 9001", "spc", "root cause analysis",
        "8d", "fmea", "ppap", "msa", "corrective actions"
    ],
    "Chemical Engineer": [
        "process design", "pfd", "p&id", "hysys",
        "safety", "quality control", "unit operations"
    ],

    # ===================== Healthcare & Pharma =====================
    "Pharmacist": [
        "dispensing", "drug interactions", "dosage", "counseling",
        "inventory management", "pharmacovigilance", "prescriptions"
    ],
    "Clinical Pharmacist": [
        "medication therapy management", "clinical guidelines",
        "drug monitoring", "patient counseling", "ward rounds"
    ],
    "Medical Representative": [
        "product knowledge", "territory management", "kols",
        "sales", "presentations", "reporting", "crm"
    ],
    "Biomedical Engineer": [
        "medical devices", "maintenance", "calibration",
        "installation", "troubleshooting", "documentation"
    ],
    "Lab Technician": [
        "sample collection", "microbiology", "hematology",
        "biochemistry", "quality control", "lab safety", "lims"
    ],

    # ===================== Creative & Content =====================
    "UX/UI Designer": [
        "user research", "wireframes", "prototyping",
        "figma", "sketch", "adobe xd", "usability testing", "design systems"
    ],
    "Graphic Designer": [
        "photoshop", "illustrator", "indesign",
        "branding", "layout", "typography", "social media design"
    ],
    "Content Writer": [
        "copywriting", "seo writing", "editing", "proofreading",
        "content strategy", "research", "cms", "wordpress"
    ],
    "Social Media Manager": [
        "content calendar", "community management", "ads manager",
        "analytics", "engagement", "branding", "reporting"
    ],

    # ===================== Education & Admin =====================
    "Teacher": [
        "lesson planning", "classroom management", "assessment",
        "curriculum design", "communication", "student engagement"
    ],
    "Teaching Assistant": [
        "class support", "grading", "lesson assistance",
        "student supervision", "materials preparation"
    ],
    "Data Entry Specialist": [
        "typing", "excel", "accuracy", "attention to detail",
        "data cleaning", "erp", "reporting"
    ]
}
skill_aliases: Dict[str, str] = {
    "ms sql": "sql server",
    "mssql": "sql server",
    "postgres": "postgresql",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "ci cd": "ci/cd",
    "ci-cd": "ci/cd",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ml": "machine learning",
    "dl": "deep learning",
    "pm": "project management",
    "ga": "google analytics"
}

BASE_SKILLS = set({s for lst in job_skills.values() for s in lst})
cert_keywords = [
    "certified", "certification", "aws", "microsoft", "pmp", "oracle",
    "cisco", "google cloud", "azure", "coursera", "udemy", "ibm",
    "data science certificate", "machine learning certification",
    "deep learning certification", "nlp certification", "tensorflow certification",
    "kaggle competition", "professional certificate", "sap certification"
]

project_keywords = [
    "project", "developed", "built", "implemented", "designed",
    "case study", "application", "system", "automation", "platform",
    "dashboard", "model", "pipeline", "analytics", "framework",
    "deployment", "prototype", "algorithm", "solution", "tool"
]

# Base skill set for raw scan (union of all role skills)
BASE_SKILLS = set({s for lst in job_skills.values() for s in lst})

# ===================== Model Loading (local or HF) =====================
@st.cache_resource(show_spinner=False)
def load_model():
    # A) Local folder (for local runs only)
    if MODEL_PATH and os.path.isdir(MODEL_PATH):
        try:
            return SentenceTransformer(MODEL_PATH)
        except Exception:
            from sentence_transformers import models as st_models
            tx = os.path.join(MODEL_PATH, "0_Transformer")
            bert = st_models.Transformer(tx if os.path.isdir(tx) else MODEL_PATH)
            pooling = st_models.Pooling(bert.get_word_embedding_dimension())
            normalize = st_models.Normalize()
            return SentenceTransformer(modules=[bert, pooling, normalize])

    # B) No local folder => load from Hugging Face Hub
    repo_id = (HF_MODEL_ID or "").strip()
    if not repo_id:
        st.error("No local model folder found and no HF model id provided.")
        st.stop()

    # If the model is private, set HF_TOKEN in Streamlit Cloud Secrets
    hf_token = os.environ.get("HF_TOKEN")
    return SentenceTransformer(repo_id, token=hf_token) if hf_token else SentenceTransformer(repo_id)

model = load_model()

# ===================== Utilities =====================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def compute_match(resume_text: str, job_text: str) -> float:
    embs = model.encode([resume_text, job_text], convert_to_numpy=True, normalize_embeddings=True)
    cos = float(np.dot(embs[0], embs[1]))     # cosine in [-1, 1]
    score = (cos + 1.0) / 2.0                 # -> [0, 1]
    return max(0.0, min(1.0, score))

def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def read_pdf(file) -> str:
    if pdfplumber is None:
        st.warning("pdfplumber not installed.")
        return ""
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def read_docx(file_bytes: bytes) -> str:
    if docx is None:
        st.warning("python-docx not installed.")
        return ""
    bio = io.BytesIO(file_bytes)
    document = docx.Document(bio)
    return "\n".join(p.text for p in document.paragraphs)

def extract_text_from_upload(uploaded_file) -> Tuple[str, str]:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".txt"):
        return read_txt(data), "txt"
    elif name.endswith(".pdf"):
        return read_pdf(io.BytesIO(data)), "pdf"
    elif name.endswith(".docx"):
        return read_docx(data), "docx"
    else:
        return read_txt(data), "unknown"

def find_keywords(text: str, words: List[str]) -> List[str]:
    low = text.lower()
    found = []
    for w in sorted(set(words)):
        pat = r"(?<![a-z0-9+#])" + re.escape(w.lower()) + r"(?![a-z0-9+#])"
        if re.search(pat, low):
            found.append(w)
    return found

def simple_skill_scan(text: str, extra: List[str] = None) -> List[str]:
    skills = set(BASE_SKILLS)
    if extra:
        skills |= {s.strip().lower() for s in extra if s.strip()}
    return find_keywords(text, list(skills))

def extract_experience_phrases(text: str) -> List[str]:
    pats = [
        r"\b(\d{1,2})\s*\+?\s*years?\b",
        r"\b(\d{1,2})\s*\+?\s*yrs?\b",
        r"\bexperience\s*[:\-\s]+(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b",
    ]
    hits = []
    for p in pats:
        hits += [m.group(0) for m in re.finditer(p, text, flags=re.IGNORECASE)]
    out, seen = [], set()
    for h in hits:
        k = h.lower()
        if k not in seen:
            out.append(h)
            seen.add(k)
    return out

def rank_roles_by_overlap(resume_skills: List[str]) -> List[Tuple[str, float, int, int]]:
    rs = set(s.lower() for s in resume_skills)
    ranking = []
    for role, skills in job_skills.items():
        role_set = set(s.lower() for s in skills)
        matched = rs & role_set
        coverage = len(matched) / max(1, len(role_set))
        ranking.append((role, coverage, len(matched), len(role_set)))
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

# ===================== UI =====================
st.title("🧠 Resume ↔ Job Match")
st.caption("Upload a resume file and paste a job description to get a match score and skills breakdown.")

with st.sidebar:
    st.subheader("Settings")
    st.write("Model source:")
    st.write("Local folder exists:", bool(MODEL_PATH and os.path.isdir(MODEL_PATH)))
    st.code(f"HF_MODEL_ID = {HF_MODEL_ID}")
    extra_skills_str = st.text_area("Extra skills (comma-separated)", placeholder="e.g., Supabase, Flutter, FastAPI, Power BI")
    selected_role = st.selectbox("Target role (optional)", ["(auto-detect top roles)"] + list(job_skills.keys()))

left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload Resume (.pdf / .docx / .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=False)
    st.caption("If the PDF is a scanned image, consider OCR or exporting to DOCX/TXT.")
with right:
    job_desc = st.text_area("Job Description / Requirements", height=220, placeholder="Paste job description here...")

analyze = st.button("Analyze Match")

if analyze:
    if not uploaded:
        st.error("Please upload a resume file.")
        st.stop()
    if not job_desc.strip():
        st.error("Please paste a job description.")
        st.stop()

    with st.spinner("Extracting text and computing embeddings…"):
        resume_text, ftype = extract_text_from_upload(uploaded)
        rt = normalize_text(resume_text)
        jt = normalize_text(job_desc)

        if len(rt) < 50:
            st.warning("Extracted resume text looks very short. If it's a scanned PDF, use OCR or export to DOCX/TXT.")

        score = compute_match(rt, jt)

        extra_list = [s.strip() for s in (extra_skills_str or "").split(",") if s.strip()]
        resume_skill_list = simple_skill_scan(rt, extra_list)
        jd_skill_list = simple_skill_scan(jt, extra_list)
        overlap = sorted(set(resume_skill_list) & set(jd_skill_list))

        role_ranking = rank_roles_by_overlap(resume_skill_list)
        years_phrases = extract_experience_phrases(rt)
        certs_found = find_keywords(rt, cert_keywords)
        projects_found = find_keywords(rt, project_keywords)

    # Theme-safe metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Match Score", f"{round(score*100)}%")
    with m2:
        st.metric("Resume File Type", ftype.upper())
    with m3:
        st.metric("JD Skills Found", len(jd_skill_list))
    with m4:
        st.metric("Resume Skills Found", len(resume_skill_list))
    st.progress(score)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Matched Skills (Resume ∩ JD)")
        st.write(", ".join(sorted(overlap)) if overlap else "—")

        st.markdown("### Resume Skills")
        st.write(", ".join(sorted(resume_skill_list)) if resume_skill_list else "—")

        st.markdown("### Experience Phrases")
        st.write("; ".join(years_phrases) if years_phrases else "—")

        st.markdown("### Certifications Mentioned")
        st.write(", ".join(sorted(set(certs_found))) if certs_found else "—")

        st.markdown("### Project Keywords Mentioned")
        st.write(", ".join(sorted(set(projects_found))) if projects_found else "—")

    with c2:
        st.markdown("### JD Skills")
        st.write(", ".join(sorted(jd_skill_list)) if jd_skill_list else "—")

        st.markdown("### Role Match (by dictionary coverage)")
        if selected_role != "(auto-detect top roles)":
            rset = set(s.lower() for s in resume_skill_list)
            role_set = set(s.lower() for s in job_skills[selected_role])
            matched = sorted(rset & role_set)
            missing = sorted(role_set - rset)
            coverage = len(matched) / max(1, len(role_set))
            st.write(f"**{selected_role}** — Coverage: {coverage*100:.1f}% ({len(matched)}/{len(role_set)})")
            st.write("**Matched:** " + (", ".join(matched) if matched else "—"))
            st.write("**Missing:** " + (", ".join(missing) if missing else "—"))
        else:
            top = role_ranking[:5]
            if not top:
                st.write("—")
            else:
                for role, cov, m, tot in top:
                    st.write(f"**{role}** — Coverage: {cov*100:.1f}%  ({m}/{tot})")

    with st.expander("Debug: First 1200 chars of extracted resume"):
        st.code(rt[:1200] + ("…" if len(rt) > 1200 else ""))

else:
    st.info("Upload a resume file and paste a job description, then click **Analyze Match**.")
