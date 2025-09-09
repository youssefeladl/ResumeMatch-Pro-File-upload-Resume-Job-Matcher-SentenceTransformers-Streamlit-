
import io
import os
import re
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
st.set_page_config(page_title="Resume Matcher", page_icon="🧠", layout="wide")
MODEL_PATH = r"D:\materials\AI track\materials\DEEP LEARNING\NLP\resume(cv)\ResumeMatch-Pro-File-upload-Resume-Job-Matcher-SentenceTransformers-Streamlit-\FIne_tunning_my_sentence_model"

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

job_skills: Dict[str, List[str]] = {

    "Django Developer": [
        "python", "django", "flask", "fastapi", "rest", "rest api", "drf",
        "web development", "html", "css", "javascript", "bootstrap",
        "mysql", "postgresql", "sqlite", "mongodb",
        "orm", "django orm", "celery", "redis", "docker", "nginx", "gunicorn",
        "unit testing", "pytest", "github", "git"
    ],
    "Java Developer": [
        "java", "spring", "spring boot", "hibernate", "maven", "gradle",
        "jsp", "servlets", "junit", "apache tomcat",
        "sql", "oracle", "mysql", "postgresql",
        "microservices", "kafka", "rabbitmq", "docker", "kubernetes",
        "rest api", "oop", "design patterns"
    ],
    "DevOps Engineer": [
        "devops", "linux", "bash", "shell scripting", "docker", "kubernetes",
        "aws", "azure", "gcp", "terraform", "ansible", "jenkins",
        "git", "github actions", "ci/cd", "prometheus", "grafana",
        "nginx", "apache", "load balancing", "scalability", "monitoring"
    ],
    "Full Stack Developer": [
        "html", "css", "javascript", "typescript", "react", "angular", "vue",
        "nodejs", "express", "django", "flask", "php", "laravel",
        "java", "spring boot", "c#", ".net", "mysql", "postgresql", "mongodb",
        "graphql", "rest api", "git", "docker"
    ],
    "iOS Developer": [
        "swift", "objective-c", "xcode", "cocoa touch", "ios sdk",
        "ui kit", "core data", "core animation", "firebase",
        "rest api", "json", "git"
    ],
    "Flutter Developer": [
        "flutter", "dart", "android", "ios", "firebase", "bloc", "riverpod",
        "rest api", "json", "sqlite", "push notifications", "ui/ux"
    ],
    "Database Administrator": [
        "sql", "mysql", "postgresql", "oracle", "sql server", "db2",
        "mongodb", "cassandra", "nosql", "pl/sql", "database design",
        "indexing", "query optimization", "backup", "replication",
        "normalization", "er diagrams"
    ],
    "Node.js Developer": [
        "nodejs", "express", "javascript", "typescript", "npm", "yarn",
        "mongodb", "mysql", "postgresql", "graphql", "rest api",
        "jwt", "oauth", "docker", "microservices"
    ],
    "Software Engineer": [
        "python", "java", "c++", "c", "git", "github", "algorithms",
        "data structures", "oop", "design patterns", "testing",
        "unit testing", "system design", "sql", "nosql"
    ],
    "Wordpress Developer": [
        "wordpress", "php", "html", "css", "javascript", "woocommerce",
        "themes", "plugins", "seo", "mysql", "elementor"
    ],
    "PHP Developer": [
        "php", "laravel", "symfony", "codeigniter", "cakephp",
        "mysql", "postgresql", "javascript", "jquery", "ajax",
        "html", "css", "git"
    ],
    "Backend Developer": [
        "nodejs", "express", "django", "flask", "java", "spring boot",
        "php", "laravel", "ruby", "rails", "mysql", "postgresql", "mongodb",
        "graphql", "rest api", "docker"
    ],
    "Network Administrator": [
        "networking", "router", "switch", "firewall", "dns", "dhcp",
        "vpn", "tcp/ip", "lan", "wan", "cisco", "load balancing",
        "troubleshooting", "wireshark"
    ],
    "Blockchain Developer": [
        "blockchain", "ethereum", "solidity", "smart contracts",
        "web3", "defi", "nft", "bitcoin", "cryptography",
        "consensus", "hyperledger", "rust", "go", "truffle", "ganache"
    ],
    "Data Scientist": [
        "python","sql", "pandas", "numpy", "scikit-learn",
        "tensorflow", "keras", "statistics", "probability",
        "machine learning", "deep learning", "nlp", "computer vision",
        "matplotlib", "seaborn","nlp","opencv", "cnn",
        "transformers"
    ],
    "Business Analyst": [
        "excel", "sql", "tableau", "powerbi", "python", "r",
        "business analysis", "requirement gathering", "uml", "data modeling",
        "statistics", "communication", "project management", "agile", "jira"
    ],
    "Testing / QA Engineer": [
        "manual testing", "automation testing", "selenium", "junit",
        "pytest", "cypress", "api testing", "postman",
        "test cases", "bug tracking", "jira", "agile", "unit testing"
    ]
}

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

BASE_SKILLS = set({s for lst in job_skills.values() for s in lst})
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return SentenceTransformer(MODEL_PATH)
    except Exception:
        # Fallback: build ST from HF folder if needed
        from sentence_transformers import models as st_models
        tx = os.path.join(MODEL_PATH, "0_Transformer")
        bert = st_models.Transformer(tx if os.path.isdir(tx) else MODEL_PATH)
        pooling = st_models.Pooling(bert.get_word_embedding_dimension())
        normalize = st_models.Normalize()
        return SentenceTransformer(modules=[bert, pooling, normalize])

model = load_model()

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
        st.warning("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def read_docx(file_bytes: bytes) -> str:
    if docx is None:
        st.warning("python-docx not installed. Run: pip install python-docx")
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

st.title("🧠 Resume ↔ Job Match (Fine-tuned)")
st.caption("Upload a resume file and paste a job description to get a match score and skills breakdown.")

with st.sidebar:
    st.subheader("Settings")
    st.write("Model Path:")
    st.code(MODEL_PATH)
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
