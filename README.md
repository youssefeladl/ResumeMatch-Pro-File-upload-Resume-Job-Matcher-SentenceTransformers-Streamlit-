# ResumeMatch Pro — Resume ⇄ Job Matcher (SentenceTransformers + Streamlit)

https://kwnhsbhypny6jhgztfjaed.streamlit.app/

Smart, explainable matching between a **Resume** (PDF/DOCX/TXT) and a **Job Description** using your **fine-tuned SentenceTransformers** model.  
Shows:
- **Overall Semantic Match** (embeddings cosine)
- **Role Coverage** from a curated **role → skills** dictionary
- **Matched / Missing skills**, **experience cues**, and **cert/project signals**

> Works fully offline with a local model folder. No cloud calls required.

---

## Table of Contents

- [Highlights](#highlights)
- [What You Get](#what-you-get)
- [How It Works](#how-it-works)
- [Scoring Explained](#scoring-explained)
- [Model Setup (No CLI)](#model-setup-no-cli)
- [App Configuration (No CLI)](#app-configuration-no-cli)
- [Skills Dictionary](#skills-dictionary)
- [Adjusting the Final Score (Optional)](#adjusting-the-final-score-optional)
- [Project Structure](#project-structure)
- [Limitations & Tips](#limitations--tips)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [License](#license)

---

## Highlights

- **File upload UX**: Drop a PDF/DOCX/TXT; no manual copy-paste needed for resumes.
- **Semantic + Symbolic**: Combines **embeddings similarity** with **skills coverage** per role.
- **Explainable**: See matched/missing skills, experience phrases, and signals (certs/projects).
- **Local model**: Point to your **fine-tuned ST** folder; nothing leaves your machine.
- **Theme-safe UI**: Clean metrics that look right in dark/light modes.

---

## What You Get

- `cv.py` — the Streamlit app file.
- A **role → skills** dictionary inside `cv.py` you can edit anytime.
- A single page app:
  - Left: upload resume file, optional extra skills, optional target role.
  - Right: paste JD text and click **Analyze Match**.
  - Summary metrics + detailed breakdowns.

---

## How It Works

1. **Resume Parsing**  
   - If you upload **PDF**/**DOCX**/**TXT**, the app extracts text locally.
   - If a PDF is a scan (images), extraction can be weak. Prefer DOCX/TXT export.

2. **Embeddings**  
   - The app loads your **fine-tuned SentenceTransformers** model from a **folder path**.
   - It encodes the **resume text** and the **JD text**, uses cosine similarity, and normalizes to a 0..1 score.

3. **Skills Coverage**  
   - The app scans both resume and JD for skills (keywords).
   - It also compares resume skills against the selected role’s skills list (or auto-ranks top roles by coverage).

4. **Signals**  
   - Extracts simple **experience phrases** (e.g., “3+ years”).
   - Detects **certification** and **project** keywords for quick hints.

---

## Scoring Explained

- **Overall Semantic Match**  
  A semantic similarity between resume and JD texts using embeddings from your fine-tuned ST model.  
  Reflects **context + wording** beyond mere keywords.

- **Role Coverage**  
  From the dictionary: “What percentage of the role’s skills appear in the resume?”  
  Pure keyword coverage; great for **tech core skills**.

> Numbers can differ (e.g., 71% vs 78%) because they measure **different dimensions**.  
> Use both for a fuller picture, or combine them into a single final score (see below).

---
