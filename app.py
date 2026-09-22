import io
import re
from typing import Optional

import streamlit as st
from groq import Groq
from pypdf import PdfReader


MODEL_NAME = "qwen/qwen3.8-27b"


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract selectable text from a PDF entirely in memory."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    return "\n\n".join(pages).strip()


def extract_txt_text(file_bytes: bytes) -> str:
    """Decode a TXT upload safely."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace").strip()


def normalize_text(text: str) -> str:
    """Clean excessive whitespace without changing report wording."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def limit_report_text(text: str, max_chars: int = 90000) -> tuple[str, bool]:
    """
    Keep a practical request size for the LLM.
    The report is truncated only at a character boundary when necessary.
    """
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def build_medical_prompt(report_text: str, was_truncated: bool) -> str:
    truncation_note = (
        "The application had to truncate the uploaded report because it was very long. "
        "Analyze only the supplied text and do not assume anything about omitted content."
        if was_truncated
        else
        "The complete extractable report text supplied by the application is below."
    )

    return f"""
You are a medical-report explanation assistant, NOT a doctor and NOT a diagnostic engine.

Your task is to explain the uploaded CT/radiology report to a person with little or no medical knowledge.

STRICT SAFETY AND GROUNDING RULES:
1. Stay grounded in the report. Use only information found in the supplied report plus general explanations of medical terminology.
2. Never invent symptoms, diagnoses, measurements, diseases, medications, medical history, age, sex, treatment plans, test results, or radiologist recommendations.
3. If requested information is absent, say exactly: "Not stated in the report."
4. Use extremely simple language. Prefer short sentences and common words.
5. Explain important medical terms in plain language.
6. Never diagnose. Do not say "you have cancer", "you definitely have pneumonia", "you have a tumor", or similar definitive statements.
7. When a report describes a finding, explain what the report says and what the term generally means. Make clear that interpretation depends on the treating clinician, symptoms, examination, history, and images.
8. Preserve uncertainty words such as possible, suspicious, likely, probable, cannot exclude, may represent, suggestive of, recommend, correlate clinically, follow-up, and incidental. Explain their meaning without increasing certainty.
9. Do not minimize potentially important findings and do not provide false reassurance.
10. Do not create emergency situations merely because a finding sounds unfamiliar. Mention urgent care only when the report itself indicates urgency OR when a described finding could reasonably represent a time-sensitive situation and cautious general safety guidance is appropriate.
11. Do not prescribe medicine, dosage, surgery, supplements, or home treatment.
12. Recommendations must be explicitly present in the report. Do not invent follow-up tests or specialist referrals.
13. Clearly distinguish:
   - What the report actually says
   - What medical terms generally mean
   - Your educational explanation/inference
14. Never claim certainty when the report is uncertain.
15. Do not assign numerical risk scores.
16. If an "Impression" section exists, treat it as the radiologist's summary and explain it separately.
17. Customize doctor questions to the actual report.
18. Keep the language reassuring in tone but never falsely reassuring.

REQUIRED RESPONSE STRUCTURE:

## 1. Simple Summary
Give a short whole-report explanation in approximately 5–8 easy bullet points.

## 2. What Was Scanned?
Explain:
- body part
- region
- side, if stated
- contrast use, if stated
- comparison with previous scan, if stated

For unavailable information, write "Not clearly stated in the report."

## 3. Main Findings
Use a simple Markdown table when useful:
| Report Finding | Simple Meaning | Importance/Context |
Explain what the report says without assigning a risk score.

## 4. Medical Terms Explained
For each important difficult term:
**Medical term:** ...
**Easy meaning:** ...
**Why it appears in the report:** ...
Do not assume a lesion, nodule, mass, or similar term automatically means cancer.

## 5. Normal Findings
List important things the report says appear normal.

## 6. Abnormal Findings
For every noteworthy abnormal finding explain:
- What the report says
- What it means in simple language
- Whether the report recommends follow-up
- What type of healthcare professional may discuss it, if appropriate
Do not diagnose.

## 7. Impression Explained
If an Impression section exists, explain each impression point in very simple language.
Clearly state that the radiologist's impression is the report's summary and should be discussed with the treating doctor.

## 8. Recommendations in the Report
Use the exact heading:
**Recommendations stated in your report**
List only recommendations explicitly written by the radiologist/report, such as follow-up imaging, MRI, ultrasound, clinical correlation, specialist consultation, repeat CT, or laboratory correlation.
If none are stated, say so.

## 9. What Should You Do Next?
Give simple general next steps such as:
1. Keep a copy of the report.
2. Discuss it with the doctor who ordered the scan.
3. Take the CT images/CD/link if available.
4. Tell the doctor about relevant symptoms.
5. Follow any follow-up recommendation written in the report.
Do not prescribe treatment.

## 10. Questions You Can Ask Your Doctor
Generate 5–10 useful questions based on the actual report.

## 11. When to Seek Prompt Medical Attention
Keep this concise. Use cautious general safety guidance. If appropriate, mention that severe or rapidly worsening symptoms such as difficulty breathing, severe chest pain, fainting, sudden weakness, confusion, or other serious symptoms warrant urgent medical care rather than relying on this explanation.

## 12. Important Reminder
End with:
"This explanation is for educational purposes only. A radiologist and your treating healthcare professional can interpret the CT findings in the context of your symptoms, examination, previous scans, and medical history."

IMPORTANT:
- Do not include a diagnosis.
- Do not fabricate missing report details.
- Do not turn general terminology explanations into patient-specific diagnoses.
- The user needs an easy, step-by-step explanation, not a technical lecture.

{truncation_note}

UPLOADED CT/RADIOLOGY REPORT:
------------------------------
{report_text}
------------------------------
"""


def analyze_report(report_text: str, was_truncated: bool) -> str:
    """Send the report to Groq and return the model's explanation."""
    if "GROQ_API_KEY" not in st.secrets:
        raise RuntimeError(
            "Groq API key is not configured. Please add GROQ_API_KEY to Streamlit Secrets."
        )

    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        raise RuntimeError(
            "Groq API key is not configured. Please add GROQ_API_KEY to Streamlit Secrets."
        )

    client = Groq(api_key=api_key, timeout=90.0)
    prompt = build_medical_prompt(report_text, was_truncated)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful medical-report explanation assistant. "
                    "You explain CT/radiology reports in simple language while "
                    "avoiding diagnosis, treatment prescriptions, fabricated facts, "
                    "and false reassurance."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_completion_tokens=7000,
    )

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise RuntimeError("The AI returned an empty response. Please try again.")

    return content.strip()


def display_results(result: str) -> None:
    """Display the generated explanation with a simple, readable hierarchy."""
    st.markdown('<div class="result-title">Your CT Report Explanation</div>', unsafe_allow_html=True)

    sections = re.split(r"(?m)^##\s+", result.strip())
    parsed = {}

    for section in sections:
        if not section.strip():
            continue
        lines = section.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        parsed[title] = body

    preferred_tabs = [
        "Simple Summary",
        "Findings",
        "Medical Terms",
        "Impression",
        "Next Steps",
        "Doctor Questions",
    ]

    available_tabs = []
    for key in preferred_tabs:
        matching = [name for name in parsed if key.lower() in name.lower()]
        if matching:
            available_tabs.append((key, matching[0]))

    if available_tabs:
        tabs = st.tabs([label for label, _ in available_tabs])
        shown = set()

        for tab, (_, actual_title) in zip(tabs, available_tabs):
            with tab:
                st.markdown(parsed[actual_title])
                shown.add(actual_title)

        remaining = [(title, body) for title, body in parsed.items() if title not in shown]
        if remaining:
            st.markdown("---")
            st.markdown("### More Details")
            for title, body in remaining:
                with st.expander(title, expanded=True):
                    st.markdown(body)
    else:
        st.markdown(result)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fc;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: #ffffff;
            border: 1px solid #e4e9f0;
            border-radius: 18px;
            padding: 2rem 2rem 1.6rem 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 5px 18px rgba(20, 40, 70, 0.05);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            line-height: 1.15;
            color: #17324d;
        }

        .hero .subtitle {
            margin-top: .55rem;
            font-size: 1.15rem;
            font-weight: 600;
            color: #38546c;
        }

        .hero .support {
            margin-top: .65rem;
            color: #617487;
            line-height: 1.6;
        }

        .upload-card {
            background: #ffffff;
            border: 1px solid #e4e9f0;
            border-radius: 16px;
            padding: 1.25rem 1.4rem .8rem 1.4rem;
            margin: 1rem 0;
        }

        .section-label {
            color: #17324d;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: .2rem;
        }

        .muted {
            color: #68798a;
        }

        .file-info {
            background: #f2f6fa;
            border: 1px solid #dce5ee;
            border-radius: 12px;
            padding: .8rem 1rem;
            margin: .7rem 0 1rem 0;
            color: #334a5f;
        }

        .result-title {
            color: #17324d;
            font-size: 1.75rem;
            font-weight: 750;
            margin: 1.5rem 0 1rem 0;
        }

        .privacy {
            color: #657687;
            font-size: .85rem;
            margin-top: 1rem;
            padding-top: .8rem;
            border-top: 1px solid #e1e7ed;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 700;
            min-height: 2.8rem;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 1.3rem;
            }

            .hero h1 {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="CT Report Assistant",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    with st.sidebar:
        st.markdown("## About")
        st.write(
            "CT Report Assistant helps explain technical CT/radiology report "
            "language in simple terms."
        )

        st.markdown("## How it works")
        st.write(
            "1. Upload your CT report.\n"
            "2. Click Analyze.\n"
            "3. Read the simplified explanation.\n"
            "4. Discuss the results with your healthcare professional."
        )

        st.markdown("## Safety")
        st.info(
            "AI explanations can contain mistakes. Always verify important "
            "medical information with a qualified healthcare professional."
        )

        st.caption(
            "Privacy: Your uploaded report is processed for this session and "
            "is not intentionally stored by this application."
        )

    st.markdown(
        """
        <div class="hero">
            <h1>CT Report Assistant</h1>
            <div class="subtitle">Understand your CT report in simple language</div>
            <div class="support">
                Upload your CT report and get an easy-to-understand explanation.
                This tool provides educational information and does not replace
                a qualified healthcare professional.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Important: This AI assistant explains information written in your CT "
        "report in simpler language. It does not diagnose diseases, determine "
        "treatment, or replace a radiologist or doctor. Always discuss your "
        "report and symptoms with a qualified healthcare professional."
    )

    st.caption(f"AI engine: {MODEL_NAME}")

    st.markdown(
        """
        <div class="upload-card">
            <div class="section-label">Upload Your CT Report</div>
            <div class="muted">Upload a PDF or TXT copy of your CT/radiology report.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a CT/radiology report",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if not uploaded_file:
        st.info("Upload a PDF or TXT report to begin.")
        return

    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.name.lower().rsplit(".", 1)[-1] if "." in uploaded_file.name else ""

    try:
        if file_type == "pdf":
            extracted_text = extract_pdf_text(file_bytes)
        elif file_type == "txt":
            extracted_text = extract_txt_text(file_bytes)
        else:
            st.error("Unsupported file type. Please upload a PDF or TXT file.")
            return
    except Exception:
        st.error("We could not read this file. Please try another PDF or TXT report.")
        return

    extracted_text = normalize_text(extracted_text)

    word_count = len(re.findall(r"\S+", extracted_text))
    st.markdown(
        f"""
        <div class="file-info">
            <strong>File:</strong> {uploaded_file.name}<br>
            <strong>Type:</strong> {file_type.upper()}<br>
            <strong>Extracted text:</strong> {word_count:,} words
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not extracted_text:
        if file_type == "pdf":
            st.error(
                "Your PDF appears to contain scanned images rather than selectable "
                "text. Please upload a text-based CT report PDF or TXT file."
            )
        else:
            st.error("We could not extract readable text from this TXT file.")
        return

    with st.expander("View Extracted Report"):
        preview_limit = 20000
        if len(extracted_text) > preview_limit:
            st.text(extracted_text[:preview_limit])
            st.caption(
                "Only the first 20,000 characters are shown in the preview. "
                "The application may process more text during analysis."
            )
        else:
            st.text(extracted_text)

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_filename" not in st.session_state:
        st.session_state.analysis_filename = None

    if st.button("Analyze CT Report", type="primary", use_container_width=True):
        try:
            with st.status("Reading your report...", expanded=False) as status:
                status.update(label="Preparing an easy-to-understand explanation...")
                report_for_model, was_truncated = limit_report_text(extracted_text)

                if was_truncated:
                    st.warning(
                        "The uploaded report is longer than the processing limit. "
                        "The application analyzed the available report text."
                    )

                status.update(label="Preparing an easy-to-understand explanation...")
                result = analyze_report(report_for_model, was_truncated)
                status.update(label="Explanation ready.", state="complete")

            st.session_state.analysis_result = result
            st.session_state.analysis_filename = uploaded_file.name

        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            error_text = str(exc).lower()

            if (
                "decommissioned" in error_text
                or "deprecated" in error_text
                or "model_not_found" in error_text
                or "model not found" in error_text
                or "not found" in error_text and MODEL_NAME.lower() in error_text
            ):
                st.error(
                    f"The configured Groq model ({MODEL_NAME}) is unavailable. "
                    "Please make sure the latest app.py is deployed."
                )
            elif "timeout" in error_text or "timed out" in error_text:
                st.error(
                    "The AI service took too long to respond. Please try again "
                    "with the same report or a shorter report."
                )
            elif (
                "authentication" in error_text
                or "unauthorized" in error_text
                or "invalid api key" in error_text
                or "api key" in error_text
            ):
                st.error(
                    "The Groq API key could not be authenticated. Please check "
                    "GROQ_API_KEY in Streamlit Secrets."
                )
            elif "rate" in error_text or "429" in error_text:
                st.error(
                    "The AI service is temporarily rate-limited. Please wait a "
                    "little and try again."
                )
            elif "413" in error_text or "too large" in error_text:
                st.error(
                    "The report request is too large. Please upload a shorter "
                    "report or use a report containing only the relevant findings."
                )
            else:
                st.error(
                    "The report could not be analyzed. Please check your Groq "
                    "API key and try again."
                )
                with st.expander("Technical error details"):
                    st.code(str(exc))

    if (
        st.session_state.analysis_result
        and st.session_state.analysis_filename == uploaded_file.name
    ):
        display_results(st.session_state.analysis_result)


if __name__ == "__main__":
    main()
