import os
import csv
import random
from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================
# CONFIG
# =========================
PATIENT_IDS = ["patient_001", "patient_002", "patient_003"]
DATA_DIR = "data"
RAG_DOCS_DIR = "rag_docs"

GLUCOSE_LOW = 70
GLUCOSE_VERY_LOW = 54
GLUCOSE_HIGH = 180
GLUCOSE_VERY_HIGH = 300

BP_CRISIS_SYS = 180
BP_CRISIS_DIA = 120


# =========================
# UTILS
# =========================
def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for pid in PATIENT_IDS:
        os.makedirs(os.path.join(DATA_DIR, pid), exist_ok=True)

def append_csv(path, header, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =========================
# SYNTHETIC DATA
# =========================
def synthetic_glucose(context):
    if context == "fasting":
        value = random.gauss(105, 15)
    elif context == "post_meal":
        value = random.gauss(165, 35)
    else:
        value = random.gauss(130, 25)

    if random.random() < 0.05:
        value += random.choice([80, 120])
    if random.random() < 0.03:
        value -= random.choice([40, 60])

    return int(max(40, min(420, round(value))))

def synthetic_bp(context):
    if context == "rest":
        sys = random.gauss(122, 12)
        dia = random.gauss(78, 8)
    else:
        sys = random.gauss(135, 18)
        dia = random.gauss(86, 10)

    if random.random() < 0.05:
        sys += random.choice([20, 40])
        dia += random.choice([10, 20])

    if random.random() < 0.01:
        sys += 70
        dia += 45

    return int(sys), int(dia)


# =========================
# ABNORMAL DETECTION
# =========================
def glucose_flag(value):
    if value < GLUCOSE_VERY_LOW:
        return True, "very_low_glucose"
    if value < GLUCOSE_LOW:
        return True, "low_glucose"
    if value > GLUCOSE_VERY_HIGH:
        return True, "very_high_glucose"
    if value > GLUCOSE_HIGH:
        return True, "high_glucose"
    return False, "normal"

def bp_flag(sys, dia):
    if sys > BP_CRISIS_SYS or dia > BP_CRISIS_DIA:
        return True, "bp_crisis"
    if sys >= 140 or dia >= 90:
        return True, "bp_stage_2"
    if 130 <= sys <= 139 or 80 <= dia <= 89:
        return True, "bp_stage_1"
    if 120 <= sys <= 129 and dia < 80:
        return True, "bp_elevated"
    return False, "normal"


# =========================
# RAG SETUP
# =========================
def build_vectorstore():
    docs = []
    for file in os.listdir(RAG_DOCS_DIR):
        if file.endswith(".txt"):
            docs.extend(TextLoader(os.path.join(RAG_DOCS_DIR, file)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )



def llm_advisor(llm, vectorstore, patient_id, payload):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    query = f"""
Analyze the following SYNTHETIC patient data and provide EDUCATIONAL advice only.

Patient ID: {patient_id}
Timestamp: {payload['timestamp']}

Measurements:
- Glucose: {payload['glucose']} mg/dL ({payload['glucose_context']})
- Blood Pressure: {payload['sys']}/{payload['dia']} mmHg ({payload['bp_context']})

Detected flags:
- Glucose: {payload['glucose_flag']}
- Blood Pressure: {payload['bp_flag']}

Instructions:
- Explain what these abnormal values mean
- Use only retrieved context
- No diagnosis
- No medication
- Provide general monitoring advice
- State clearly this is educational and based on synthetic data

Output format:
Summary:
Why it matters:
Suggested next steps:
Safety note:
"""

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
AUTHORITATIVE CONTEXT:
{context}

USER QUERY:
{query}
"""

    return llm.invoke(prompt).content


# =========================
# MAIN
# =========================
def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    ensure_dirs()

    vectorstore = build_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    for pid in PATIENT_IDS:
        ts = now_utc()

        g_context = random.choice(["fasting", "post_meal", "random"])
        bp_context = random.choice(["rest", "stress"])

        glucose = synthetic_glucose(g_context)
        sys, dia = synthetic_bp(bp_context)

        g_abn, g_flag = glucose_flag(glucose)
        bp_abn, bp_flag_value = bp_flag(sys, dia)

        if not (g_abn or bp_abn):
            continue  # abnormal only

        payload = {
            "timestamp": ts,
            "glucose": glucose,
            "glucose_context": g_context,
            "glucose_flag": g_flag,
            "sys": sys,
            "dia": dia,
            "bp_context": bp_context,
            "bp_flag": bp_flag_value,
        }

        patient_dir = os.path.join(DATA_DIR, pid)

        append_csv(
            os.path.join(patient_dir, "abnormal_events.csv"),
            [
                "timestamp",
                "glucose",
                "glucose_context",
                "glucose_flag",
                "sys",
                "dia",
                "bp_context",
                "bp_flag",
            ],
            payload,
        )

        advice = llm_advisor(llm, vectorstore, pid, payload)

        print("\n" + "=" * 60)
        print(f"ABNORMAL EVENT | {pid} | {ts}")
        print(f"Glucose: {glucose} mg/dL ({g_context}) → {g_flag}")
        print(f"BP: {sys}/{dia} mmHg ({bp_context}) → {bp_flag_value}")
        print("-" * 60)
        print(advice)
        print("=" * 60)

        append_csv(
            os.path.join(patient_dir, "analysis_log.csv"),
            ["timestamp", "advice"],
            {"timestamp": ts, "advice": advice},
        )


if __name__ == "__main__":
    main()
