"""문서 요약 + 질문 답변 서비스 (Hugging Face Inference API + Gradio)

PDF / 웹페이지 URL / 붙여넣은 글을 넣으면
1) 원하는 난이도로 요약하고
2) 문서 내용에 대해 질문하면 근거 문장과 함께 답해준다.
"""
import os

import pymupdf
import gradio as gr
import trafilatura
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
load_dotenv("../../.env")  # 수업용 공용 .env 도 허용

MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
client = InferenceClient(api_key=os.environ["HF_TOKEN"])

CHUNK_SIZE = 800  # 검색용 청크 글자 수
SEGMENT_SIZE = 6000  # 요약 1회에 넣는 글자 수
MAX_SEGMENTS = 8  # 너무 긴 문서는 앞부분만 요약
TOP_K = 4

LEVELS = {
    "쉽게 (비전공자)": "전문 용어를 풀어서, 배경지식이 없는 사람도 이해할 수 있게",
    "보통": "핵심 위주로 간결하게",
    "전문가": "전문 용어를 그대로 쓰고 방법·수치·한계까지 구체적으로",
}


def chat(system: str, user: str, max_tokens: int = 900) -> str:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return res.choices[0].message.content.strip()


# ---------- 1. 텍스트 추출 ----------
def load_text(url: str, pdf_path: str | None, pasted: str) -> str:
    if pdf_path:
        with pymupdf.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    if url and url.strip():
        html = trafilatura.fetch_url(url.strip())
        text = trafilatura.extract(html) if html else None
        if not text:
            raise gr.Error("웹페이지에서 본문을 가져오지 못했어요. 다른 URL을 시도해 주세요.")
        return text
    if pasted and pasted.strip():
        return pasted
    raise gr.Error("PDF, URL, 텍스트 중 하나를 입력해 주세요.")


# ---------- 2. 청킹 ----------
def split_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    chunks, buf = [], ""
    for para in (p.strip() for p in text.splitlines()):
        if not para:
            continue
        while len(para) > size:  # 문단이 너무 길면 강제로 자른다
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(para[:size])
            para = para[size:]
        if len(buf) + len(para) > size and buf:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{para}".strip()
    if buf:
        chunks.append(buf)
    return chunks


# ---------- 3. 요약 (긴 문서는 나눠서 요약 후 합치기) ----------
def summarize(text: str, level: str) -> str:
    style = LEVELS[level]
    system = f"너는 문서 요약 전문가다. 항상 한국어로, {style} 설명한다. 원문에 없는 내용은 지어내지 않고, 숫자와 단위(B, M, %, $ 등)는 원문 표기 그대로 쓴다."
    segments = [text[i : i + SEGMENT_SIZE] for i in range(0, len(text), SEGMENT_SIZE)]
    segments = segments[:MAX_SEGMENTS]

    if len(segments) == 1:
        partials = [text]
    else:
        partials = [
            chat(system, f"다음 글의 핵심을 정리해줘.\n\n{seg}", max_tokens=500)
            for seg in segments
        ]

    merged = "\n\n".join(partials)
    return chat(
        system,
        "아래 내용을 다음 형식으로 요약해줘.\n"
        "### 한 줄 요약\n### 핵심 내용 (3~5개 불릿)\n### 알아둘 점\n\n" + merged,
        max_tokens=900,
    )


# ---------- 4. 질문 답변 (TF-IDF 검색 + LLM) ----------
def build_index(chunks: list[str]):
    # 한국어는 띄어쓰기 변형이 많아 글자 n-gram 이 단어 단위보다 잘 맞는다
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    return vec, vec.fit_transform(chunks)


def to_search_query(question: str, sample: str) -> str:
    """질문 언어와 문서 언어가 다르면 검색이 안 되므로, 문서 언어의 검색 키워드로 바꿔준다."""
    try:
        keywords = chat(
            "사용자의 질문을 [문서 샘플]이 쓰는 언어(영어 문서면 영어)의 검색 키워드로 바꿔라. "
            "동의어도 함께 5~8개 적고, 키워드만 공백으로 구분해 한 줄로 출력하라.",
            f"[문서 샘플]\n{sample}\n\n질문: {question}",
            max_tokens=60,
        )
    except Exception:  # 키워드 변환이 실패해도 원래 질문으로 검색은 계속한다
        keywords = ""
    return f"{question} {keywords}"


def retrieve(question: str, state: dict) -> list[tuple[int, str]]:
    q = state["vec"].transform([to_search_query(question, state["sample"])])
    scores = cosine_similarity(q, state["matrix"])[0]
    top = scores.argsort()[::-1][:TOP_K]
    return [(int(i), state["chunks"][i]) for i in top if scores[i] > 0]


def answer(question: str, state: dict) -> tuple[str, str]:
    hits = retrieve(question, state)
    if not hits:
        return "문서에서 관련된 내용을 찾지 못했어요.", ""
    context = "\n\n".join(f"[{n + 1}] {c}" for n, c in hits)
    reply = chat(
        "너는 문서 기반 질의응답 도우미다. 반드시 아래 [문서 조각]에 있는 내용만으로 한국어로 답하고, "
        "근거가 된 조각 번호를 [1] 처럼 붙인다. 숫자와 단위(B, M, %, $ 등)는 원문 표기 그대로 쓴다. 문서에 답이 없으면 '문서에 없는 내용이에요'라고 답한다.",
        f"[문서 조각]\n{context}\n\n질문: {question}",
        max_tokens=600,
    )
    sources = "\n\n".join(f"**[{n + 1}]** {c}" for n, c in hits)
    return reply, sources


# ---------- Gradio UI ----------
def on_analyze(url, pdf_path, pasted, level):
    text = load_text(url, pdf_path, pasted)
    chunks = split_chunks(text)
    if not chunks:
        raise gr.Error("읽을 수 있는 텍스트가 없어요. 스캔본 PDF는 지원하지 않아요.")
    vec, matrix = build_index(chunks)
    # 표·목록이 아닌 본문 위주의 조각을 언어 판별용 샘플로 쓴다
    sample = max(chunks[:5], key=lambda c: sum(ch.isalpha() for ch in c))[:300]
    state = {"chunks": chunks, "vec": vec, "matrix": matrix, "sample": sample}
    summary = summarize(text, level)
    info = f"총 {len(text):,}자 · {len(chunks)}개 조각으로 나눠 읽었어요."
    return summary, info, state, [], ""


def on_ask(question, history, state):
    if not state:
        raise gr.Error("먼저 위에서 문서를 분석해 주세요.")
    if not question.strip():
        return history, "", ""
    reply, sources = answer(question, state)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]
    return history, sources, ""


with gr.Blocks(title="문서 요약 & Q&A") as demo:
    gr.Markdown(
        "# 📄 문서 요약 & Q&A\n"
        "PDF·웹페이지·글을 넣으면 **요약**해주고, 궁금한 건 **근거와 함께** 답해줘요."
    )
    state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("웹페이지 URL"):
                    url = gr.Textbox(label="URL", placeholder="https://...")
                with gr.Tab("PDF 업로드"):
                    pdf = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                with gr.Tab("텍스트 붙여넣기"):
                    pasted = gr.Textbox(label="본문", lines=8)
            level = gr.Radio(list(LEVELS), value="보통", label="요약 난이도")
            run = gr.Button("분석하기", variant="primary")
            info = gr.Markdown()
        with gr.Column(scale=1):
            summary = gr.Markdown(label="요약")

    gr.Markdown("## 💬 문서에 질문하기")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=320)
            question = gr.Textbox(label="질문", placeholder="예: 이 글의 결론이 뭐야?")
        with gr.Column(scale=1):
            sources = gr.Markdown("### 📌 근거 문장\n질문하면 여기에 표시돼요.")

    run.click(
        on_analyze,
        [url, pdf, pasted, level],
        [summary, info, state, chatbot, sources],
    )
    question.submit(on_ask, [question, chatbot, state], [chatbot, sources, question])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
