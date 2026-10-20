"""점심 메뉴 추천기 (카카오맵 + Hugging Face)

주소와 오늘의 기분을 입력하면 카카오맵으로 주변 음식점을 찾고,
Hugging Face 임베딩 모델(bge-m3)로 취향과 가까운 곳을 골라 LLM이 한마디 붙여준다.
👍/👎/🔄 피드백을 누를수록 취향 벡터가 움직여서 추천이 달라진다.
"""
import os

import gradio as gr
import numpy as np
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
load_dotenv("../../.env")

KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY", "").strip()
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
LLM_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
client = InferenceClient(api_key=os.environ["HF_TOKEN"])

TOP_K = 3
LIKE_W, DISLIKE_W = 0.5, 0.4  # 취향 벡터 이동 강도

# 카카오 키가 없을 때 쓰는 샘플 (부천역 주변 느낌의 가상 데이터)
SAMPLE_PLACES = [
    ("할매순두부", "한식 > 찌개,전골", 120), ("김밥천국 부천역점", "분식", 90),
    ("마라공방", "중식 > 마라탕", 310), ("스시로", "일식 > 초밥,롤", 450),
    ("엄마손칼국수", "한식 > 국수", 200), ("버거킹 부천점", "패스트푸드 > 햄버거", 260),
    ("돈까스공방", "일식 > 돈까스,우동", 340), ("할리스 샌드위치", "카페 > 샌드위치", 150),
    ("옛날부대찌개", "한식 > 찌개,전골", 410), ("파스타하우스", "양식 > 이탈리안", 520),
    ("차이나식당", "중식 > 중국요리", 380), ("본죽&비빔밥", "한식 > 죽", 180),
    ("명동떡볶이", "분식 > 떡볶이", 230), ("고깃집 불꽃", "한식 > 육류,고기", 600),
    ("쌀국수 하노이", "아시아음식 > 베트남음식", 470), ("설렁탕 명가", "한식 > 설렁탕", 290),
    ("샐러드랩", "양식 > 샐러드", 330), ("이자카야 하루", "일식 > 술집", 700),
    ("뼈해장국 24시", "한식 > 해장국", 360), ("멕시칸 타코", "양식 > 멕시칸", 800),
]


def chat(system: str, user: str, max_tokens: int = 400) -> str:
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return res.choices[0].message.content.strip()


# ---------- 1. 카카오맵으로 주변 음식점 찾기 ----------
def kakao_get(path: str, **params) -> dict:
    res = requests.get(
        f"https://dapi.kakao.com/v2/local/search/{path}.json",
        headers={"Authorization": f"KakaoAK {KAKAO_KEY}"},
        params=params,
        timeout=10,
    )
    if res.status_code == 401:
        raise gr.Error("카카오 API 키가 올바르지 않아요. .env 의 KAKAO_REST_API_KEY 를 확인해 주세요.")
    res.raise_for_status()
    return res.json()


def reverse_geocode(x: str, y: str) -> str:
    """좌표 → 행정동 이름. 위치가 잘못 잡혔는지 사용자가 바로 알 수 있게 보여준다."""
    try:
        res = requests.get(
            "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json",
            headers={"Authorization": f"KakaoAK {KAKAO_KEY}"},
            params={"x": x, "y": y},
            timeout=10,
        )
        docs = res.json().get("documents", [])
        return docs[0]["address_name"] if docs else "알 수 없음"
    except Exception:
        return "알 수 없음"


def search_places(address: str, radius: int, coords: str = "") -> tuple[list[dict], str]:
    if not KAKAO_KEY:
        places = [
            {"id": str(i), "name": n, "category": c, "distance": d, "url": ""}
            for i, (n, c, d) in enumerate(SAMPLE_PLACES)
            if d <= radius
        ]
        return places, "⚠️ 카카오 키가 없어 **샘플 데이터**로 보여드려요."

    if coords:  # 브라우저에서 받은 현재 위치 "위도,경도"
        y, x = coords.split(",")
        label = f"📍 **현재 위치({reverse_geocode(x, y)})**"
    else:
        found = kakao_get("keyword", query=address, size=1).get("documents", [])
        if not found:
            raise gr.Error(f"'{address}' 위치를 찾지 못했어요. 역 이름이나 건물명으로 다시 입력해 보세요.")
        x, y = found[0]["x"], found[0]["y"]
        label = f"📍 **{found[0]['place_name']}**"

    places = []
    for page in (1, 2, 3):  # 페이지당 15개, 최대 45개
        data = kakao_get(
            "category", category_group_code="FD6", x=x, y=y,
            radius=radius, sort="distance", page=page, size=15,
        )
        for d in data.get("documents", []):
            places.append({
                "id": d["id"],
                "name": d["place_name"],
                "category": d["category_name"].replace("음식점 > ", ""),
                "distance": int(d["distance"] or 0),
                "url": d["place_url"],
            })
        if data.get("meta", {}).get("is_end"):
            break
    return places, f"{label} 주변 {radius}m 안에서 {len(places)}곳을 찾았어요."


# ---------- 2. 임베딩 기반 추천 ----------
def expand_mood(mood: str) -> str:
    """짧은 기분 문장을 구체적인 메뉴 후보로 풀어서 검색 품질을 높인다."""
    try:
        menus = chat(
            "사용자의 점심 기분/상황에 어울리는 한국 식당 메뉴와 음식 종류를 8~10개, 쉼표로만 나열하라. "
            "다른 말은 하지 마라.",
            mood,
            max_tokens=80,
        )
    except Exception:  # 확장이 실패해도 원문으로 계속한다
        menus = ""
    return f"{mood}. 어울리는 메뉴: {menus}"



def embed(texts: list[str]) -> np.ndarray:
    vecs = np.asarray(client.feature_extraction(texts, model=EMBED_MODEL), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def rank(state: dict, dist_weight: float) -> list[int]:
    """취향 벡터와의 유사도 + 가까움 점수로 순위를 매긴다."""
    embs, places = state["embs"], state["places"]

    # Rocchio: 질문 벡터를 좋아요 쪽으로 당기고 싫어요에서 밀어낸다
    taste = state["query_emb"].copy()
    if state["liked"]:
        taste += LIKE_W * embs[list(state["liked"])].mean(axis=0)
    if state["disliked"]:
        taste -= DISLIKE_W * embs[list(state["disliked"])].mean(axis=0)

    sim = embs @ taste
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-9)
    radius = max(p["distance"] for p in places) or 1
    near = np.array([1 - p["distance"] / radius for p in places])
    score = (1 - dist_weight) * sim + dist_weight * near

    banned = state["disliked"] | state["skipped"]
    order = [int(i) for i in score.argsort()[::-1] if int(i) not in banned]
    if not order:  # 다 걸러졌으면 건너뛴 기록을 초기화
        state["skipped"].clear()
        order = [int(i) for i in score.argsort()[::-1] if int(i) not in state["disliked"]]
    return order[:TOP_K]


# ---------- 3. LLM 한마디 ----------
def comments_for(top: list[dict], mood: str) -> list[str]:
    listing = "\n".join(f"{i + 1}. {p['name']} ({p['category']})" for i, p in enumerate(top))
    try:
        text = chat(
            "너는 센스 있는 점심 추천 친구다. 식당마다 '왜 오늘 기분에 어울리는지'를 "
            "친구한테 말하듯 가볍고 재치 있게 한 문장(40자 이내)으로 쓴다. "
            "근거는 식당 이름과 종류, 사용자의 기분뿐이다. 맛, 재료, 가격, 후기는 절대 지어내지 말고, "
            "기분과 잘 안 맞으면 억지로 맞추지 말고 '기분 전환용으로 색다르게'처럼 솔직하게 쓴다. "
            "번호를 붙여 한 줄에 하나씩, 식당 수만큼만 출력한다.",
            f"오늘의 기분: {mood}\n\n{listing}",
        )
        lines = [ln.split(".", 1)[-1].strip() for ln in text.splitlines() if ln.strip()]
    except Exception:
        lines = []
    return (lines + [""] * len(top))[: len(top)]


def render(state: dict, top_idx: list[int], comments: list[str], note: str) -> str:
    out = [note, ""]
    for n, (i, cm) in enumerate(zip(top_idx, comments), 1):
        p = state["places"][i]
        link = f" · [카카오맵에서 보기]({p['url']})" if p["url"] else ""
        out.append(f"### {n}. {p['name']}\n`{p['category']}` · {p['distance']}m{link}")
        if cm:
            out.append(f"> {cm}")
        out.append("")
    return "\n".join(out)


def show(state, dist_weight, note):
    top_idx = rank(state, dist_weight)
    state["top"] = top_idx
    top = [state["places"][i] for i in top_idx]
    md = render(state, top_idx, comments_for(top, state["mood"]), note)
    names = [p["name"] for p in top]
    return md, gr.Radio(choices=names, value=names[0] if names else None)


# ---------- 이벤트 ----------
def on_recommend(address, mood, radius, dist_weight, coords=""):
    # 접힌 아코디언 안의 입력칸이나 JS 에서 온 값은 None 으로 올 수 있다
    address, mood, coords = address or "", mood or "", coords or ""
    radius = radius or 800
    dist_weight = 0.2 if dist_weight is None else dist_weight
    if coords.startswith("ERR"):
        raise gr.Error("현재 위치를 가져오지 못했어요. 브라우저에서 위치 권한을 허용하거나, 아래에 주소를 직접 입력해 주세요.")
    if KAKAO_KEY and not coords and not address.strip():
        raise gr.Error("주소나 역 이름을 입력해 주세요. (예: 부천역)")
    mood = mood.strip() or "아무거나 맛있는 점심"
    places, note = search_places(address.strip(), int(radius), coords)
    if len(places) < TOP_K:
        raise gr.Error("주변에서 음식점을 충분히 찾지 못했어요. 반경을 넓혀 보세요.")

    embs = embed([f"식당 {p['name']}, 종류: {p['category']}" for p in places])
    query_emb = embed([expand_mood(mood)])[0]
    state = {
        "places": places, "embs": embs, "query_emb": query_emb, "mood": mood,
        "liked": set(), "disliked": set(), "skipped": set(), "top": [],
    }
    md, radio = show(state, dist_weight, note)
    return md, radio, state


def on_feedback(kind, choice, state, dist_weight):
    if not state:
        raise gr.Error("먼저 추천을 받아 주세요.")
    top = state["top"]
    names = [state["places"][i]["name"] for i in top]
    target = top[names.index(choice)] if choice in names else top[0]

    if kind == "like":
        state["liked"].add(target)
        note = f"👍 **{state['places'][target]['name']}** 취향을 반영해서 비슷한 곳을 더 찾았어요."
    elif kind == "dislike":
        state["disliked"].add(target)
        state["liked"].discard(target)
        note = f"👎 **{state['places'][target]['name']}** 은(는) 제외하고 다시 골랐어요."
    else:
        state["skipped"].update(top)
        note = "🔄 다른 후보로 바꿨어요."
    md, radio = show(state, dist_weight, note)
    return md, radio, state


GET_LOCATION_JS = """
async (address, mood, radius, dist_w, coords) => {
  const pos = await new Promise((resolve) => {
    if (!navigator.geolocation) return resolve(null);
    navigator.geolocation.getCurrentPosition(resolve, () => resolve(null),
      {enableHighAccuracy: true, timeout: 10000});
  });
  const c = pos ? pos.coords.latitude + "," + pos.coords.longitude : "ERR";
  return [address, mood, radius, dist_w, c];
}
"""

with gr.Blocks(title="점심 메뉴 추천기") as demo:
    gr.Markdown(
        "# 🍚 오늘 점심 뭐 먹지?\n"
        "기분을 적으면 **주변 식당**을 찾아 취향에 맞게 추천해요. "
        "👍/👎를 누를수록 내 취향을 학습합니다."
    )
    state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            mood = gr.Textbox(label="오늘의 기분/상황", placeholder="예: 비 와서 얼큰한 국물, 혼밥, 가볍게 샐러드")
            radius = gr.Slider(300, 1500, value=800, step=100, label="검색 반경 (m)")
            dist_w = gr.Slider(0, 0.6, value=0.2, step=0.05, label="가까운 곳 선호도")
            run_here = gr.Button("📍 내 위치로 추천받기", variant="primary")
            with gr.Accordion("위치가 안 잡히면 주소로 찾기", open=False):
                address = gr.Textbox(label="주소/역/건물명", placeholder="예: 부천역, 상동역")
                run = gr.Button("🍽️ 이 주소로 추천받기")
            coords = gr.Textbox(visible=False)
        with gr.Column(scale=2):
            result = gr.Markdown("추천 결과가 여기에 표시돼요.")
            pick = gr.Radio(choices=[], label="마음에 드는(안 드는) 곳을 고르세요")
            with gr.Row():
                like = gr.Button("👍 이런 곳 더")
                dislike = gr.Button("👎 별로")
                other = gr.Button("🔄 다른 곳")

    run.click(
        lambda a, m, r, w: on_recommend(a, m, r, w, ""),
        [address, mood, radius, dist_w],
        [result, pick, state],
    )
    # 브라우저에서 현재 위치를 받아 coords 에 채운 뒤 파이썬 함수를 호출한다
    run_here.click(
        on_recommend,
        [address, mood, radius, dist_w, coords],
        [result, pick, state],
        js=GET_LOCATION_JS,
    )
    for btn, kind in ((like, "like"), (dislike, "dislike"), (other, "other")):
        btn.click(
            lambda choice, s, w, k=kind: on_feedback(k, choice, s, w),
            [pick, state, dist_w],
            [result, pick, state],
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
