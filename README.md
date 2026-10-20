# 🍚 점심 메뉴 추천기 (261020_bucheon_huggingface_ysh)

주소와 오늘의 기분을 입력하면 **카카오맵으로 주변 음식점**을 찾고,
**Hugging Face 모델**로 취향에 맞는 곳을 골라 한마디와 함께 추천해주는 Gradio 서비스입니다.
👍 / 👎 / 🔄 를 누를수록 내 취향을 학습해서 추천이 달라집니다.

## 사용 모델 (Hugging Face Inference API)
| 용도 | 모델 |
|---|---|
| 기분 → 메뉴 후보 확장, 추천 멘트 | `meta-llama/Llama-3.3-70B-Instruct` |
| 식당·취향 임베딩 | `BAAI/bge-m3` |

## 추천 알고리즘
```
기분 문장 → LLM이 어울리는 메뉴로 확장 → 임베딩(질문 벡터)
카카오맵 주변 음식점 → "이름 + 카테고리" 임베딩
점수 = (1-w) × 취향 유사도 + w × 가까움      (w: 슬라이더)
👍 : 질문 벡터를 좋아요한 식당 쪽으로 이동   (Rocchio)
👎 : 해당 식당 제외 + 그 방향에서 멀어지도록 이동
🔄 : 지금 보여준 후보를 건너뛰고 다음 후보
```

## 실행 방법
1. 의존성 설치 (Python 3.10+)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
   > 이미 만든 `.venv`에서 numpy/scipy 오류가 나면 `.venv` 폴더를 지우고 다시 만드세요.
2. 환경 변수: `.env.example`을 `.env`로 복사해 키 입력
   ```
   HF_TOKEN=hf_xxxxxxxx              # https://huggingface.co/settings/tokens
   KAKAO_REST_API_KEY=xxxxxxxx       # https://developers.kakao.com → 앱 만들기 → REST API 키
   ```
   카카오 키가 없으면 샘플 데이터(부천역 주변 가상 식당)로 동작합니다. `.env`는 Git에 올라가지 않습니다.
3. 실행
   ```bash
   python lunch_app.py
   ```
   브라우저에서 http://127.0.0.1:7860 접속

## 사용법
1. 오늘의 기분(예: "비 와서 얼큰한 국물")을 입력하고 **📍 내 위치로 추천받기** (브라우저 위치 권한 허용)
   - 위치가 안 잡히면 "주소로 찾기"에 역 이름(예: 부천역)을 입력합니다.
2. 추천된 3곳 중 하나를 고르고 **👍 이런 곳 더 / 👎 별로 / 🔄 다른 곳**
3. 카카오맵 링크로 위치와 리뷰 확인

## 한계
- 카카오 로컬 API는 평점·리뷰를 제공하지 않아 "맛집 순위"가 아닌 **취향·거리 기반** 추천입니다.
- 식당 이름과 카테고리만으로 판단하므로 실제 메뉴와 다를 수 있습니다.

## 부록: 문서 요약 & Q&A (`app.py`)
PDF·URL·글을 요약하고 근거와 함께 질문에 답하는 다른 데모입니다. `python app.py`로 실행합니다.
