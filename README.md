# hearing-to-seeing

음성의 특성(화자, 타이밍, 음량)을 동적/키네틱 자막으로 변환하는 서비스입니다. 화자별 색상(작품 등장인물 정보 기반), 단어 단위 싱크 애니메이션, 음량에 따른 움직임(큰 소리는 커지고, 속삭임은 작아짐)으로 청각장애인 및 난청인을 위한 자막을 만듭니다. 출력 포맷으로는 애니메이션·서식 기능을 지원하는 ASS(Advanced SubStation Alpha)를 사용합니다.

## 설치 및 실행

이 프로젝트는 패키지 관리에 [`uv`](https://docs.astral.sh/uv/)를 사용합니다 (Python 3.12).

그 외에 **ffmpeg**와, YouTube 링크를 받기 위한 **JS 런타임**(Deno, 또는 Node.js 20+)이 필요합니다.

```bash
# ffmpeg (ffprobe 포함)
sudo apt install ffmpeg

# Deno (Node.js 20+가 이미 있으면 생략)
curl -fsSL https://deno.land/install.sh | sh
```

```bash
# 의존성 설치
uv sync

# CLI — WAV를 받아 같은 폴더에 sample.ass 생성
# (WAV 만들기: ffmpeg -i data/input/sample.mp4 -ac 1 -ar 16000 data/input/sample.wav)
uv run hearing-to-seeing data/input/sample.wav

# CLI — 옵션 전부: 출력 경로, 중간 JSON, 언어, 화자 수(아는 경우), 원본 영상(GEMINI_API_KEY 필요, 화면을 보고 화자 확인)
uv run hearing-to-seeing data/input/sample.wav -o data/output/sample.ass --json data/output/sample.json \
    --language ko --speakers 6 --video data/input/sample.mp4

# 웹 — http://127.0.0.1:8000 (YouTube 링크를 받아 영상 제목으로 등장인물을 찾아 화자 색상 결정)
uv run hearing-to-seeing-web --host 127.0.0.1 --port 8000
```

## 환경 변수

[`.env.example`](.env.example)을 `.env`로 복사한 뒤 값을 채웁니다.

| 변수 | 필수 | 설명 |
|------|------|------|
| `WHISPERX_API_KEY` | ✅ | 원격 WhisperX 서버(STT·정렬·화자 분리) API 키 |
| `WHISPERX_API_URL` | ✅ | 원격 WhisperX 서버 주소 |
| `GEMINI_API_KEY` | | 설정하면 웹 작업에서 영상 제목으로 작품의 등장인물을 찾아 화자별 색상을 정하고, 원본 영상을 보며 발화마다 화자를 확인함 ([knowledge/README.md](src/hearing_to_seeing/knowledge/README.md)). 없으면 색상환에 고르게 배정. [발급](https://aistudio.google.com/apikey) (무료 티어로 충분) |
| `H2S_LLM_MODEL` | | 색상 조회에 쓸 Gemini 모델 (기본 `gemini-3.6-flash`, 한도·과부하 시 `gemini-3.5-flash-lite`로 자동 폴백) |
| `H2S_REVIEW_MODEL` | | 영상 검토(발화별 화자, 엇갈린 단어 선택)에 쓸 Gemini 모델 (기본 `gemini-3.5-flash`, 폴백 없음 — 실패하면 검토를 건너뜀). [docs/STT_TUNING.md](docs/STT_TUNING.md) 4차 참조 |
| `H2S_FFMPEG` / `H2S_FFPROBE` | | `PATH`에 없는 ffmpeg / ffprobe 경로. `H2S_FFPROBE`는 생략하면 `H2S_FFMPEG`와 같은 폴더에서 찾음 |
| `H2S_LIBRARY_DIR` | | 웹 화면의 영상 라이브러리 저장 위치 (기본 `data/output/library`) |

아키텍처와 저장소 구조는 [CLAUDE.md](CLAUDE.md)를, 브랜치 전략과 커밋 컨벤션은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
