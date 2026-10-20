# hearing-to-seeing

음성의 특성(화자, 타이밍, 음량)을 동적/키네틱 자막으로 변환하는 서비스입니다. 화자별 색상(작품 등장인물 정보 기반), 단어 단위 싱크 애니메이션, 음량에 따른 움직임(큰 소리는 커지고, 속삭임은 작아짐)으로 청각장애인 및 난청인을 위한 자막을 만듭니다. 출력 포맷으로는 애니메이션·서식 기능을 지원하는 ASS(Advanced SubStation Alpha)를 사용합니다.

## 설치 및 실행

모두 Docker로 띄웁니다. [`.env.example`](.env.example)을 `.env`로 복사해 값을 채운 뒤, 저장소 루트에서 실행합니다.

```bash
# 이미지 빌드 (ffmpeg, Node.js, 변환 파이프라인 포함)
docker compose build

# CLI — WAV를 받아 같은 폴더에 sample.ass 생성 (./data가 컨테이너에 연결됨)
# (WAV 만들기: docker compose run --rm --entrypoint ffmpeg cli -i data/input/sample.mp4 -ac 1 -ar 16000 data/input/sample.wav)
docker compose run --rm cli data/input/sample.wav

# CLI — 옵션 전부: 출력 경로, 중간 JSON, 언어, 화자 수(아는 경우), 원본 영상(GEMINI_API_KEY 필요, 화면을 보고 화자 확인)
docker compose run --rm cli data/input/sample.wav -o data/output/sample.ass --json data/output/sample.json \
    --language ko --speakers 6 --video data/input/sample.mp4

# 웹 — http://127.0.0.1:8080 (YouTube 링크로 자막을 만들고, 만들어 둔 영상을 검색·장르로 찾아 YouTube 플레이어에서 자막과 함께 재생)
docker compose --profile web up web
# 워커 — 웹이 큐에 넣은 영상을 하나씩 변환 (웹과 함께 실행)
docker compose up -d worker
```

테스트는 [`uv`](https://docs.astral.sh/uv/)로 돌립니다 (Python 3.12): `uv sync --all-extras && uv run pytest`.

## 배포

웹은 Vercel에서 돌아가요(`index.py`, 기본 의존성만). 워커는 WhisperX 서버가 있는 머신에서 Docker로 돌아가요(`pipeline` extra 포함). 둘은 Supabase `clips` 테이블로만 이어져요. 웹은 요청을 `queued` 행으로 넣기만 하고, 워커가 그 행을 가져가 변환해요. 그래서 워커 머신이 꺼져 있어도 사이트는 떠 있고, 새 요청은 워커가 켜질 때까지 기다려요.

### 코드를 바꾼 뒤

1. **웹 배포:** 저장소 루트에서 실행해요.
   ```bash
   vercel deploy --prod
   ```
   `vercel git connect`로 저장소를 연결해 두었다면 `main`에 푸시하는 것만으로 배포돼요. 배포가 끝나면 https://hearing-to-seeing.vercel.app 에서 목록과 재생 화면이 열리는지 확인해요.
2. **워커 배포:** WhisperX 머신에서 실행해요.
   ```bash
   git pull
   docker compose up -d --build     # 다시 빌드하고 교체
   docker compose logs -f worker    # 로그
   ```
   처리 중인 영상이 있을 때 다시 올리면 그 영상은 "서버 재시작으로 처리가 중단되었습니다"로 끝나요. 트레이에 진행 중인 작업이 없을 때 올리는 게 좋아요.
3. **확인:** 사이트에서 짧은 YouTube 링크로 자막을 요청해요. 트레이에 "순서를 기다리는 중"이 떴다가 "완성 · 보러 가기"로 바뀌는지 봐요. 실패하면 화면에는 짧은 안내만 나오고, 원인은 `docker compose logs worker`의 `처리 실패` 아래 traceback에 남아요.

### 처음 한 번만

- **웹:** `vercel login`과 `vercel link`를 하고, Vercel 환경 변수에 `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `ADMIN_TOKEN`을 등록해요(`vercel env add <이름> production`). 웹은 WhisperX와 Gemini를 부르지 않으니 그 값들은 필요 없어요.
- **워커:** WhisperX 머신에서 `.env`를 채워요(아래 환경 변수 참고). `WHISPERX_API_URL`은 `http://host.docker.internal:<포트>/whisperx`로 두면 컨테이너가 같은 머신의 WhisperX에 터널 없이 바로 붙어요. Linux Docker Engine을 쓴다면 WhisperX가 `0.0.0.0`에서 연결을 받아야 해요.
- 워커 컨테이너는 비root 사용자로 돌고 머신의 파일을 마운트하지 않아요. GPU STT보다 우선순위를 낮추려고 `nice -n 10`으로 실행되고, 재부팅해도 자동으로 다시 떠요(`restart: unless-stopped`).

## 환경 변수

[`.env.example`](.env.example)을 `.env`로 복사한 뒤 값을 채웁니다.

| 변수 | 필수 | 설명 |
|------|------|------|
| `WHISPERX_API_KEY` | ✅ | 원격 WhisperX 서버(STT·정렬·화자 분리) API 키 |
| `WHISPERX_API_URL` | ✅ | 원격 WhisperX 서버 주소 |
| `GEMINI_API_KEY` | | 설정하면 워커가 영상 제목으로 작품의 등장인물을 찾아 화자별 색상을 정하고, 원본 영상을 보며 발화마다 화자를 확인하고, 장르를 분류함 ([knowledge/README.md](src/hearing_to_seeing/knowledge/README.md)). 없으면 색상환에 고르게 배정. [발급](https://aistudio.google.com/apikey) (무료 티어로 충분) |
| `SUPABASE_URL` / `SUPABASE_SECRET_KEY` | 웹 ✅ | 웹 카탈로그를 저장할 Supabase 프로젝트 (`clips` 테이블). 키는 서버 전용 secret key |
| `ADMIN_TOKEN` | | 웹에서 영상을 삭제할 때 묻는 관리자 토큰. 없으면 아무도 삭제할 수 없음 |

아키텍처와 저장소 구조는 [CLAUDE.md](CLAUDE.md)를, 브랜치 전략과 커밋 컨벤션은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
