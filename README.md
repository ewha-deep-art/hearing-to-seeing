# hearing-to-seeing

음성의 특성(화자, 타이밍, 음량)을 동적/키네틱 자막으로 변환하는 서비스입니다. 화자별 색상, 단어 단위 싱크 애니메이션, 음량에 따른 글자 크기 변화로 청각장애인 및 난청인을 위한 자막을 만듭니다. 출력 포맷으로는 애니메이션·서식 기능을 지원하는 ASS(Advanced SubStation Alpha)를 사용합니다.

## 설치 및 실행

이 프로젝트는 패키지 관리에 [`uv`](https://docs.astral.sh/uv/)를 사용합니다 (Python 3.12).

```bash
# 의존성 설치
uv sync

# 출력(.ass / .json)은 따로 지정하지 않으면 data/output/<입력 파일명>… 에 저장된다

# 자막 생성 (기본 경로: WhisperX STT + 화자 분리 실행, GPU 필요)
uv run hearing-to-seeing run data/input/sample.mp4 --json

# 작품 정보 함께 기록 (화자 색상 단계에서 인물 정보를 찾는 데 사용됨)
uv run hearing-to-seeing run data/input/sample.mp4 --title "기생충" --url "https://youtu.be/VIDEO_ID"

# 화자에 인물 이름 직접 지정
uv run hearing-to-seeing run data/input/sample.mp4 --speaker-map "SPEAKER_00=기택,SPEAKER_01=충숙"

# 제목으로 작품을 특정해 위키백과·나무위키 문서를 읽고, 등장인물·색상 근거·화자→인물 매핑을 정한다
# (Gemini API 무료 티어 키 GEMINI_API_KEY 필요, .env 참고). --speaker-map 을 함께 주면 그 화자는 추론보다 우선한다.
uv run hearing-to-seeing run data/input/sample.mp4 --json --lookup \
  --title "[4K] 기생충 명장면 | 아들아 너는 계획이 다 있구나"

# 영상 없이 인물 검색·색상·화자 매핑만 테스트 (transcript JSON → data/output/<이름>.lookup.json, 음량 단계는 건너뜀)
uv run hearing-to-seeing lookup "data/input/result_[왕과_사는_남자]_공식_예고편.json"

# 결과 JSON 의 character_map 에서 틀린 이름을 고친 뒤 다시 돌리면 고친 매핑이 확정으로 채택된다.
# (work · character_map 은 재사용되어 검색·추론을 다시 하지 않음. 처음부터 다시 하려면 --fresh)
uv run hearing-to-seeing from-json data/input/sample.mp4 data/output/sample.json --json

# 기존 transcript로부터 생성 (GPU 없는 환경을 위한 임시 대안)
# 중간 JSON, 세그먼트 export({video_title, video_url, segments}), 글자 단위 export 모두 읽는다
uv run hearing-to-seeing from-json data/input/sample.mp4 transcript.json

# 자막을 영상에 입혀 미리보기 생성
uv run python -m web.render data/input/sample.mp4 data/output/sample.ass data/output/preview.mp4

# 테스트 실행
uv run pytest

# 특정 테스트 파일만 실행
uv run pytest tests/path/to/test_file.py

# 의존성 추가
uv add <package>
```

전체 파이프라인 설계는 [docs/PIPELINE.md](docs/PIPELINE.md)를, 아키텍처와 저장소 구조는 [CLAUDE.md](CLAUDE.md)를, 브랜치 전략과 커밋 컨벤션은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
