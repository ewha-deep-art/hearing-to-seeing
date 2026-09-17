# hearing-to-seeing

음성의 특성(화자, 타이밍, 음량)을 동적/키네틱 자막으로 변환하는 서비스입니다. 화자별 색상, 단어 단위 싱크 애니메이션, 음량에 따른 글자 크기 변화로 청각장애인 및 난청인을 위한 자막을 만듭니다. 출력 포맷으로는 애니메이션·서식 기능을 지원하는 ASS(Advanced SubStation Alpha)를 사용합니다.

## 설치 및 실행

이 프로젝트는 패키지 관리에 [`uv`](https://docs.astral.sh/uv/)를 사용합니다 (Python 3.12).

```bash
# 의존성 설치
uv sync

# 자막 생성 (기본 경로: WhisperX STT + 화자 분리 실행, GPU 필요)
uv run hearing-to-seeing run data/input/sample.mp4 --json

# 작품 정보 함께 기록 (화자 색상 단계에서 인물 정보를 찾는 데 사용됨)
uv run hearing-to-seeing run data/input/sample.mp4 --title "기생충" --url "https://youtu.be/VIDEO_ID"

# 기존 transcript로부터 생성 (GPU 없는 환경을 위한 임시 대안)
uv run hearing-to-seeing from-json data/input/sample.mp4 transcript.json

# 자막을 영상에 입혀 미리보기 생성
uv run python -m web.render data/input/sample.mp4 data/input/sample.ass data/output/preview.mp4

# 테스트 실행
uv run pytest

# 특정 테스트 파일만 실행
uv run pytest tests/path/to/test_file.py

# 의존성 추가
uv add <package>
```

아키텍처와 저장소 구조는 [CLAUDE.md](CLAUDE.md)를, 브랜치 전략과 커밋 컨벤션은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
