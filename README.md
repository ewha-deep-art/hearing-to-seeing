# hearing-to-seeing

음성의 특성(화자, 타이밍, 음량)을 동적/키네틱 자막으로 변환하는 서비스입니다. 화자별 색상, 단어 단위 싱크 애니메이션, 음량에 따른 글자 크기 변화로 청각장애인 및 난청인을 위한 자막을 만듭니다. 출력 포맷으로는 애니메이션·서식 기능을 지원하는 ASS(Advanced SubStation Alpha)를 사용합니다.

## 설치 및 실행

이 프로젝트는 패키지 관리에 [`uv`](https://docs.astral.sh/uv/)를 사용합니다 (Python 3.12).

```bash
# 의존성 설치
uv sync

# 자막 생성 (원격 WhisperX API로 STT + 화자 분리, .env에 WHISPERX_API_KEY 필요)
uv run hearing-to-seeing run data/input/sample.mp4 --json

# 자막을 영상에 입혀 미리보기 생성
uv run python -m hearing_to_seeing.render data/input/sample.mp4 data/input/sample.ass data/output/preview.mp4
```

아키텍처와 저장소 구조는 [CLAUDE.md](CLAUDE.md)를, 브랜치 전략과 커밋 컨벤션은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
