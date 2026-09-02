# TODO 목록

기획서 또는 소스 코드에서 아직 확정되지 않은 결정 사항 및 미구현 항목.
모듈별로 정리하며, 각 항목은 관련 소스 라인에 링크된다.

---

## speaker.py — 화자 색상 매핑

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 1 | [speaker.py:3](../src/hearing_to_seeing/speaker.py#L3) | **색상 배정 방식 미확정.** 현재는 단순 팔레트 순서 배정 방식 사용 중. 기획서에서 RAG 기반(인물 정보 → LLM 색상 결정) 방식이 대안으로 제시되었으나 최종 결정이 없음. | 기획서 §다음 논의 필요 사항 |
| 2 | [speaker.py:5](../src/hearing_to_seeing/speaker.py#L5) | **ASS BGR hex 값 오류 가능성.** ASS 색상 형식은 `&HAABBGGRR`임. Wong 팔레트의 Orange (RGB `#E69F00`)는 ASS에서 `&H00009FE6`이어야 하며 `&H00E69F00`은 잘못된 값임. 팔레트 전체 항목 검수 필요. | 기획서 §4 (색맹 팔레트) |
| 3 | [speaker.py:20](../src/hearing_to_seeing/speaker.py#L20) | **색상이 라벨 정렬 순서 기준으로 배정됨.** 화자가 알파벳순(`SPEAKER_00`, `SPEAKER_01`, …)으로 정렬되어 대화 등장 순서와 색상이 자연스럽게 대응되지 않을 수 있음. 첫 등장 순서 기반 배정 방식 검토 필요. | — |

---

## sync.py — 단어 단위 싱크

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 4 | [sync.py:14](../src/hearing_to_seeing/sync.py#L14) | **`\k` 와이프 vs `\kf` 채우기.** 현재 태그(`\k`)는 좌→우 색상 와이프 효과를 적용함. 기획서는 "색이 채워지는" fill 효과를 요구하며, 이는 `\kf` 태그에 해당함. `sync.py`와 `ass.py` 두 곳 모두 수정 필요. | 기획서 §4 (싱크 — 색 채우기 애니메이션) |
| 5 | [sync.py:16](../src/hearing_to_seeing/sync.py#L16) | **위로 뜨는 효과 미구현.** 기획서는 "글자가 살짝 위로 뜸" 효과를 요구함. ASS Dialogue 라인에서 단어별 `\move` 또는 `\pos`+`\an` 애니메이션으로 구현해야 함. | 기획서 §4 (싱크 — 위치 이동) |

---

## volume.py — 진폭 → 글자 크기

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 6 | [volume.py:5](../src/hearing_to_seeing/volume.py#L5) | **`FONT_SIZE_MIN`/`FONT_SIZE_MAX` 값 미확정.** 현재 값(24/48)은 접근성 표준을 근거로 설정된 것이 아님. Netflix Timed Text Style Guide, BBC Subtitle Guidelines, WCAG 등을 조사한 후 최종 값 확정 필요. | 기획서 §다음 논의 필요 사항 (표준 자막 크기 기준 조사) |
| 7 | [volume.py:19](../src/hearing_to_seeing/volume.py#L19) | **전체 정규화로 인한 국소 음량 대비 손실.** 속삭이는 장면과 소리치는 장면 모두 동일한 `FONT_SIZE_MIN`–`FONT_SIZE_MAX` 범위에 매핑됨. 장면별 또는 화자별 정규화 방식 검토 필요. | 기획서 §4 (크기 — 음량 강조) |

---

## converter/ass.py — ASS 자막 생성

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 8 | [ass.py:8](../src/hearing_to_seeing/converter/ass.py#L8) | **해상도 1920×1080 고정.** 파라미터로 받거나 `ffprobe`로 입력 영상 해상도를 자동 감지하도록 개선 필요. | — |
| 9 | [ass.py:43](../src/hearing_to_seeing/converter/ass.py#L43) | **`\k` vs `\kf` — sync.py #4와 동일한 문제.** sync.py 결정 후 이 파일도 함께 수정해야 함 (두 파일이 독립적으로 카라오케 태그를 생성하고 있음). | 기획서 §4 (싱크) |
| 10 | [ass.py:45](../src/hearing_to_seeing/converter/ass.py#L45) | **최대 줄 길이 제한 없음.** 화자가 많은 단어를 연속으로 발화하면 가독성이 심각하게 떨어지는 긴 자막 줄이 생성됨. 단어 수 또는 글자 수 기준으로 여러 `Dialogue` 이벤트로 분할하는 로직 필요. | 기획서 §10 (자막 서식의 가독성) |

---

## stt.py — WhisperX STT + 화자 분리

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 11 | [stt.py:34](../src/hearing_to_seeing/stt.py#L34) | **Whisper 모델 크기 고정(`large-v3`).** 속도·정확도 트레이드오프를 위해 호출자가 모델 크기를 선택할 수 있도록 파라미터화 필요. | 기획서 §8 (화자 분리·STT) |
| 12 | [stt.py:38](../src/hearing_to_seeing/stt.py#L38) | **HuggingFace 토큰 미연결.** `DiarizationPipeline`은 pyannote 모델 사용을 위해 `use_auth_token`이 필요함. 파라미터 또는 환경 변수 `HF_TOKEN`으로 주입하는 방식 추가 필요. | — |
| 13 | [stt.py:40](../src/hearing_to_seeing/stt.py#L40) | **화자 수 상한 미설정.** 기획서 §10 리스크 대응방안에서 `max_speakers` 상한을 두어 유사한 목소리에서 발생하는 오분류를 줄이도록 권고함. | 기획서 §10 (화자 분리 정확도) |
| 14 | [stt.py:77](../src/hearing_to_seeing/stt.py#L77) | **정렬 실패 단어 무음 삭제.** 강제 정렬 타임스탬프가 없는 단어는 로그 없이 조용히 제거됨. 어느 구간이 누락되었는지 진단하기 어려움. 경고 로그 또는 플래그 처리 필요. | 기획서 §10 (STT 정확도) |

---

## pipeline.py — 전체 파이프라인 오케스트레이션

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 15 | [pipeline.py:15](../src/hearing_to_seeing/pipeline.py#L15) | **샘플레이트 불일치 위험.** ffmpeg에서 44100 Hz로 리샘플링하지만 WhisperX `load_audio`는 16000 Hz를 기대함. ffmpeg `-ar` 값 검토 및 수정 필요. | — |
| 16 | [pipeline.py:37](../src/hearing_to_seeing/pipeline.py#L37) | **화자 라벨 수동 보정 기능 없음.** 기획서 §10에서 화자 오분류에 대한 수동 보정 옵션을 리스크 대응방안으로 제시했으나 아직 미구현. | 기획서 §10 (화자 분리 정확도) |

---

## 미확정 기획 결정 사항 (기획서 §다음 논의 필요 사항)

코드 문제가 아닌 구현에 영향을 미치는 미결 제품 결정 사항:

- **색상 배정 방식** — 단순 팔레트 배정 vs RAG 기반 LLM 색상 결정 최종 확정 필요 (TODO #1 선행 조건).
- **글자 크기 범위** — 접근성 표준 자막 크기 지침 조사 후 `FONT_SIZE_MIN`/`FONT_SIZE_MAX` 최종값 확정 (TODO #6 선행 조건).
- **사용자 검증** — 청각장애인 대상 인터뷰·테스트 세션을 통해 색상·싱크·크기 효과의 실효성 검증 일정 수립 (모든 UX 튜닝의 선행 조건).
