# TODO 목록

기획서 또는 소스 코드에서 아직 확정되지 않은 결정 사항 및 미구현 항목.
모듈별로 정리하며, 각 항목은 관련 소스에 링크된다.

---

## design/speaker.py — 화자 색상 매핑

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 1 | [speaker.py:131](../src/hearing_to_seeing/design/speaker.py#L131) | **색상 선호의 출처 미확정.** 색상환 배정(`assign_hues()`)은 구현됐으나, 각 인물이 어느 hue를 선호하는지 정하는 단계가 비어 있음. 파이프라인 [6]b는 1단계 의상 대표색([4]e), 2단계 RAG 성격·역할을 제시하나 둘 다 미구현. | 기획서 §다음 논의 필요 사항 |
| 2 | [speaker.py:131](../src/hearing_to_seeing/design/speaker.py#L131) | **선호가 없을 때의 배정이 라벨 정렬 순서에 의존.** 화자가 알파벳순(`SPEAKER_00`, `SPEAKER_01`, …)으로 정렬되어 색상환에 배치되므로, 대화 등장 순서와 색상이 자연스럽게 대응되지 않을 수 있음. 첫 등장 순서 기반 배정 검토 필요. | — |
| 2-1 | [speaker.py:28](../src/hearing_to_seeing/design/speaker.py#L28) | **`LIGHTNESS` 0.75와 채도 비율(0.95 / 0.32)이 미검증.** 기존 Wong 팔레트의 밝기 대역에 맞춰 정한 값이며, 실제 영상 위 — 특히 밝은 장면 — 에서의 가독성은 확인되지 않음. | 기획서 §10 (자막 서식의 가독성) |
| 2-2 | [speaker.py:51](../src/hearing_to_seeing/design/speaker.py#L51) | **`MIN_CONFIDENCE` 임계값 0.7은 경험값.** 전략이 내놓은 답을 채택할 최소 신뢰도. 실제 전략(RAG/VLM)을 붙인 뒤 신뢰도별 실제 정확도를 측정해 재조정 필요. | — |
| 2-3 | [speaker.py:40](../src/hearing_to_seeing/design/speaker.py#L40) | **색맹 안전성 미고려.** 표현력을 위해 팔레트 제한을 없애면서 색맹 안전 보장도 함께 사라짐. 필요 시 색상환 결과를 검증된 팔레트로 스냅하는 별도 옵션으로 분리 검토. | — |

---

## design/volume.py — 진폭 → 글자 크기

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 3 | [volume.py:5](../src/hearing_to_seeing/design/volume.py#L5) | **`FONT_SIZE_MIN`/`FONT_SIZE_MAX` 값 미확정.** 현재 값(24/48)은 접근성 표준을 근거로 설정된 것이 아님. Netflix Timed Text Style Guide, BBC Subtitle Guidelines, WCAG 등을 조사한 후 최종 값 확정 필요. | 기획서 §다음 논의 필요 사항 (표준 자막 크기 기준 조사) |
| 4 | [volume.py:18](../src/hearing_to_seeing/design/volume.py#L18) | **전체 정규화로 인한 국소 음량 대비 손실.** 속삭이는 장면과 소리치는 장면 모두 동일한 `FONT_SIZE_MIN`–`FONT_SIZE_MAX` 범위에 매핑됨. 장면별 또는 화자별 정규화 방식 검토 필요. | 기획서 §4 (크기 — 음량 강조) |

---

## design/sync.py — 단어 단위 싱크

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 5 | [sync.py:17](../src/hearing_to_seeing/design/sync.py#L17) | **위로 뜨는 효과가 실제 위치 이동이 아님.** ASS의 `\move`/`\pos`는 Dialogue 이벤트 전체에만 적용되므로, 한 줄 안의 단어 하나를 옮기려면 단어별 이벤트 분할과 글꼴 메트릭 기반 x 좌표 계산이 필요함. 현재는 `\t`로 `\fscy`(위로 자람) + `\shad`(그림자 이격)를 애니메이션해 부양감만 근사함. 사용자 검증 후 실제 좌표 이동 구현 여부 결정. | 기획서 §4 (싱크 — 위치 이동) |

---

## converter/ass.py — ASS 자막 생성

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 6 | [ass.py:23](../src/hearing_to_seeing/converter/ass.py#L23) | **해상도 1920×1080 고정.** `PLAY_RES` 상수로 모아 두어 프리뷰 렌더러와는 일치하지만 여전히 하드코딩임. 파라미터로 받거나 `ffprobe`로 입력 영상 해상도를 자동 감지하도록 개선 필요. | — |
| 7 | [ass.py:26](../src/hearing_to_seeing/converter/ass.py#L26) | **Default 스타일 폰트가 한글 미지원.** `[V4+ Styles]`의 폰트가 Arial이라 한글 자모를 그리지 못하며, 현재는 렌더 시 `--force-style "FontName=…"`로 우회 중. 한글 지원 폰트를 기본값으로 확정하고 배포 방식(시스템 폰트 의존 vs 번들) 결정 필요. | 기획서 §10 (자막 서식의 가독성) |

---

## stt.py — WhisperX STT + 화자 분리

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 8 | [stt.py](../src/hearing_to_seeing/stt.py) | **정확도 테스트 후 개선 필요.** 실제 음성으로 STT·화자 분리 정확도를 측정한 뒤 개선 방향 결정. | 기획서 §10 (STT·화자 분리 정확도) |

---

## pipeline.py — 전체 파이프라인 오케스트레이션

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 9 | [pipeline.py:93](../src/hearing_to_seeing/pipeline.py#L93) | **화자 라벨 수동 보정 기능 도입 여부 미확정.** 기획서 §10에서 화자 오분류 리스크 대응방안으로 제시되었으나, 색상 배정 방식(TODO #1)과 마찬가지로 넣을지 말지 자체가 아직 결정되지 않음. | 기획서 §10 (화자 분리 정확도) |

---

## cli.py / pipeline.py — `from-json` 임시 경로

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 10 | [pipeline.py:43](../src/hearing_to_seeing/pipeline.py#L43), [cli.py:68](../src/hearing_to_seeing/cli.py#L68) | **`from-json`(`run_from_json`)은 GPU 없는 환경을 위한 임시 우회 경로.** STT·정렬·화자 분리(WhisperX, GPU 필요)를 건너뛰고 외부에서 만든 transcript JSON을 읽어 나머지 파이프라인만 로컬에서 돌린다. GPU 환경 구성(WhisperX 의존성, CUDA 등)을 마치고 `run` 경로가 로컬에서 정상 동작하는 것을 확인하면, `from-json` 서브커맨드와 `run_from_json` 관련 코드를 제거해야 한다. | — |

---

## 미확정 기획 결정 사항 (기획서 §다음 논의 필요 사항)

코드 문제가 아닌 구현에 영향을 미치는 미결 제품 결정 사항:

- **색상 배정 방식** — 단순 팔레트 배정 vs RAG 기반 LLM 색상 결정 최종 확정 필요 (TODO #1 선행 조건).
- **글자 크기 범위** — 접근성 표준 자막 크기 지침 조사 후 `FONT_SIZE_MIN`/`FONT_SIZE_MAX` 최종값 확정 (TODO #3 선행 조건).
- **화자 라벨 수동 보정 기능 도입 여부** — 화자 오분류 리스크 대응방안으로 기획서에 제시되었으나 실제로 넣을지 자체가 미결정 (TODO #9 선행 조건).
- **사용자 검증** — 청각장애인 대상 인터뷰·테스트 세션을 통해 색상·싱크·크기 효과의 실효성 검증 일정 수립 (모든 UX 튜닝의 선행 조건).

