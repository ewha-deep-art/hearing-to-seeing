# TODO 목록

기획서 또는 소스 코드에서 아직 확정되지 않은 결정 사항 및 미구현 항목.
모듈별로 정리하며, 각 항목은 관련 소스에 링크된다.

---

## pipeline.py — 화자 색상 매핑

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 1 | [pipeline.py](../src/hearing_to_seeing/pipeline.py) | **색상 배정 값 검증 필요** ([knowledge/README.md](../src/hearing_to_seeing/knowledge/README.md)). (1) `MIN_CONFIDENCE`(0.7)·`LIGHTNESS`(0.75)·채도 비율은 경험값 — 실제 영상 위 가독성·매핑 정확도 측정 후 조정. (2) 확인 UI(인물명·색상 수정, 화자 병합)와 수동 매핑 미구현. (3) Gemini 무료 티어가 수요 폭주(503)일 때 SDK 재시도로 수 분이 걸릴 수 있음 — 영상 검토는 첫 STT 패스 뒤 나머지 패스와 겹쳐 돌지만, 그보다 길어지거나 대사 기반 화자 추론(⑤)으로 넘어가면 작업 시간이 그만큼 늘어남. | 기획서 §다음 논의 필요 사항 |

---

## design/volume.py — 음량 → 크기

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 2 | [volume.py](../src/hearing_to_seeing/design/volume.py) | **`BASE_FONT_SIZE`(64, [layout.py](../src/hearing_to_seeing/design/layout.py))·분류 임계값 미확정.** 기본 글자 크기는 고정이고, 화자 기준 음량을 `LOUD_THRESHOLD`(0.9)/`QUIET_THRESHOLD`(0.1)로 loud/whisper/normal 세 단계로 나눠 loud=확대 후 복귀, whisper=작게 고정으로 표현함. 글자 크기와 임계값 모두 접근성 표준·사용자 검증에 근거하지 않은 값. | 기획서 §다음 논의 필요 사항 (표준 자막 크기 기준 조사) |
| 3 | [volume.py](../src/hearing_to_seeing/design/volume.py) | **전체 min-max 정규화 + 고정 임계값 조합이 한쪽으로 쏠림.** 튀는 단어 하나가 최댓값을 차지하면 나머지가 모두 낮게 눌림 — 실제 라이브러리 영상(검사외전 클립)에서 443단어 중 434개가 whisper로 분류됨(정규화 음량 중앙값 0.12). **임시 대응:** ASS 변환 직전 `relative_volumes`가 저장된 음량을 화자별 백분위(중앙값 0.5, 5·95 백분위 0·1)로 다시 잡아 분류함. 근본 해결은 `annotate_volumes`에서 원시 RMS로 화자별 dB z-score를 구하는 것 — 장면별 정규화와 함께 검토 필요. | 기획서 §4 (크기 — 음량 강조) |

---

## converter/ass.py — ASS 자막 생성

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 4 | [ass.py](../src/hearing_to_seeing/converter/ass.py) | **해상도 1920×1080 고정.** `PLAY_RES` 상수로 모아 두어 프리뷰 렌더러와는 일치하지만 여전히 하드코딩임. 파라미터로 받거나 `ffprobe`로 입력 영상 해상도를 자동 감지하도록 개선 필요. | — |

---

## stt.py — WhisperX STT + 화자 분리

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 5 | [stt.py](../src/hearing_to_seeing/stt.py) | **디코딩 옵션 일부만 동작** ([evaluation/README.md](../evaluation/README.md) 4.2절). (1) `temperatures`·`hotwords`·`repetition_penalty`가 적용되지 않음(WhisperX 배치 디코딩 한계로 추정) — 반복 루프를 디코더에서 막으려면 서버에 temperature fallback 경로가 필요. 인물표 `initial_prompt`는 재측정 후 채택 여부 결정. (2) 화자 분리: 리뷰 없이(목소리 화자 분리 `voice.diarize` + 목소리 병합) 평균 화자 정확도 0.804, Gemini 영상 검토([review.py](../src/hearing_to_seeing/review.py))까지 0.92 안팎(4.3절). 검토는 키·한도(모델당 하루 20회)가 있을 때만 돌아서, 한도가 차면 0.804로 돌아감. 유료 티어 전환이 대안. | 기획서 §10 (STT·화자 분리 정확도) |
| 9 | [review.py](../src/hearing_to_seeing/review.py), [voice.py](../src/hearing_to_seeing/voice.py) | **화자 정확도 남은 오류** ([evaluation/README.md](../evaluation/README.md) 4.2절, 평균 0.932). (1) macro P/R/F1 0.9는 단어 1~9개뿐인 단역까지 맞혀야 도달한다. 1초 미만 발화는 목소리로 확인할 수 없어 리뷰의 판단에 달려 있다. (2) 마녀 정확도 0.886. 남은 오류의 절반은 gold와 리뷰·목소리가 엇갈리는 한 발화다. (3) 리뷰는 temperature 0에서도 실행마다 조금씩 달라진다. 위키 문서가 바뀌면 캐스트와 리뷰 프롬프트도 달라진다. | 기획서 §10 (STT·화자 분리 정확도) |
| 12 | [stt.py](../src/hearing_to_seeing/stt.py) | **gold 이외 영상 검증 필요.** 현재 측정은 gold 클립에만 근거함. 다른 영상에서도 STT·화자 정확도를 확인해야 함. | 기획서 §10 (STT·화자 분리 정확도) |
| 13 | [stt.py](../src/hearing_to_seeing/stt.py) | **긴 공백 뒤 대사 타이밍 어긋남.** 공백이 큰 시점에서 대사 시작 시각이 실제와 어긋남(예: "다친 다리로 심부름에 나선 김유정, 그 모습이 마음에 걸린 박진영 \| tvN 261018 방송"의 "여기는 왜 오셨어요?"). | — |

---

## converter/ass.py — 대사 라인 분할 (미확정 기획)

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 7 | [ass.py](../src/hearing_to_seeing/converter/ass.py) | **줄 분할 기준값 미확정.** `split_into_lines`가 (1) 화자 전환, (2) 직전 단어와의 침묵이 `MAX_GAP`(0.7초) 초과, (3) 누적 폭이 `MAX_WIDTH`(전각 기준 42자 × 2행) 초과, (4) 자막 노출 시간이 `MAX_DURATION`(6초) 초과, (5) 직전 단어가 문장 종결부호(`.?!…`)로 끝남 — 이 다섯 조건 중 하나를 만나면 새 자막(Dialogue)을 시작함. 이후 `_merge_unterminated_lines`가 종결부호 없이(즉 화자 전환/침묵/폭/길이 때문에) 끊긴 줄을, 위 다섯 제약을 다시 만족하는 한도 내에서 문장을 마저 끝내는 다음 줄과 재병합함. 한 화면에 두 줄까지 허용하며, `_row_break_index`가 폭이 `ROW_WIDTH`(21자 상당)를 넘을 때 글자 수가 가장 균등하게 갈리는 지점에서 개행함. **현재 값(`MAX_GAP`/`MAX_WIDTH`/`MAX_DURATION`/`ROW_WIDTH`)은 임의로 설정된 것으로, 접근성 표준이나 사용자 검증에 근거하지 않음.** 실사용 테스트 결과에 따라 기준값 및 분할 로직 자체(예: 문장 단위 대신 구/절 단위 분할 등) 재검토 필요. | — |

---

## 웹 화면 — 실사용 테스트 피드백 반영 (미확정 기획)

| # | 파일 | 설명 | 기획서 참조 |
|---|------|------|------------|
| 8 | [src/web/](../src/web/) | **웹 UI 실사용 테스트 진행 중.** 문구, 레이아웃, 디자인 등 세부 수정 필요 — 구체적인 개선 항목은 테스트 피드백을 취합한 후 정리 예정. | — |
| 10 | [convert.py](../src/worker/convert.py), [watch.js](../src/web/static/watch.js) | **도메인 제한 영상을 요청 단계에서 못 거름.** yt-dlp `playable_in_embed`·oEmbed·임베드 클라이언트 모두 "허용"인데 플레이어에서 오류 150이 나는 영상이 있음(니켈로디언 클립). 현재는 시청 화면 안내만 하고 할당량은 쓰인 채로 남음. 배포 도메인에서 빈도를 본 뒤 플레이어 오류 시 자동 숨김 여부 결정. | — |
| 11 | [watch.js](../src/web/static/watch.js), [convert.py](../src/worker/convert.py) | **YouTube 정책 검토 필요(공개 배포 전).** 임베드 플레이어 위 자막 오버레이는 Required Minimum Functionality의 오버레이 금지 조항과, 처리 중 영상 다운로드는 API 약관 III.E.1.a와 충돌. YouTube 자동자막 숨김은 비공개 API `unloadModule`에 의존. | — |

---

## 미확정 기획 결정 사항 (기획서 §다음 논의 필요 사항)

코드 문제가 아닌 구현에 영향을 미치는 미결 제품 결정 사항:

- **색상 배정 방식** — 임계값·밝기·채도 등 세부 값과 확인 UI 여부 확정 필요 (TODO #1 선행 조건).
- **글자 크기·음량 분류 기준** — 접근성 표준 자막 크기 지침 조사 후 `BASE_FONT_SIZE`와 loud/whisper 임계값 확정 (TODO #2 선행 조건).
- **대사 라인 분할 기준** — `MAX_GAP`/`MAX_WIDTH`/`MAX_DURATION`/`ROW_WIDTH` 등 분할 기준값과 분할 방식 자체를 실사용 테스트를 통해 확정 (TODO #7 선행 조건).
- **웹 화면 문구·디자인** — 실사용 테스트 피드백을 바탕으로 문구, 레이아웃, 디자인 확정 (TODO #8 선행 조건).
- **사용자 검증** — 청각장애인 대상 인터뷰·테스트 세션을 통해 색상·싱크·크기 효과의 실효성 검증 일정 수립 (모든 UX 튜닝의 선행 조건).

