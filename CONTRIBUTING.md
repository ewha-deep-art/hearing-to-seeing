# CONTRIBUTING

이 문서는 팀 내 Git 협업 방식을 정리한 문서입니다.

## 브랜치 전략

- `main` — 배포/안정 브랜치
- `dev` — 개발 통합 브랜치. 모든 feature 브랜치의 base
- `feature/*` — 기능 개발 브랜치. `dev`에서 분기

### 워크플로우

1. `dev`를 base로 `feature/기능명` 브랜치를 생성한다.
2. 작업이 끝나면 `feature/*` → `dev`로 merge한다.
3. `dev` → `main` merge는 팀 전체가 상의하여 결정한다. (임의로 merge하지 않음)

## 커밋 메시지 컨벤션

타입은 아래 세 가지만 사용한다. 그 외 타입(`chore`, `refactor`, `style`, `test` 등)은 사용하지 않는다.

- `feat` — 기능 추가/변경
- `fix` — 버그 수정
- `docs` — 문서 수정

형식: `<type>: <description>`

```
feat: 화자별 색상 매핑 로직 추가
fix: 볼륨 정규화 시 0으로 나누는 버그 수정
docs: README에 설치 방법 추가
```

## 작업 목록

상세 작업 필요 목록은 [docs/TODO.md](docs/TODO.md)를 확인한다.
