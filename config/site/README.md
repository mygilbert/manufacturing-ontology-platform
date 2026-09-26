# config/site/ — 사이트 오버레이 (Git 추적 제외)

이 디렉토리는 **사내 고유 정보만** 담는다. `.gitignore` 로 전체가 제외되며,
`*.example.yaml` 과 이 README 만 커밋된다.

## 왜 분리하는가

익명화는 커밋할 때마다 "이건 지워도 되나"를 판단해야 한다.
수백 번 판단하면 몇 번은 반드시 틀린다.
분리는 한 번만 판단하면 된다. 코어에 사내 값이 **애초에 없으면** 실수할 여지가 없다.

## 무엇이 여기 들어가는가

| 위치 | 내용 |
|---|---|
| `mappings/` | 실제 테이블·컬럼명, 파일 경로, 파라미터 ID 체계 |
| `knowledge/` | 채워진 전문가 지식 엑셀 (양식은 코어에 있다) |
| `secrets/` | 접속 정보. `.env` 류 |

## 무엇이 여기 들어가면 안 되는가

범용 로직, 표준 기반 스키마, 방법론. 그런 것은 코어에 둔다.
판단 기준은 `docs/09_배포_경계.md` 참조.

## 시작하기

```bash
cp config/site/mappings/battery.example.yaml config/site/mappings/battery.yaml
# 사내 실제 테이블/컬럼명으로 채운다
python scripts/profile_site_data.py --landing /path/to/landing
```
