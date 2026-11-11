# trendai — 과거 → 현재 → 미래, 트렌드를 읽고 예측하는 AI

어떤 시계열이든 넣으면(검색 관심도, 위키백과 조회수, 매출, 가격, KPI 등) **trendai**가

1. **과거**: 장기 추세·계절성을 분해하고, 추세가 꺾이거나 가속한 **구조 변화 시점**과 이상치를 찾고,
2. **현재**: 트렌드가 지금 **수명주기의 어디**(태동기·성장기·정점·쇠퇴기·안정기·재부상·변동성)에 있는지와 모멘텀을 진단하고,
3. **미래**: 여러 예측 모델을 과거 데이터로 **백테스트해 자동으로 고르고 앙상블**해서 예측값과 **불확실성 구간(80%/95%)**, 상승 확률, 시나리오를 내고,
4. **해석**: 이 모든 것을 사람이 읽을 수 있는 한국어(또는 영어) 리포트로 설명합니다. 원하면 **Claude**가 해석문을 쓰고 질문에 답합니다.

결과는 터미널 요약, **HTML 리포트**(차트 포함, 단일 파일), JSON으로 받을 수 있습니다.

## 설치

```bash
pip install -e .              # numpy, pandas만 필요
pip install -e ".[llm]"       # Claude 해석/질문 기능 (anthropic SDK)
pip install -e ".[connectors]" # 위키백과/Google Trends 데이터 수집
pip install -e ".[dev]"       # 테스트 (pytest)
```

## 빠르게 써보기

```bash
# 내장 합성 샘플(가상 데이터)로 데모
python -m trendai list-samples
python -m trendai sample emerging --html out/emerging.html
python -m trendai sample all --html out/compare.html        # 9가지 트렌드 유형 비교·순위

# 내 데이터 (CSV/TSV/Excel/JSON — 날짜 열과 숫자 열을 자동 감지)
python -m trendai analyze data.csv --horizon 12 --html report.html --json report.json
python -m trendai analyze data.csv --value-col 검색량 --freq M

# 여러 시계열을 비교해서 '뜨는 트렌드' 순위 매기기
python -m trendai compare a.csv b.csv --html compare.html

# 온라인 데이터 (인터넷 필요)
python -m trendai wiki "인공지능" "메타버스" --html wiki.html      # 위키백과 월별 조회수
python -m trendai trends "생성형 AI" --geo KR                      # Google Trends (pytrends)

# Claude가 해석문을 작성 / 분석 결과에 대해 질문
export ANTHROPIC_API_KEY=...        # 또는 `ant auth login`
python -m trendai analyze data.csv --llm
python -m trendai ask data.csv "앞으로 6개월 동안 어떻게 될까? 위험 요인은?"
```

CSV 예시 (열이 여러 개면 각각 분석하고 비교 순위까지 만듭니다):

```csv
날짜,검색량,판매량
2023-01,120,40
2023-02,135,38
...
```

## 파이썬에서 사용

```python
from trendai import run, run_many, rank_trends, load_table, render_html

series = load_table("data.csv")["검색량"]      # 또는 아무 pd.Series (DatetimeIndex)
report = run(series, horizon=12)

report.analysis.phase.label_ko          # '성장기'
report.analysis.changepoints            # 구조 변화 시점들
report.analysis.metrics.cagr_pct        # 연평균 성장률
report.forecast.to_frame()              # 예측값 + 80/95% 구간
report.outlook.prob_increase            # 상승 확률
print(report.narrative)                 # 과거/현재/미래 해석
render_html(report, "report.html")

reports = run_many(load_table("many_columns.csv"))
print(rank_trends(reports))             # 상승 트렌드 순위
```

## 어떻게 동작하나

| 단계 | 방법 |
|---|---|
| 데이터 정규화 | 주기(일/주/월/분기/년) 자동 감지, 정렬·중복 집계·결측 보간 |
| 추세/계절 분해 | STL 방식의 강건한 가법 분해 (LOESS 추세 + 위상별 계절 성분) |
| 구조 변화 | 비계절화 시계열의 구간별 선형 분할 + BIC 벌점 → 가속/감속/반전/수준 변화 |
| 추세 유의성 | Mann-Kendall 검정, Theil-Sen 기울기 |
| 수명주기 진단 | 성장률·곡률·정점 위치·낙폭·최근 vs 장기 기울기 기반 점수 |
| 예측 모델 | 단순/계절 나이브, 드리프트, 지수평활(SES·Holt·감쇠 Holt·Holt-Winters), Theta, 선형추세+푸리에, AR, **릿지 회귀(ML)**, **과거 유사 패턴(kNN 아날로그)**, **S-커브(로지스틱 성장)**, 로그 공간 변형 |
| 모델 선택 | 롤링 원점 백테스트(MASE) → 상위 모델 역오차 가중 앙상블 |
| 불확실성 | 백테스트 오차의 예측 시점별 분포로 80%/95% 구간 보정 |
| 해석 | 규칙 기반 한국어/영어 내러티브, 선택적으로 Claude(`claude-opus-5-5`) |

자세한 모듈 구조는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 참고하세요.

## Claude 연동 메모

- `--llm`/`ask`는 공식 `anthropic` SDK로 `claude-opus-5-5`를 호출합니다. 모델에는 **계산된 사실(JSON)만** 전달하고, 주어진 사실 밖의 사건을 단정하지 말고 불확실성을 밝히도록 지시합니다.
- 안전 분류기 거절 시 서버 측 대체 모델로 자동 재시도하는 `fallbacks="default"`(베타 `server-side-fallback-2026-07-01`)가 켜져 있습니다.
- 키가 없거나 호출이 실패하면 자동으로 규칙 기반 해석으로 대체합니다.

## 한계와 주의

- 예측은 과거 패턴이 이어진다는 가정에 기반합니다. 정책·사건 같은 외부 충격은 반영되지 않으며, 예측 구간은 먼 미래일수록 넓어집니다.
- 내장 샘플은 **합성 데이터**입니다(실제 데이터 아님).
- 투자·사업 판단의 유일한 근거로 쓰지 마세요.

## 테스트

```bash
python -m pytest
```
