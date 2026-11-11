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
pip install openpyxl          # .xlsx 파일 읽기 (.xls는 xlrd)
```

## 빠르게 써보기

```bash
# 내장 합성 샘플(가상 데이터)로 데모
python -m trendai list-samples
python -m trendai sample emerging --html out/emerging.html
python -m trendai sample all --html out/compare.html        # 9가지 트렌드 유형 비교·순위 (+ 개별 리포트)

# 내 데이터 (CSV/TSV/TXT/Excel/JSON — 인코딩·구분자·날짜 열·숫자 열을 자동 감지)
python -m trendai analyze data.csv --horizon 12 --html report.html --json report.json
python -m trendai analyze data.csv --value-col 검색량 --freq M
python -m trendai analyze data.csv --date-col 날짜 --value-col 검색량 --value-col 판매량

# 여러 시계열을 비교해서 '뜨는 트렌드' 순위 매기기 (주기가 달라도 연환산해서 비교)
python -m trendai compare a.csv b.csv --html compare.html
python -m trendai compare a.csv --samples emerging,fad

# 온라인 데이터 (인터넷 필요)
python -m trendai wiki "인공지능" "메타버스" --html wiki.html      # 위키백과 월별 조회수
python -m trendai trends "생성형 AI" --geo KR                      # Google Trends (pytrends 필요)

# Claude가 해석문을 작성 / 분석 결과에 대해 질문
export ANTHROPIC_API_KEY=...        # 또는 `ant auth login`
python -m trendai analyze data.csv --llm
python -m trendai ask data.csv "앞으로 6개월 동안 어떻게 될까? 위험 요인은?"
```

CSV 예시 (값 열이 여러 개면 각각 분석하고 비교 순위까지 만듭니다):

```csv
날짜,검색량,판매량
2023-01,120,40
2023-02,135,38
...
```

- 시계열이 하나면 해석문까지 출력하고 HTML 리포트 1개를 씁니다. 여러 개면 요약 + 순위를 출력하고
  (해석문은 HTML/JSON에 들어 있음), `--html out.html`이면 비교 페이지 `out.html`과 개별 리포트
  `out_<이름>.html`을 씁니다. `--json`은 시계열 1개면 객체, 여러 개면 배열입니다.
- `--lang en`은 해석문과 CLI 안내 문구를 영어로 바꿉니다(터미널 요약표와 HTML 리포트는 한국어).
- 데이터를 읽으면서 바꾼 점은 stderr에 한 줄씩 알려 줍니다: `안내:`(어떻게 해석했는지 — 감지한 날짜 열·형식,
  제외한 행 번호 열, 모호한 일/월 순서 등), `경고:`(값을 버리거나 메운 경우 — 읽을 수 없는 값, 보간한
  빈 기간, 평일만 있는 데이터, 분석에 실패한 시계열 등).
- 종료 코드: `0` 성공, `1` 실행 실패(네트워크·Claude·예기치 못한 오류), `2` 입력 오류(파일·데이터·인자).

### 읽을 수 있는 데이터

| 항목 | 지원 형식 |
|---|---|
| 파일 | `.csv` `.tsv` `.txt`(구분자 `,` `;` 탭 `\|` 자동 감지), `.xlsx`(openpyxl), `.xls`(xlrd), `.json`(레코드 목록, `{날짜: 값}`, `DataFrame.to_json()`의 columns/index/split/table, JSON Lines) |
| 인코딩 | UTF-8(BOM 유무), UTF-16(엑셀 "유니코드 텍스트"), CP949/EUC-KR(한국어 엑셀·공공데이터) |
| 날짜 | `2024-01-05`, `2024-01`, `2024.01`·`2024. 1. 5.`(KOSIS·한국어 엑셀), `202401`·`20240105`(ECOS), `2024년 1월`, `2024년 1분기`·`2024 1/4분기`, `2024 Q1`·`2024Q1`·`Q1 2024`, `2024-W01`(ISO 주), `05.01.2024`·`01/05/2024`(일/월 순서는 열 전체를 보고 한 번에 결정, 모호하면 안내), `Jan 2024`, 연도(순서·중복 무관), Unix 초/밀리초, 시간대 오프셋이 붙은 시각(현지 시각 유지) |
| 숫자 | `1,234`, `1.234,5`(소수점 쉼표, `;` 구분 CSV), `1 234`, `12%`, `$1,234`, `1,234원`, `(1,234)`(음수), `1.2억`·`3,400만`·`12K`·`1.5M`, 결측 표시 `-` `NA` `#N/A` 등 |
| 표 모양 | 날짜 열 1개 + 값 열 N개(세로형), `연도` 열 + `월`(또는 `분기`) 열, 날짜가 열 머리글에 놓인 가로형(KOSIS 기본), 머리글 없는 파일 |
| 제외하는 열 | `Unnamed: 0`(`DataFrame.to_csv()`의 행 번호), `id`·`no`·`번호`·`순번`, 0·1부터 1씩 증가하는 정수 열 (`--value-col`로 지정하면 포함) |

평일(영업일)만 있는 데이터(주가 등)는 주말을 보간해 달력일 기준으로 분석하며 이를 경고로 알립니다
(주 단위로 보려면 `--freq W`). 주간 데이터는 원래 요일(예: 월요일)을 유지하고 예측 날짜도 그 요일로 이어집니다.

## 파이썬에서 사용

```python
from trendai import run, run_many, rank_trends, load_table, render_html, render_compare_html

series = load_table("data.csv")["검색량"]      # 또는 아무 pd.Series (DatetimeIndex·PeriodIndex·연도 정수 인덱스)
report = run(series, horizon=12)

report.analysis.phase.label_ko          # '성장기'
report.analysis.changepoints            # 구조 변화 시점들
report.analysis.metrics.cagr_pct        # 연평균 성장률
report.forecast.to_frame()              # 예측값 + 80/95% 구간
report.forecast.skill                   # 백테스트 MAE / 최선 나이브 MAE (<1이면 나이브보다 정확)
report.outlook.prob_increase            # 상승 확률
print(report.narrative)                 # 과거/현재/미래 해석
render_html(report, "report.html")

reports = run_many(load_table("data.csv"))   # {이름: TrendReport}, 실패한 시계열은 경고 후 건너뜀
ranking = rank_trends(reports)               # 상승 트렌드 순위 (pandas DataFrame)
print(ranking[["name", "phase_ko", "expected_change_ann_pct", "prob_increase", "score"]])
render_compare_html(list(reports.values()), ranking, "compare.html")
```

- 입력을 쓸 수 없으면 `trendai.DataError`(한국어 메시지), 데이터를 고쳐서 쓴 경우 `trendai.DataWarning`,
  해석 방식 안내는 `trendai.DataNotice` 경고가 발생합니다.
- `load_table`이 돌려준 시계열의 `series.attrs["trendai"]`에 감지한 날짜 열·형식이 들어 있습니다.
- 순위 점수(rising score, 0–100, 50=중립)는 예상 변화·상승 확률·최근 변화·모멘텀을 섞되, 변화율은 연환산하고
  모멘텀은 월 기준으로 맞춰 일·주·월·분기 데이터를 공정하게 비교합니다. 표에는 각 리포트의 원래 값
  (`expected_change_pct` 등)과 점수 계산에 쓴 연환산 값(`expected_change_ann_pct`, `recent_change_ann_pct`,
  `momentum_per_month`)이 함께 들어 있습니다.

## 어떻게 동작하나

| 단계 | 방법 |
|---|---|
| 데이터 정규화 | 인코딩·구분자·날짜/숫자 형식 감지, 주기(일/주/월/분기/년) 자동 감지, 정렬·중복 집계·결측 보간(바꾼 점은 경고로 안내) |
| 추세/계절 분해 | STL 방식의 강건한 가법 분해 (LOESS 추세 + 위상별 계절 성분) |
| 구조 변화 | 비계절화 시계열의 구간별 선형 분할 + BIC 벌점 → 가속/감속/반전/수준 변화 |
| 추세 유의성 | Mann-Kendall 검정, Theil-Sen 기울기 |
| 수명주기 진단 | 성장률·곡률·정점 위치·낙폭·최근 vs 장기 기울기 기반 점수 |
| 예측 모델 | 단순/계절 나이브, 드리프트, 지수평활(SES·Holt·감쇠 Holt·Holt-Winters), Theta, 선형추세+푸리에, AR, **릿지 회귀(ML)**, **과거 유사 패턴(kNN 아날로그)**, **S-커브(로지스틱 성장)**, 로그 공간 변형 |
| 모델 선택 | 롤링 원점 백테스트(MASE) → 상위 모델 역오차 가중 앙상블, 나이브 대비 정확도(skill) |
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
python -m pytest            # 전체 (약 1분)
python -m pytest tests/test_data.py tests/test_cli.py
```
