# trendai 웹 앱 운영 안내

trendai 웹 앱은 분석 엔진(`trendai` 패키지)을 브라우저에서 쓰게 해 주는 단일 프로세스 서버입니다. 표준 라이브러리
WSGI 서버와 JSON API, 빌드 과정이 없는 단일 페이지 앱(`trendai/web/static/`)으로 이루어져 있고, 실행에 필요한 의존성은
`numpy`와 `pandas`뿐입니다. 이 문서는 운영자를 위한 안내입니다. API와 화면 데이터의 정식 계약은
[`docs/WEB_SPEC.md`](WEB_SPEC.md), 모듈 구조는 [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)의 `web/` 절에 있습니다.

## 빠른 시작

```bash
pip install -e .            # 엑셀 내보내기와 AI 해석까지 쓰려면: pip install -e ".[web]"
trendai serve               # http://127.0.0.1:8000
```

실행하면 표준 출력에 `trendai 웹 앱: http://127.0.0.1:8000 (Ctrl+C로 종료)`가 찍힙니다. 주요 옵션은 다음과 같습니다.

| 옵션 | 뜻 |
|---|---|
| `--host HOST` | 접속 주소 (기본: `TRENDAI_HOST` 또는 `127.0.0.1`) |
| `--port PORT` | 포트 (기본: `TRENDAI_PORT` 또는 `8000`, `0`이면 빈 포트를 골라 실제 주소를 출력) |
| `--data-dir DIR` | 결과·캐시 저장 폴더 (기본: `TRENDAI_DATA_DIR` 또는 `./.trendai-data`) |
| `--dev` | 개발 모드: 정적 파일을 요청마다 새로 읽고 `/static/dev/`(가짜 API로 화면을 띄우는 `mock.html`)를 제공 |
| `--lang ko\|en` | 터미널 안내 문구의 언어 |

우선순위는 **명령줄 옵션 > 환경 변수 > 기본값**입니다. 종료 코드는 `0`(정상 종료), `2`(설정 오류: 잘못된 환경 변수,
공개 주소인데 연락처 없음, 데이터 폴더를 만들 수 없음), `1`(포트를 열 수 없음, 또는 종료 신호를 두 번 받음)입니다.

키 없이 실행하면 네이버 출처는 ‘준비 중’으로 보이고, 위키백과·내 파일·샘플은 바로 쓸 수 있습니다. 시작할 때 표준
오류에 다음 안내가 나올 수 있습니다.

* `경고: TRENDAI_CONTACT가 설정되지 않아 …` — 로컬 실행에서는 경고만 합니다. 공개 운영 전에 설정하세요(아래 ‘Wikimedia 연락처’).
* `안내: 네이버 API 키가 없어 네이버 출처는 ‘준비 중’으로 표시됩니다.`
* `경고: TRENDAI_TRUST_PROXY를 쓸 때는 TRENDAI_PUBLIC_URL도 설정하는 것이 안전합니다.`

## 환경 변수

불리언 값은 `1/true/yes/on`, `0/false/no/off`(대소문자 무관)를 받습니다. 값이 잘못되면 시작할 때
`오류: TRENDAI_WORKERS: expected an integer, got 'x'`처럼 변수 이름을 밝히고 종료 코드 2로 끝납니다.

| 환경 변수 | 기본값 | 뜻 |
|---|---|---|
| `TRENDAI_HOST` | `127.0.0.1` | `trendai serve`의 접속 주소. `0.0.0.0`이나 다른 외부 주소면 ‘공개’로 봅니다 |
| `TRENDAI_PORT` | `8000` | 포트 (`0` = 빈 포트) |
| `TRENDAI_DATA_DIR` | `./.trendai-data` | 저장소 폴더 (권한 `0700`으로 만듦) |
| `TRENDAI_PUBLIC_URL` | 없음 | 외부에서 보이는 주소(예: `https://trendai.example`). Wikimedia User-Agent와 Origin 검사에 쓰고, 설정하면 ‘공개’로 봅니다 |
| `TRENDAI_CONTACT` | 없음 | 운영자 이메일이나 URL. Wikimedia User-Agent에 들어가며 **공개 운영 시 필수** |
| `TRENDAI_ALLOW_NO_CONTACT` | `false` | 비공개 배포에서 연락처 요구를 명시적으로 끄기 |
| `TRENDAI_TRUST_PROXY` | `false` | 클라이언트 IP를 `X-Forwarded-For`의 마지막 값에서, 자기 주소를 `X-Forwarded-Proto`/`X-Forwarded-Host`에서 읽기 |
| `TRENDAI_DEV` | `false` | `--dev`와 같음 |
| `TRENDAI_LOG_LEVEL` | `INFO` | `CRITICAL`·`ERROR`·`WARNING`·`INFO`·`DEBUG` |
| `TRENDAI_WORKERS` | `2` | 동시에 실행하는 분석 작업 수 |
| `TRENDAI_MAX_QUEUED` | `8` | 기다릴 수 있는 분석 작업 수. 넘치면 `503 server_busy` |
| `TRENDAI_JOB_TIMEOUT` | `180` | 이 시간(초)이 지나면 다음 단계 경계에서 작업을 `job_timeout`으로 실패 처리 |
| `TRENDAI_UPLOAD_MAX_MB` | `10` | 업로드 최대 크기(MB) |
| `TRENDAI_UPLOAD_MAX_SERIES` | `20` | 업로드 한 번에 분석할 수 있는 값 열 수 |
| `TRENDAI_RESULT_TTL_H` | `24` | 결과 보관 시간(시간) |
| `TRENDAI_SHARE_TTL_D` | `30` | 공유 링크 보관 기간(일) |
| `TRENDAI_NAVER_MONTHLY_LIMIT` | `50000` | 네이버 API 월 호출 한도(예산 가드 기준) |
| `TRENDAI_NAVER_IP_PER_HOUR` | `20` | IP당 시간당 네이버 새 조회 수(캐시 적중은 세지 않음) |
| `TRENDAI_NAVER_CACHE_DAYS` | `21` | 네이버 데이터 캐시 최대 일수. 21보다 크면 21로 줄입니다 |
| `NAVER_API_HUB_KEY_ID`, `NAVER_API_HUB_KEY` | 없음 | NAVER API HUB 키(권장) |
| `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | 없음 | 레거시 네이버 Developers 키(HUB 키가 없을 때만, 2027-06-30까지) |
| `TRENDAI_AI` | `auto` | `auto`(`anthropic`이 설치돼 있고 `ANTHROPIC_API_KEY`가 있으면 켬) · `on` · `off` |
| `TRENDAI_AI_MODEL` | `claude-opus-5-5` | AI 해석에 쓸 Claude 모델 |
| `TRENDAI_AI_IP_PER_HOUR` | `10` | IP당 시간당 AI 해석 요청 수 |
| `TRENDAI_AI_DAILY_LIMIT` | `200` | 하루(한국 시간 기준) AI 해석 총 횟수 |
| `ANTHROPIC_API_KEY` | 없음 | Anthropic SDK가 직접 읽습니다 |
| `TRENDAI_HSTS` | `false` | `Strict-Transport-Security` 헤더 보내기(TLS 뒤에서만 켜세요) |
| `TRENDAI_STORE_MAX_MB` | `2048` | 저장소가 이 크기를 넘으면 오래된 결과부터 지움. 분석이 많으면 늘리세요(아래 ‘용량 산정’) |
| `TRENDAI_SHUTDOWN_GRACE` | `30` | 종료 신호 뒤 실행 중인 작업을 기다리는 최대 시간(초) |

API 키(`NAVER_*`, `ANTHROPIC_API_KEY`)는 설정 객체에 담기지 않고, 응답·로그·URL·오류 문구 어디에도 나오지 않습니다.

## 네이버 연결

trendai는 네이버 데이터랩 **검색어트렌드** API를 NAVER API HUB로 호출합니다.

1. 네이버 클라우드 콘솔에서 NAVER API HUB 애플리케이션을 만들고 키 ID와 키를 발급받습니다.
2. 그 애플리케이션에서 **검색어트렌드 API를 켭니다**. 쇼핑인사이트만 켜 두면 검색어트렌드 호출이 인증 오류(401)로 끝납니다.
3. 서버 환경에 키를 넣고 다시 시작합니다.

   ```bash
   export NAVER_API_HUB_KEY_ID=…
   export NAVER_API_HUB_KEY=…
   trendai serve
   ```

   호출 주소는 `https://naverapihub.apigw.ntruss.com/search-trend/v1/search`이고 헤더는
   `X-NCP-APIGW-API-KEY-ID` / `X-NCP-APIGW-API-KEY`입니다.
4. 기존 네이버 Developers 키(`NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET`, `openapi.naver.com/v1/datalab/search`)는 HUB
   키가 없을 때만 씁니다. 레거시 경로는 **2027-06-30까지** 지원되는 것으로 알려져 있으니(공식 공지 미확인) 그 전에 HUB로
   옮기세요. 두 쌍이 다 있으면 HUB를 씁니다. `/api/config`의 `sources.naver.endpoint`에서 어느 쪽을 쓰는지 볼 수 있습니다.

**월 한도와 예산 가드**: 검색어트렌드는 키당 **월 50,000건**(쇼핑인사이트와 별도)이고, 한도에 닿으면 네이버가 자동으로
차단합니다. 앱은 한국 시간 기준 달마다 호출 수를 `budget/naver-YYYY-MM.json`에 세고, 호출 **전에** 1을 더합니다(실패한
호출도 셉니다).

| 이번 달 사용량 | 동작 |
|---|---|
| 80 % 미만 | 정상. IP당 시간당 새 조회 20건(`TRENDAI_NAVER_IP_PER_HOUR`) |
| 80 % 이상 | IP당 한도가 시간당 5건으로 줄어듭니다 (`/api/config`의 `budget_state: "low"`) |
| 90 % 이상 | 새 조회를 막고(`503 naver_budget_exhausted`) 캐시에 있는 결과만 보여 줍니다. 네이버 출처가 ‘한도 소진’으로 보입니다 |

비교 하나는 요청 한 번입니다(검색어 그룹 최대 5개를 한 요청에 묶음). 같은 조회가 동시에 들어오면 한 번만 호출하고,
캐시(최근 구간 포함 24시간, 그 밖에는 최대 21일)에 있으면 호출하지 않습니다. 네이버 화면 문구·표기 규칙과 AI 차단은
‘정식 공개 전 법무 확인 항목’을 보세요.

## Wikimedia 연락처

위키백과 조회수는 Wikimedia Pageviews API로 가져옵니다. Wikimedia의 User-Agent 정책은 연락처가 든 설명형 UA를 요구하므로
요청마다 `trendai-web/0.1.0 (+https://trendai.example; ops@example.com)` 형식의 UA를 보냅니다. 괄호 안은
`TRENDAI_PUBLIC_URL`과 `TRENDAI_CONTACT`에서 옵니다.

* **공개 운영**(접속 주소가 루프백(`127.0.0.0/8`, `::1`, `localhost`)이 아니거나 `TRENDAI_PUBLIC_URL`이 설정됨)에서는
  `TRENDAI_CONTACT`가 **필수**입니다. 없으면 `trendai serve`와 `trendai.web.wsgi`가 시작하지 않습니다(종료 코드 2).
* 사내망 등 비공개 배포라면 `TRENDAI_ALLOW_NO_CONTACT=1`로 이 요구를 끌 수 있습니다.
* 로컬(루프백) 실행에서는 경고만 출력합니다.

## AI 해석

결과 화면의 ‘AI로 다시 쓰기’는 규칙 기반 해석을 Claude가 다시 쓰게 합니다.

* 켜는 법: `pip install anthropic`(또는 `pip install -e ".[web]"`) 후 `ANTHROPIC_API_KEY`를 설정합니다. `TRENDAI_AI=auto`
  (기본)이면 둘 다 갖춰졌을 때 켜지고, `off`면 늘 꺼집니다.
* Claude에는 분석으로 계산한 사실(요약 수치·국면·예측)만 보냅니다. 업로드 원본은 서버에 남지 않으므로 보낼 수도 없습니다.
* **네이버 출처 결과는 AI로 보내지 않습니다**(버튼이 비활성화되고 이유를 보여 줌, API는 `403 ai_naver_blocked`). 공유
  링크에서도 AI 해석을 새로 만들 수 없습니다.
* 한도: IP당 처음 3건, 이후 시간당 10건(`TRENDAI_AI_IP_PER_HOUR`), 하루 200건(`TRENDAI_AI_DAILY_LIMIT`, 한국 시간
  기준), 동시에 2건 실행·4건 대기. 같은 결과·시계열·언어의 AI 해석은 한 번 만들면 저장해 두고 다시 씁니다.
* 시간 제한: Anthropic 클라이언트는 **90초** 타임아웃(재시도 1회)으로 만들고, 120초가 지나도 끝나지 않은 작업은
  `ai_failed`로 실패 처리합니다. 실패해도 규칙 기반 해석은 그대로 남습니다.

## 배포

### 단일 프로세스 규칙

작업 목록, 업로드 임시 캐시, 요청 한도는 **프로세스 메모리**에 있습니다. 그래서 반드시 프로세스 하나로 실행하세요.
여러 워커 프로세스로 나누면 진행 상황 조회·취소·한도가 엉킵니다.

```bash
# 내장 서버 (스레드 방식, 연결마다 스레드 하나, 느린 클라이언트는 30초 뒤 끊음)
trendai serve --host 127.0.0.1 --port 8000

# gunicorn: 워커 1개 + 스레드. --preload는 절대 쓰지 마세요(fork 뒤에는 작업 스레드가 살아남지 않음)
gunicorn -w 1 --threads 8 trendai.web.wsgi:application

# waitress: 업로드(10 MB + 여유)를 받을 수 있게 입력 버퍼를 키움
waitress-serve --threads=8 --inbuf-overflow=12582912 trendai.web.wsgi:application
```

`trendai.web.wsgi:application`은 지연 생성됩니다. 모듈을 import할 때는 아무것도 만들지 않고, 첫 요청에서 설정 확인
(`TRENDAI_CONTACT` 규칙 포함)·로그 설정·앱 생성을 한 번 합니다. gunicorn/waitress를 쓸 때도 설정은 같은 환경 변수로 합니다.

**정상 종료**: `trendai serve`는 SIGINT/SIGTERM을 받으면 먼저 ‘배수(drain)’ 상태가 됩니다. `/api/health`가 `503
draining`을 돌려주고, 새 분석·AI·업로드 요청은 `503 server_busy`(`Retry-After: 30`)로 거절하며 대기 중인 작업은
취소합니다. 진행 조회·결과·내보내기는 계속 응답합니다. 실행 중인 작업이 끝나거나 `TRENDAI_SHUTDOWN_GRACE`(30초)가
지나면 서버를 닫고 종료 코드 0으로 끝냅니다. 신호를 한 번 더 받으면 즉시 종료 코드 1로 끝납니다.

### 리버스 프록시와 TLS

공개 운영은 nginx 같은 리버스 프록시 뒤에서 TLS로 하는 것을 권장합니다(연결 수 제한, 느린 클라이언트 차단도 프록시가
맡습니다).

```nginx
server {
    listen 443 ssl;
    server_name trendai.example;
    # ssl_certificate … / ssl_certificate_key …

    client_max_body_size 11m;          # 업로드 10 MB + multipart 여유
    proxy_request_buffering off;       # 끄지 않으면 업로드 본문이 nginx 임시 파일로 디스크에 써집니다
                                       # (앱의 ‘원본 파일을 저장하지 않음’ 약속이 깨짐)
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
    }
}
```

앱 쪽에는 다음을 설정합니다.

```bash
TRENDAI_TRUST_PROXY=1                          # 클라이언트 IP와 자기 주소를 X-Forwarded-* 헤더에서 읽음
TRENDAI_PUBLIC_URL=https://trendai.example     # Origin 검사 기준 주소 (가장 확실한 방법)
TRENDAI_CONTACT=ops@example.com
TRENDAI_HSTS=1                                 # TLS로만 서비스할 때
```

`TRENDAI_TRUST_PROXY`는 프록시가 `X-Forwarded-For`를 직접 덮어쓰거나 덧붙일 때만 켜세요. 프록시 없이 켜면 클라이언트가
헤더를 꾸며 요청 한도를 피할 수 있습니다.

### Docker

```bash
docker build -t trendai .
docker run -d --name trendai -p 8000:8000 \
  -e TRENDAI_CONTACT=ops@example.com \
  -e TRENDAI_PUBLIC_URL=https://trendai.example \
  -e NAVER_API_HUB_KEY_ID=… -e NAVER_API_HUB_KEY=… \
  -v trendai-data:/data \
  trendai
```

이미지는 `python:3.12-slim` 기반이고 비루트 사용자(`trendai`, uid 10001)로 실행하며 `openpyxl`과 `anthropic`을 함께
설치합니다. `0.0.0.0`에 바인드하므로 `TRENDAI_CONTACT`(또는 `TRENDAI_ALLOW_NO_CONTACT=1`) 없이는 시작하지 않습니다.
데이터는 `/data` 볼륨에 쌓이고, `HEALTHCHECK`가 30초마다 컨테이너 안의 `http://127.0.0.1:$TRENDAI_PORT/api/health`를
확인합니다(`200`이면 정상, 저장소에 쓸 수 없거나 종료 중이면 `503`이라 비정상). `docker stop`의 SIGTERM으로 위의
정상 종료가 일어나므로 `--stop-timeout`은 `TRENDAI_SHUTDOWN_GRACE`보다 길게(기본 30초면 40초 정도) 두세요.

* **포트 바꾸기**: `-e TRENDAI_PORT=9000 -p 9000:9000`처럼 환경 변수로 바꾸세요. 상태 확인도 같은 포트를 봅니다.
  `docker run … trendai trendai serve --port 9000`처럼 명령으로 바꾸면 상태 확인은 여전히 `TRENDAI_PORT`(기본 8000)를
  보므로 컨테이너가 계속 ‘unhealthy’로 표시됩니다. 컨테이너에서는 `TRENDAI_PORT=0`(빈 포트)을 쓰지 마세요.
* **프록시**: 바깥으로 나가는 요청(네이버·위키백과·AI)을 위해 `HTTP_PROXY`/`HTTPS_PROXY`를 넣어도 상태 확인은 프록시를
  거치지 않고 `127.0.0.1`로 바로 접속하므로, 상태 확인 때문에 `NO_PROXY`를 따로 둘 필요는 없습니다.

## 처리량

분석 엔진은 GIL 아래에서 도는 CPU 작업이라, 한 프로세스는 CPU 코어를 사실상 하나만 씁니다. 아래 값은 4 vCPU(Intel Xeon
2.1 GHz), Python 3.13에서 네트워크 없이 가짜 데이터 소스로 잰 것입니다. 실제 운영에서는 네이버·위키백과 응답 시간이
더해지고, CPU가 느리면 그만큼 줄어듭니다.

| 분석 | 한 건(혼자 돌 때) | 한 프로세스 처리량 |
|---|---|---|
| 단일 시계열 (월간·주간 5년, 일간 1–3년) | 0.2–1.3초 | — |
| 5개 비교, 월간 위키백과 5년 | 약 1.2초 | — |
| 5개 비교, 주간 5년 (네이버 / 위키백과) | 약 2.9초 / 4.7초 | **분당 약 15건 / 12건** |
| 업로드 10열, 월간 10년 | 약 4.7초 | — |

* 처리량은 10–20건을 한꺼번에 넣어 모두 끝날 때까지 잰 값입니다. 이렇게 몰리면 한 건이 끝나기까지 대기 시간을 포함해
  30–50초가 걸립니다(화면에는 진행 상황이 보임).
* `TRENDAI_WORKERS`를 늘려도 처리량은 거의 늘지 않습니다. 긴 작업 뒤에 짧은 작업이 막히지 않게 해 줄 뿐입니다.
* 같은 조회가 동시에 들어오면 한 작업으로 합쳐지고(진행 중인 작업에 합류), 끝난 결과는 캐시에서 바로 돌려줍니다.
  같은 네이버 요청은 한 번만 호출합니다.
* 실행 2건 + 대기 8건(`TRENDAI_MAX_QUEUED`)을 넘으면 `503 server_busy`(`Retry-After: 10`)로 거절합니다. 업로드 파싱은
  동시에 2건까지이고 넘치면 `503 server_busy`(`Retry-After: 5`)입니다.

## 요청 한도와 NAT

한도는 클라이언트 IP마다 토큰 버킷으로 셉니다(`burst` = 처음에 쓸 수 있는 수, 이후 시간당 `refill`만큼 회복). IPv6는
/64 단위로 묶습니다. 넘으면 `429`와 `Retry-After`를 돌려줍니다.

| 버킷 | 대상 | burst | 시간당 |
|---|---|---|---|
| `analyze` | `POST /api/analyze` | 10 | 60 |
| `upload` | `POST /api/upload` | 5 | 30 |
| `naver` | 네이버 새 조회(캐시 적중 제외) | 5 | 20 (예산 80 % 이상이면 5) |
| `ai` | `POST /api/narrative` | 3 | 10 |
| `share` | 공유 만들기·삭제 | 5 | 20 |
| `export` | 내보내기 | 30 | 300 |
| `poll` | 작업 진행 조회 (IP와 작업 ID 조합마다) | 60 | 21,600 |
| `read` | 결과·공유·설정·샘플 조회 | 60 | 1,200 |

IP당 한도는 **같은 주소 뒤의 사용자끼리 나눠 씁니다**. 통신사 CGNAT나 사무실 NAT 뒤의 사용자는 한 IP로 보이고, 리버스
프록시 뒤에서 `TRENDAI_TRUST_PROXY`를 켜지 않으면 **모든 사용자가 프록시의 IP 하나**로 보여 금세 한도에 걸립니다.

## 데이터 보관

```
$TRENDAI_DATA_DIR/            폴더 0700, 파일 0600
  results/<id>/               분석 결과 (화면 데이터 ko/en, JSON 내보내기, HTML 리포트, AI 해석)   24시간
  shares/<share_id>/          공유 링크 스냅숏 (result_id는 담지 않음)                           30일
  cache/naver/                네이버 응답 캐시                                       24시간, 최대 21일
  cache/wiki/                 위키백과 응답 캐시                              월간 24시간, 주간·일간 6시간
  cache/results/              같은 조회의 결과 재사용 색인
  budget/                     네이버 월 사용량, AI 일 사용량 (400일 뒤 삭제)
```

* **업로드 원본은 디스크에 쓰지 않습니다.** 파싱한 표는 메모리 캐시에 30분(최대 64개·256 MB)만 두고, 분석이 끝난
  결과만 저장합니다. 앞단 프록시가 본문을 디스크에 버퍼링하지 않게 하세요(위 nginx 설정).
* 정리 스레드가 10분마다 만료된 결과·공유·캐시, 1시간 넘게 읽을 수 없는 파일, 400일 지난 예산 파일을 지웁니다. 결과와
  캐시가 자기 몫(`TRENDAI_STORE_MAX_MB`에서 공유 링크가 쓰는 만큼을 뺀 것, 공유 링크는 최대 절반까지만 쓸 수 있고 넘으면 새
  공유를 거절)을 넘으면 오래된 결과부터 그 몫의 90 % 아래로 줄입니다(공유와 예산 파일은 용량 때문에 지우지 않음). 이때 로그에
  `store above … MB: evicted N oldest results`가 남습니다. 이 줄이 자주 보이면 결과가 `TRENDAI_RESULT_TTL_H`보다 먼저
  사라지고 있다는 뜻이니 한도를 늘리세요.
* **백업**: 의미 있는 것은 `shares/`뿐입니다. 결과는 24시간 뒤 사라지고 캐시는 다시 만들 수 있습니다. 네이버 캐시는
  약관상 21일 넘게 보관하면 안 되므로 백업에서 빼세요.
* **업그레이드**: 화면 데이터 형식 번호(`trendai/web/viewmodel.py`의 `WEB_VM_SCHEMA`)가 바뀌는 버전으로 올리면 기존
  결과와 **공유 링크가 모두 무효**가 됩니다(없는 것으로 보고 지움). 릴리스 노트에서 확인하세요.

### 용량 산정

결과 하나의 크기(화면 데이터 ko/en, JSON 내보내기 ko/en, HTML 리포트를 합친 디스크 크기)는 시계열 수와 데이터 길이에
비례합니다. 이 버전에서 잰 값은 다음과 같습니다.

| 결과 | 디스크 크기 |
|---|---|
| 네이버 월간 5년, 시계열 1개 | 약 0.07 MB |
| 네이버 주간 5년, 시계열 1개 | 약 0.15 MB |
| 위키백과 월간 5년 · 샘플, 시계열 1개 | 약 0.2–0.3 MB |
| 위키백과 주간 5년, 시계열 1개 | 약 0.5 MB |
| 위키백과·업로드 일간 1–5년, 시계열 1개 | 약 0.5–1.2 MB |
| 업로드 월간 10년, 값 열 1개마다 | 약 0.27 MB (올린 파일 크기와 거의 무관: 3 KB짜리 CSV도 열 3개면 약 0.8 MB) |
| 비교 분석 | 각 시계열 크기의 합 (위키백과 주간 5개 약 2.5 MB, 월간 5개 약 1.1 MB, 네이버 주간 5개 약 0.75 MB) |

필요한 한도는 대략 이렇게 잡습니다.

```
TRENDAI_STORE_MAX_MB ≈ (하루 분석 수 × 결과 평균 크기(MB) × TRENDAI_RESULT_TTL_H / 24 + 캐시) ÷ 0.9 + 공유 링크 용량
```

* 기본값 2,048 MB(공유가 적을 때 결과 몫 약 1.8 GB)는 결과 평균 0.3 MB 기준 하루 약 6,000건, 위키백과 주간 5개 비교만
  있다면 하루 약 700건까지 24시간 보관합니다.
* 한 프로세스가 쉬지 않고 주간 5개 비교를 처리하면(분당 약 12건 × 2.5 MB) 2 GB는 **약 1시간**에 찹니다. 그런 부하를
  예상하면 하루치(약 50 GB)에 맞춰 늘리거나 `TRENDAI_RESULT_TTL_H`를 줄이세요.
* 실제 크기는 `du -sh $TRENDAI_DATA_DIR/results/*`로 확인할 수 있습니다.

## 보안 메모

* **헤더**: 모든 응답에 `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; …;
  frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
  `Permissions-Policy`, `Cross-Origin-Opener-Policy`/`Cross-Origin-Resource-Policy: same-origin`을 붙입니다. 앱에는
  인라인 스크립트·스타일이 없습니다. HTML 리포트 내보내기는 `sandbox` CSP로 내려갑니다. CORS는 없습니다.
* **요청 검사**: POST/DELETE는 `Origin`이 자기 주소와 다르면 `403 cross_origin`입니다. 본문 크기는 읽기 전에
  확인하고(JSON 64 KB, 업로드 10 MB), JSON 형식을 요구합니다. 압축 폭탄(압축 해제 30 MB 또는 항목 1,000개 초과)과
  지나치게 큰 표(셀 200만 개, 열 5,000개 초과 등)는 거절합니다.
* **정적 파일**은 시작할 때 만든 목록에 있는 경로만 제공하므로 경로 조작(`..`)이 통하지 않습니다. `/static/dev/`는
  개발 모드에서만 열립니다.
* **로그 정책**: 접속 로그는 `rid=… ip=<12자리 해시> GET /api/results/{id} 200 12ms 3456B` 형식입니다. IP는 하루마다
  바뀌는 소금으로 해시하고, 경로 대신 라우트 이름을 남깁니다. **검색어, 쿼리 문자열, 업로드 파일 이름과 내용, 요청·응답
  본문, 헤더 값, 전체 ID와 토큰, API 키는 남기지 않습니다.** 예외는 종류와 파일·줄 위치만 남깁니다(입력 내용을 인용할 수
  있는 예외 메시지는 남기지 않음).
* **키 관리**: 키는 환경 변수(또는 컨테이너 비밀값)로만 넣고 저장소에 커밋하지 마세요. 키는 설정 객체·로그·응답·URL에
  들어가지 않습니다. 공유 링크 삭제 토큰은 SHA-256 해시로만 저장합니다.

## 정식 공개 전 법무 확인 항목

현재 구현은 아래 항목을 **보수적으로 가정**합니다. 공개 전에 법무 검토로 확인하세요.

1. **2026-09 개정 네이버 API 약관(검색 API 특약)이 데이터랩 검색어트렌드에도 적용되는지.** 적용된다고 가정하고
   다음을 지킵니다: 네이버 데이터를 AI(Claude)로 보내지 않음, 서버 캐시 21일 이하, 결과 페이지 광고 없음, 원본 값 재배포
   없음.
2. **네이버 비율 값의 표시와 내려받기 구분.** 화면(차트, 읽기 값, 표 보기)과 그 화면을 그리는 API 응답에 비율 값을
   *표시*하는 것은 허용되고, 파일 내려받기(CSV·엑셀·JSON·값 표)와 복사 기능에는 원본 비율을 넣지 않는다는 판단입니다.
   네이버 결과의 HTML 리포트 내보내기는 막혀 있습니다. 추세·예측·범위처럼 계산한 값은 내보냅니다. 계산값 내보내기와 차트
   PNG가 ‘제3자 제공’에 해당하지 않는지 확인이 필요합니다.
3. **네이버 결과에 ‘기준 시점=100’ 변환을 막은 것이 충분한지.** 결과의 ‘수정·재배열’ 금지 조항을 고려해 네이버와 Google
   트렌드 비교에서는 이 축을 제공하지 않습니다.
4. **출처 표기.** 네이버는 로고·브랜드 색 없이 텍스트로 `출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB)`를, Google
   트렌드 업로드는 Google 인용 문구를, 위키백과는 `Wikimedia Pageviews`를 화면·내보내기·PNG에 적습니다. NAVER가 지정
   표기를 요구하는지 확인하세요.
5. **Google 트렌드 업로드**: 사용자가 직접 내려받은 파일이므로 내보내기를 허용하고 출처 문구를 붙입니다. 이 판단이 맞는지.
6. **Wikimedia**: User-Agent 정책의 현재 문구와 요청 한도, 조회수 데이터의 CC0 여부(출처는 늘 적음).
7. **AI 표시 의무**: AI가 쓴 해석에는 `AI 해석 · Claude`를 붙입니다. 인공지능 기본법(2026-01-22 시행)의 생성형 AI 고지
   의무가 이 규모의 서비스에 적용되는지.
8. **업로드와 개인정보**: 원본을 저장하지 않고 결과는 기본 비공개(24시간), 공유는 추측할 수 없는 링크(30일, 삭제 가능)
   입니다. `/privacy` 페이지 문구와 개인정보보호법 대응을 확인하세요.
9. **면책**: ‘투자·사업 판단의 유일한 근거로 쓰지 마세요’ 문구가 충분한지.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 위키백과·네이버 요청이 TLS 인증서 오류로 실패 (회사 프록시 뒤) | 앱은 시스템 CA와 `HTTPS_PROXY`를 씁니다. 프록시의 CA 묶음을 `SSL_CERT_FILE=/path/ca-bundle.crt`로 지정하세요 |
| `오류: 127.0.0.1:8000을 열 수 없습니다 — 이미 다른 프로그램이 이 포트를 쓰고 있습니다` | 다른 포트(`--port 8001`, 또는 `--port 0`으로 빈 포트)를 쓰거나 기존 프로세스를 끄세요 |
| `오류: 공개 주소로 실행하려면 TRENDAI_CONTACT…` | `TRENDAI_CONTACT`를 설정하거나, 비공개 배포라면 `TRENDAI_ALLOW_NO_CONTACT=1` |
| 네이버 결과가 ‘네이버 연결에 문제가 생겼어요’ (`naver_auth`, 로그에 HTTP 401) | 키 ID·키가 맞는지, HUB 애플리케이션에서 **검색어트렌드 API를 켰는지** 확인하세요. 로그의 `log_detail`에 네이버 오류 코드가 있습니다 |
| 네이버 출처가 ‘한도 소진’ | 이번 달 사용량이 90 %를 넘었습니다. `budget/naver-YYYY-MM.json`을 보고 다음 달(한국 시간)을 기다리거나 한도를 조정하세요 |
| 분석·업로드 POST가 `403 cross_origin` | 브라우저가 보는 주소와 앱이 아는 자기 주소가 다릅니다. `TRENDAI_PUBLIC_URL`을 실제 주소로 설정하고, 프록시라면 `TRENDAI_TRUST_PROXY=1`과 `Host`/`X-Forwarded-*` 헤더를 확인하세요 |
| 사용자들이 자꾸 `429` | 프록시 뒤에서 `TRENDAI_TRUST_PROXY`가 꺼져 있어 모두 한 IP로 보이는지 확인하세요 |
| `/`가 `503 trendai: static files are missing` | 정적 파일이 설치되지 않았습니다. `pip install .`로 다시 설치하세요(패키지 데이터 `trendai/web/static/**`) |
| Docker 컨테이너가 계속 `unhealthy` | 포트를 `--port`로 바꿨다면 `TRENDAI_PORT`로 바꾸세요(상태 확인은 `TRENDAI_PORT`를 봄). `/api/health`가 `503`이면 데이터 볼륨이 가득 찼거나 쓸 수 없는지(`store.writable`) 확인하세요 |
| 결과 링크가 24시간 전에 ‘찾을 수 없어요’ | 로그에 `store above … MB: evicted …`가 있으면 저장소 한도에 걸린 것입니다. ‘용량 산정’대로 `TRENDAI_STORE_MAX_MB`를 늘리세요 |
| AI 버튼이 늘 비활성 | `anthropic` 설치와 `ANTHROPIC_API_KEY`, `TRENDAI_AI`를 확인하세요. `/api/config`의 `features.ai`에서 상태를 볼 수 있습니다 |

## 테스트 실행

```bash
python -m pytest                       # 전부 (node와 playwright가 있으면 E2E 포함, 마커 e2e)
python -m pytest -m "not e2e"          # 파이썬 테스트만
python -m pytest tests/test_web_*.py   # 웹 백엔드
node --test tests/js                   # 프런트엔드 단위 테스트 (Node 22)
NODE_PATH="$(npm root -g)" node --test --test-concurrency=1 tests/e2e   # 브라우저 E2E를 직접
```

* E2E는 Node의 `node:test`와 **전역으로 설치한** Playwright(`npm install -g playwright`, Chromium 포함)를 씁니다. 전역
  모듈은 `NODE_PATH`로만 찾을 수 있어서 위처럼 `NODE_PATH="$(npm root -g)"`를 줍니다. pytest 래퍼
  (`tests/e2e/test_e2e.py`)는 필요하면 `NODE_PATH`를 스스로 채우고, `node`나 `playwright`가 없으면 건너뜁니다.
* E2E 서버(`tests/e2e/server.py`)는 가짜 네이버·위키백과·AI를 쓰므로 네트워크에 접속하지 않습니다. 파이썬은
  `TRENDAI_E2E_PYTHON`(기본 `python3`)으로 고를 수 있습니다.
* 화면만 따로 고칠 때는 `trendai serve --dev`로 띄운 뒤 `/static/dev/mock.html`을 열면 서버 데이터 없이 고정된 응답으로
  화면을 볼 수 있습니다.
