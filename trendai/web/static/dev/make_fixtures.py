"""Regenerate the dev fixtures in ``static/dev/fixtures/`` from the real backend with its fake fetchers.

Dev only (``static/dev/**`` is never served in production nor shipped in the wheel; ``.py`` files are not indexed by
the static file server). Run from the repository root::

    python trendai/web/static/dev/make_fixtures.py

Every file is a real API response (spec C/D shapes) produced by ``trendai.web.testing.make_test_app`` with
``FakeNaver`` / ``FakeWiki`` / ``fake_llm`` (H.4.2), so the fixtures stay valid against ``vm.schema.json``
(``tests/js/schema.test.mjs``). ``dev/mock.js`` serves them through a ``fetch`` shim (``dev/mock.html``).
Result ids are replaced by stable 22-character ids so mock URLs stay readable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from trendai.web.testing import TestClient, make_test_app  # noqa: E402

OUT = pathlib.Path(__file__).with_name("fixtures")

# fixture id (22 chars, the C.0 id pattern) -> analyze request
QUERIES = {
    "mockSampleDetail000001": {"source": "sample", "series": [{"terms": ["growing"]}]},
    "mockSampleCompare00001": {"source": "sample",
                               "series": [{"terms": ["emerging"]}, {"terms": ["fad"]}, {"terms": ["seasonal"]}]},
    "mockWikiDetail00000001": {"source": "wiki", "series": [{"terms": ["인공지능"]}], "unit": "month"},
    "mockWikiCompareFail001": {"source": "wiki", "series": [{"terms": ["인공지능"]}, {"terms": ["없는문서"]}]},
    "mockNaverCompare000001": {"source": "naver", "series": [{"terms": ["선크림", "자외선차단제"]},
                                                             {"terms": ["패딩"]}, {"terms": ["저검색어"]}],
                               "dev": "mo", "gender": "f", "ages": ["4", "5"]},
    "mockNaverDetail0000001": {"source": "naver", "series": [{"terms": ["선크림"]}]},
}

SALES_CSV = "날짜,매출,방문수,번호,메모\n" + "".join(
    f"{2021 + (i // 12)}-{i % 12 + 1:02d},{1200000 + i * 18000 + (i % 12) * 52000},{300 + i * 3 + (i % 12) * 9},{i + 1},"
    f"{'행사' if i % 12 == 6 else ''}\n" for i in range(60))


def dump(name: str, data) -> None:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def swap_ids(obj, real: str, fake: str):
    return json.loads(json.dumps(obj, ensure_ascii=False).replace(real, fake))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        client = TestClient(make_test_app(tmp, naver=True, ai=True))
        dump("config", client.get("/api/config").json())
        for hl in ("ko", "en"):
            dump(f"samples-{hl}", client.get(f"/api/samples?hl={hl}").json())

        for fid, body in QUERIES.items():
            started = client.analyze(body)
            assert started.status in (200, 202), started.json()
            real = started.json()["job_id"]
            snap = client.wait_job(real)
            assert snap["status"] == "done", snap
            for hl in ("ko", "en"):
                env = client.get(f"/api/results/{real}?hl={hl}").json()
                dump(f"result-{fid}-{hl}", swap_ids(env, real, fid))
            if fid == "mockSampleDetail000001":     # one terminal analyze snapshot (C.6) for the schema test
                dump(f"job-{fid}-done", swap_ids(client.get(f"/api/jobs/{real}?since=0").json(), real, fid))

        # an error envelope (job-level failure) and an upload round trip
        failed = client.analyze({"source": "naver", "series": [{"terms": ["에러"]}]})
        snap = client.wait_job(failed.json()["job_id"])
        dump("job-failed-naver-server", swap_ids(snap, failed.json()["job_id"], "mockNaverFailed0000001"))

        up = client.upload("sales.csv", SALES_CSV.encode("utf-8"))
        assert up.status == 200, up.json()
        upload = up.json()
        real_upload = upload["upload_id"]
        dump("upload-sales", swap_ids(upload, real_upload, "mockUploadSales0000001"))
        started = client.analyze({"source": "upload", "upload_id": real_upload, "date_col": None,
                                  "value_cols": ["매출"], "freq": "auto", "h": "auto", "unit_label": "원"})
        real = started.json()["job_id"]
        client.wait_job(real)
        for hl in ("ko", "en"):
            env = client.get(f"/api/results/{real}?hl={hl}").json()
            dump(f"result-mockUploadResult00001-{hl}", swap_ids(env, real, "mockUploadResult00001"))
        share = client.post("/api/share", json={"result_id": real}).json()
        dump("share-created", swap_ids(share, real, "mockUploadResult00001"))
        for hl in ("ko", "en"):
            shared = client.get(f"/api/shared/{share['share_id']}?hl={hl}").json()
            dump(f"shared-{hl}", shared)

        missing = client.get("/api/results/AAAAAAAAAAAAAAAAAAAAAA")
        dump("error-result-not-found", missing.json())
    print(f"wrote {len(list(OUT.glob('*.json')))} fixtures to {OUT}")


if __name__ == "__main__":
    main()
