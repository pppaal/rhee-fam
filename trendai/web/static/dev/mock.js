// Dev-only fetch shim (spec B.4): lets the SPA run against the JSON fixtures in static/dev/fixtures/ without the
// backend. Load dev/mock.html (served by `TRENDAI_DEV=1 trendai serve`, or by any static server whose /static/ maps to
// trendai/web/static/). The page moves itself to "/" with history.replaceState so the router sees an app route.
// Owner: WP3-app. Never shipped (static/dev/** is excluded from the wheel) and never served without dev mode.
//
// Fixtures come from dev/make_fixtures.py (real backend output with fake fetchers). Analyses are matched to a
// fixture by source and number of series; job polling is simulated in a few steps so progress, progressive
// rendering and cancel can be exercised. Add `?mockdelay=ms` to change the simulated poll delay (default 700).
(function () {
  'use strict';
  var realFetch = window.fetch.bind(window);
  var FIX = '/static/dev/fixtures/';
  var jobs = {};
  var counter = 0;
  var delay = Number(new URLSearchParams(window.location.search).get('mockdelay')) || 700;

  if (/\/dev\/mock\.html$/.test(window.location.pathname)) {
    var search = window.location.search.replace(/([?&])mockdelay=\d+&?/, '$1').replace(/[?&]$/, '');
    window.history.replaceState(null, '', '/' + search + window.location.hash);
  }

  function json(status, body, headers) {
    return new Response(status === 204 ? null : JSON.stringify(body), {
      status: status,
      headers: Object.assign({ 'Content-Type': 'application/json; charset=utf-8' }, headers || {}),
    });
  }

  function fixture(name) {
    return realFetch(FIX + name + '.json').then(function (r) {
      if (!r.ok) throw new Error('mock: fixture ' + name + ' missing');
      return r.json();
    });
  }

  function hlOf(url) {
    return url.searchParams.get('hl') === 'en' ? 'en' : 'ko';
  }

  function pickResult(body) {
    var n = (body.series || []).length;
    if (body.source === 'upload') return 'mockUploadResult00001';
    if (body.source === 'sample') return n > 1 ? 'mockSampleCompare00001' : 'mockSampleDetail000001';
    if (body.source === 'wiki') return n > 1 ? 'mockWikiCompareFail001' : 'mockWikiDetail00000001';
    return n > 1 ? 'mockNaverCompare000001' : 'mockNaverDetail0000001';
  }

  function errorEnvelope(code, status, ko, en) {
    return { error: { code: code, status: status, message_ko: ko, message_en: en, params: {}, fixes: [], retry_after: null, request_id: 'mock0000' } };
  }

  /** A C.6 analyze snapshot for poll number `step` of a job whose final envelope is `env`. */
  function snapshot(job, env, since) {
    var total = env.series.length + env.failed.length;
    var step = job.polls;
    var done = Math.max(0, Math.min(total, step - 2));
    var stage = step < 1 ? 'queued' : step < 2 ? 'data' : done < total ? 'series' : step < total + 3 ? 'finish' : 'done';
    var finished = env.series.filter(function (vm) { return vm.index < done && vm.index >= since; });
    var failed = env.failed.filter(function (f) { return f.index < done; });
    var current = stage === 'series' ? { index: done, name: (env.series.concat(env.failed)).filter(function (x) { return x.index === done; })[0].name, step: step % 2 ? 'analyze' : 'forecast' } : null;
    return {
      job_id: job.id, kind: 'analyze', status: stage === 'done' ? 'done' : stage === 'queued' ? 'queued' : 'running',
      stage: stage, series_total: total, series_done: done, current: current,
      series: finished.map(function (vm) { return { index: vm.index, status: 'done', vm: vm }; }),
      failed: failed, notices: [], compare: stage === 'done' ? env.compare : null, view: env.view, error: null,
      result_url: stage === 'done' ? '/api/results/' + job.result : null, elapsed_ms: step * delay,
      poll_after_ms: stage === 'done' ? null : delay,
    };
  }

  function route(method, url, init) {
    var path = url.pathname;
    var m;
    if (method === 'GET' && path === '/api/config') return fixture('config').then(function (b) { return json(200, b); });
    if (method === 'GET' && path === '/api/samples') return fixture('samples-' + hlOf(url)).then(function (b) { return json(200, b); });
    if (method === 'POST' && path === '/api/upload') return fixture('upload-sales').then(function (b) { return json(200, b); });
    if (method === 'POST' && path === '/api/analyze') {
      var body = JSON.parse((init && init.body) || '{}');
      var id = 'mockjob' + String(++counter).padStart(15, '0');
      jobs[id] = { id: id, result: pickResult(body), polls: 0, cancelled: false };
      return Promise.resolve(json(202, { job_id: id, status: 'queued', cached: false, joined: false, queue_position: 0, poll_url: '/api/jobs/' + id, result_url: '/api/results/' + id }));
    }
    if ((m = /^\/api\/jobs\/([A-Za-z0-9_-]+)$/.exec(path))) {
      var job = jobs[m[1]];
      if (method === 'DELETE') {
        if (job) job.cancelled = true;
        return Promise.resolve(json(202, { job_id: m[1], status: 'cancelling' }));
      }
      if (!job) {
        // a /r/<id> page of a fixture result: report it as done
        return fixture('result-' + m[1] + '-' + hlOf(url)).then(function () {
          return json(200, { job_id: m[1], kind: 'analyze', status: 'done', stage: 'done', series_total: 1, series_done: 1, current: null, series: [], failed: [], notices: [], compare: null, view: 'detail', error: null, result_url: '/api/results/' + m[1], elapsed_ms: 0, poll_after_ms: null });
        }, function () {
          return json(404, errorEnvelope('job_not_found', 404, '분석 작업을 찾을 수 없어요. 다시 분석해 주세요.', 'The analysis job was not found. Please run it again.'));
        });
      }
      if (job.kind === 'narrative') {
        job.polls += 1;
        var ready = job.polls > 2;
        return Promise.resolve(json(200, { job_id: job.id, kind: 'narrative', status: ready ? 'done' : 'running', error: null, elapsed_ms: job.polls * delay, poll_after_ms: ready ? null : delay,
          narrative: ready ? { markdown: '**AI 테스트 해석** — 목업\n\n## 과거\n- 테스트 문장입니다.', caveats_markdown: '- 테스트 주의입니다.', source: 'llm', badge: 'AI 해석 · Claude', model: 'mock', note: 'AI가 작성한 해석이에요. 수치는 위 분석 결과를 근거로 해요.' } : null }));
      }
      if (job.cancelled) {
        return Promise.resolve(json(200, { job_id: job.id, kind: 'analyze', status: 'cancelled', stage: 'data', series_total: 1, series_done: 0, current: null, series: [], failed: [], notices: [], compare: null, view: 'detail', error: null, result_url: null, elapsed_ms: 0, poll_after_ms: null }));
      }
      job.polls += 1;
      var since = Number(url.searchParams.get('since')) || 0;
      return fixture('result-' + job.result + '-' + hlOf(url)).then(function (env) { return json(200, snapshot(job, env, since)); });
    }
    if (method === 'GET' && (m = /^\/api\/results\/([A-Za-z0-9_-]+)$/.exec(path))) {
      var name = jobs[m[1]] ? jobs[m[1]].result : m[1];
      return fixture('result-' + name + '-' + hlOf(url)).then(function (b) { return json(200, b); }, function () { return fixture('error-result-not-found').then(function (e) { return json(404, e); }); });
    }
    if (method === 'GET' && /^\/api\/shared\//.test(path)) return fixture('shared-' + hlOf(url)).then(function (b) { return json(200, b); });
    if (method === 'POST' && path === '/api/share') return fixture('share-created').then(function (b) { return json(201, b); });
    if (method === 'DELETE' && /^\/api\/share\//.test(path)) return Promise.resolve(json(204, null));
    if (method === 'POST' && path === '/api/narrative') {
      var nid = 'mocknarr' + String(++counter).padStart(14, '0');
      jobs[nid] = { id: nid, kind: 'narrative', polls: 0 };
      return Promise.resolve(json(202, { job_id: nid, status: 'queued', poll_url: '/api/jobs/' + nid }));
    }
    return Promise.resolve(json(404, errorEnvelope('not_found', 404, '페이지나 데이터를 찾을 수 없어요.', 'Not found.')));
  }

  window.fetch = function (input, init) {
    var url = new URL(typeof input === 'string' ? input : input.url, window.location.href);
    if (url.origin !== window.location.origin || url.pathname.indexOf('/api/') !== 0) return realFetch(input, init);
    var method = ((init && init.method) || 'GET').toUpperCase();
    return new Promise(function (resolve) { setTimeout(resolve, 60); }).then(function () { return route(method, url, init); });
  };
  // eslint-disable-next-line no-console
  console.info('trendai dev mock: /api/* is served from static/dev/fixtures/');
}());
