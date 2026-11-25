FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TRENDAI_HOST=0.0.0.0 TRENDAI_PORT=8000 TRENDAI_DATA_DIR=/data
# binds 0.0.0.0, so the container refuses to start without -e TRENDAI_CONTACT=... (or TRENDAI_ALLOW_NO_CONTACT=1)
WORKDIR /app
COPY pyproject.toml README.md ./
COPY trendai ./trendai
RUN pip install --no-cache-dir . openpyxl anthropic \
 && useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin trendai \
 && mkdir -p /data && chown trendai /data
USER trendai
VOLUME ["/data"]
EXPOSE 8000
# probes the port the server really listens on (TRENDAI_PORT) and never goes through an HTTP(S)_PROXY set for the app
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:%s/api/health' % (os.environ.get('TRENDAI_PORT') or '8000'), timeout=3)"
CMD ["trendai", "serve"]
