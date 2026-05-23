"""
StaticFiles with revalidate-every-request cache headers (Phase 14).

The default ``fastapi.staticfiles.StaticFiles`` sends ``ETag`` and
``Last-Modified`` but no ``Cache-Control``, so browsers cache app.js /
index.html heuristically. The symptom: a code change appears in the
file on disk and in the server's response — but the browser keeps
showing the old version until you hard-refresh (Cmd+Shift+R).

This subclass appends ``Cache-Control: no-cache, must-revalidate`` so
the browser sends an ``If-None-Match`` on every request. ETag still
produces a 304 when the file is unchanged, so the cost is just a small
round-trip — trivial on localhost.
"""
from __future__ import annotations

from fastapi.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response
