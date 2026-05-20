"""
Phase 11.5 — deployment scheduler.

Turns the deploy pipeline (runner + submitter) into something that runs
itself. Each armed deployment gets one recurring APScheduler cron job;
when the job fires it executes the full pipeline for that deployment:

    fetch price history + T212 portfolio
        -> runner.compute_intents      (target vs current -> order deltas)
        -> orders.submit_intents       (idempotent persist + place)
        -> repo.update_deployment_last_run

Design notes
------------

Scheduler kind. `BackgroundScheduler` (not `AsyncIOScheduler`): every step
of the pipeline is synchronous, blocking I/O — yfinance, sqlite, httpx's
sync client. BackgroundScheduler runs each job on a worker thread, so the
blocking is fine and the FastAPI event loop is never touched.

One job per deployment. The job id is `deployment:<id>`.
`register_deployment` uses `replace_existing=True`, so re-registering
(after an edit, or a second `start()`) updates the job in place rather
than duplicating it.

In-memory jobstore. APScheduler does NOT persist the jobs. The source of
truth is the `deployments` table; `register_all_armed()` rebuilds every
job from SQLite on startup. This is deliberate — a deployment's schedule
can never drift out of sync with the database.

`as_of` is the last bar, not "now". The runner's idempotency hash is keyed
on `as_of_timestamp`. We pass the timestamp of the LAST BAR in the fetched
data, not the wall clock. Two consequences, both wanted:
  * Firing the job twice for the same trading day yields the same hash
    -> the duplicate order is skipped, no double-trade.
  * A new trading day's bar yields a fresh hash -> the order goes through.

Misfire = skip. APScheduler's default `misfire_grace_time` is ~1s. If the
machine is asleep at the scheduled minute, the run is SKIPPED, not run
late. For trading that is the safe default — a stale order placed hours
after the signal is worse than no order at all. Phase 11.6's
manual-trigger route covers the "I missed it, run it now" case.

Armed is the gate, checked twice. Only armed deployments get a job
(`register_all_armed`), and `run_deployment` re-checks `armed` when it
fires. The second check is defence in depth: a disarmed deployment whose
job somehow lingered will still not place orders.

Testing. `run_deployment` is the unit under test — call it directly with
monkeypatched price / portfolio / T212 seams; it never raises (a
top-level job boundary catches everything into a DeploymentRunResult).
The registration helpers are tested against a non-started scheduler,
which APScheduler fully supports (jobs sit "pending" and stay
inspectable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.data.prices import get_prices
from backend.db import repo
from backend.db.conn import connect
from backend.deploy.orders import SubmitResult, submit_intents
from backend.deploy.runner import OrderIntent, compute_intents
from backend.t212 import client as t212_client
from backend.t212 import instruments

logger = logging.getLogger(__name__)

# All cron schedules are interpreted in this timezone. UTC keeps a
# deployment's "21:05" meaning the same thing regardless of the host
# machine's locale or daylight-saving state.
SCHEDULER_TIMEZONE = "UTC"

# How far back to fetch price history per timeframe, in calendar days.
# Must comfortably exceed the longest indicator warm-up a strategy needs
# (an SMA(200) on daily data needs ~290 calendar days). yfinance also
# caps how much intraday history it serves, so those windows stay modest.
_LOOKBACK_DAYS: dict[str, int] = {"1d": 800, "1h": 60, "5m": 30}
_DEFAULT_LOOKBACK_DAYS = 800

# APScheduler job-id prefix — one job per deployment.
_JOB_PREFIX = "deployment:"


# --- result type ----------------------------------------------------------


@dataclass(frozen=True)
class DeploymentRunResult:
    """Outcome of one `run_deployment` call.

    `run_deployment` never raises; every path returns one of these so
    callers (the scheduler job, and Phase 11.6's manual-trigger route)
    get a uniform, inspectable result.

    status:
        OK         pipeline ran (intent_count may still be 0 — "stay put")
        DISARMED   deployment exists but armed=0; nothing was done
        NOT_FOUND  no deployment with this id
        NO_DATA    price fetch returned an empty frame; nothing submitted
        ERROR      an exception was caught — see `error`
    """

    deployment_id: str
    status: str
    ran_at: str
    as_of_timestamp: str | None = None
    intent_count: int = 0
    submit_results: list[SubmitResult] = field(default_factory=list)
    error: str | None = None


# --- ticker resolution + small helpers ------------------------------------


def resolve_price_ticker(t212_ticker: str) -> str:
    """Map a T212 instrument code to the symbol yfinance expects.

    A deployment stores its instrument as a T212 code (e.g. `AAPL_US_EQ`)
    because that is what order placement needs. Price history, though,
    comes from yfinance, which wants the plain symbol (`AAPL`). This is
    the one place the two identifiers are bridged — everything else in
    the pipeline keeps using the T212 code.

    Resolution order:
      1. T212's own `shortName` for that instrument code (authoritative).
      2. Failing that, strip the venue suffix (`AAPL_US_EQ` -> `AAPL`).
      3. Failing that, return the input unchanged.

    Never raises — a deployment run must not die because instrument
    metadata was briefly unavailable.
    """
    try:
        matches = instruments.find_by_ticker(t212_ticker)
    except Exception:  # noqa: BLE001 — metadata lookup must not break a run
        logger.warning(
            "instrument lookup failed for %s; using suffix fallback",
            t212_ticker, exc_info=True,
        )
        matches = []

    if matches:
        short = matches[0].get("shortName")
        if short:
            return str(short)

    if "_" in t212_ticker:
        return t212_ticker.split("_", 1)[0]
    return t212_ticker


def _lookback_days(timeframe: str) -> int:
    return _LOOKBACK_DAYS.get(timeframe, _DEFAULT_LOOKBACK_DAYS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_strategy_code(strategy_id: str) -> str:
    """Read a strategy's source from disk. Raises RuntimeError (caught by
    `run_deployment`) if the strategy row or its code file is missing."""
    with connect() as conn:
        strat = repo.get_strategy(conn, strategy_id)
    if strat is None:
        raise RuntimeError(f"strategy {strategy_id} not found")
    code_path = Path(strat["code_path"])
    if not code_path.exists():
        raise RuntimeError(f"strategy code file missing: {code_path}")
    return code_path.read_text(encoding="utf-8")


# --- the pipeline (one deployment, one fire) ------------------------------


def run_deployment(deployment_id: str) -> DeploymentRunResult:
    """Run the full deploy pipeline once for one deployment.

    This is the function APScheduler calls on every cron tick, and the
    one Phase 11.6's `POST /deployments/{id}/run` will call for a manual
    trigger. It never raises: every failure is caught and reported in the
    returned DeploymentRunResult, because an uncaught exception inside a
    scheduler job is invisible to the user.
    """
    ran_at = _now_iso()

    with connect() as conn:
        deployment = repo.get_deployment(conn, deployment_id)

    if deployment is None:
        logger.warning("run_deployment: no deployment %s", deployment_id)
        return DeploymentRunResult(deployment_id, "NOT_FOUND", ran_at)

    if not deployment["armed"]:
        logger.info("run_deployment: deployment %s is disarmed; skipping", deployment_id)
        return DeploymentRunResult(deployment_id, "DISARMED", ran_at)

    status = "OK"
    error: str | None = None
    as_of: str | None = None
    intents: list[OrderIntent] = []
    submit_results: list[SubmitResult] = []

    try:
        strategy_code = _load_strategy_code(deployment["strategy_id"])

        price_ticker = resolve_price_ticker(deployment["ticker"])
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=_lookback_days(deployment["timeframe"]))
        data = get_prices(price_ticker, deployment["timeframe"], start, end)

        if data is None or data.empty:
            logger.warning(
                "run_deployment %s: empty price data for %s (%s)",
                deployment_id, price_ticker, deployment["timeframe"],
            )
            status = "NO_DATA"
        else:
            # as_of = the last bar, NOT wall-clock now: this is what makes
            # re-firing the job for the same trading day idempotent.
            as_of = data.index[-1].isoformat()
            portfolio = t212_client.get_portfolio()
            intents = compute_intents(
                deployment=deployment,
                strategy_code=strategy_code,
                data=data,
                portfolio=portfolio,
                as_of_timestamp=as_of,
            )
            if intents:
                with connect() as conn:
                    submit_results = submit_intents(
                        conn, deployment_id=deployment_id, intents=intents,
                    )
            n_submitted = sum(1 for r in submit_results if r.status == "SUBMITTED")
            logger.info(
                "run_deployment %s: as_of=%s intents=%d submitted=%d",
                deployment_id, as_of, len(intents), n_submitted,
            )
    except Exception as exc:  # noqa: BLE001 — top-level job boundary
        logger.exception("run_deployment %s failed", deployment_id)
        status = "ERROR"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # last_run_at records the last *fire* of the job — updated on OK,
        # NO_DATA and ERROR alike, so the Tracking UI can tell the job is
        # alive even when it is erroring. (Not reached for DISARMED /
        # NOT_FOUND — those return above, before this try.)
        try:
            with connect() as conn:
                repo.update_deployment_last_run(conn, deployment_id, ran_at)
        except Exception:  # noqa: BLE001
            logger.exception(
                "run_deployment %s: could not update last_run_at", deployment_id,
            )

    return DeploymentRunResult(
        deployment_id=deployment_id,
        status=status,
        ran_at=ran_at,
        as_of_timestamp=as_of,
        intent_count=len(intents),
        submit_results=submit_results,
        error=error,
    )


# --- APScheduler singleton + registration ---------------------------------

_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    """Lazily build the process-wide BackgroundScheduler. Building it does
    NOT start its thread — call `start()` for that."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)
    return _scheduler


def _job_id(deployment_id: str) -> str:
    return f"{_JOB_PREFIX}{deployment_id}"


def register_deployment(deployment: dict[str, Any]) -> None:
    """Add (or replace) the cron job for one deployment.

    `deployment` needs at least `id` and `schedule_cron`. Raises
    ValueError if `schedule_cron` is not a valid 5-field crontab. Phase
    11.6's create-deployment route is expected to validate before
    persisting; registration still fails loudly rather than silently
    dropping the job.
    """
    deployment_id = deployment["id"]
    # Parse (and validate) the crontab first: a bad cron raises here,
    # before any existing job is touched — a deployment never loses its
    # schedule to a typo.
    trigger = CronTrigger.from_crontab(
        deployment["schedule_cron"], timezone=SCHEDULER_TIMEZONE,
    )
    # Drop any prior job for this deployment, then add fresh. This makes
    # register_deployment idempotent whether or not the scheduler thread
    # is running — APScheduler's own `replace_existing` only de-dupes once
    # the scheduler has started.
    unregister_deployment(deployment_id)
    get_scheduler().add_job(
        run_deployment,
        trigger=trigger,
        args=[deployment_id],
        id=_job_id(deployment_id),
        name=f"deployment {deployment_id}",
        replace_existing=True,
    )
    logger.info(
        "registered deployment %s (cron=%r)",
        deployment_id, deployment["schedule_cron"],
    )


def unregister_deployment(deployment_id: str) -> None:
    """Remove a deployment's cron job. A no-op if no such job exists."""
    try:
        get_scheduler().remove_job(_job_id(deployment_id))
        logger.info("unregistered deployment %s", deployment_id)
    except JobLookupError:
        pass


def register_all_armed() -> None:
    """Rebuild jobs for every armed deployment from the database. One bad
    crontab is logged and skipped — it does not stop the others."""
    with connect() as conn:
        armed = repo.list_deployments(conn, armed_only=True)
    for deployment in armed:
        try:
            register_deployment(deployment)
        except Exception:  # noqa: BLE001 — one bad cron must not skip the rest
            logger.exception(
                "could not register deployment %s (cron=%r)",
                deployment.get("id"), deployment.get("schedule_cron"),
            )
    logger.info("register_all_armed: %d armed deployment(s) processed", len(armed))


def start() -> None:
    """Start the scheduler thread (if not already running) and register a
    job for every armed deployment. Safe to call once at server startup."""
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        logger.info("deployment scheduler started (tz=%s)", SCHEDULER_TIMEZONE)
    try:
        register_all_armed()
    except Exception:  # noqa: BLE001 — scheduler wiring must not crash boot
        logger.exception("register_all_armed failed during start()")


def shutdown() -> None:
    """Stop the scheduler thread. Safe to call when it is not running."""
    sched = _scheduler
    if sched is not None and sched.running:
        sched.shutdown(wait=False)
        logger.info("deployment scheduler stopped")


def _reset_scheduler() -> None:
    """Test seam: drop the singleton (shutting it down first if running)
    so each test gets a clean scheduler. Mirrors
    `t212.client._reset_client_cache`."""
    global _scheduler
    sched = _scheduler
    _scheduler = None
    if sched is not None and sched.running:
        try:
            sched.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            logger.exception("scheduler shutdown during reset failed")
