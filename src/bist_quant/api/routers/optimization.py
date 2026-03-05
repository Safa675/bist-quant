"""Strategy parameter optimization endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bist_quant.api.schemas import OptimizationRunRequest
from bist_quant.services import CoreBackendService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimize", tags=["optimization"])


@router.post("/run")
def optimization_run(
    payload: OptimizationRunRequest,
    async_job: bool = Query(default=True),
) -> dict[str, Any]:
    """Run strategy parameter optimization.

    By default runs as an async job since optimization is compute-heavy.
    Set ``?async_job=false`` for synchronous execution.
    """
    try:
        base_request = {
            "factor_name": payload.signal,
            **payload.params,
        }

        if not async_job:
            core = CoreBackendService(strict_paths=False)
            return core.optimize_strategy(
                base_request=base_request,
                method=payload.method,
                parameter_space=payload.parameter_space,
                max_trials=payload.max_trials,
                random_seed=payload.random_seed,
                train_ratio=payload.train_ratio,
                walk_forward_splits=payload.walk_forward_splits,
                constraints=payload.constraints,
                objective=payload.objective,
            )

        # Async job path
        from bist_quant.api.main import job_manager

        def _run() -> dict[str, Any]:
            core = CoreBackendService(strict_paths=False)
            return core.optimize_strategy(
                base_request=base_request,
                method=payload.method,
                parameter_space=payload.parameter_space,
                max_trials=payload.max_trials,
                random_seed=payload.random_seed,
                train_ratio=payload.train_ratio,
                walk_forward_splits=payload.walk_forward_splits,
                constraints=payload.constraints,
                objective=payload.objective,
            )

        record = job_manager.create(
            kind="optimize",
            fn=_run,
            meta={"signal": payload.signal, "method": payload.method},
            request=payload.model_dump(),
        )
        return job_manager.to_dict(record)
    except Exception as exc:
        logger.exception("optimization_run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
