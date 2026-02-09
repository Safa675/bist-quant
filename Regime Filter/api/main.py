"""
FastAPI application for Regime Filter API

Endpoints:
- GET  /regime/current     - Current regime prediction
- GET  /regime/history     - Historical regimes
- GET  /regime/prediction  - N-day ahead forecast
- GET  /features/current   - Latest feature values
- GET  /health             - API health check
- POST /regime/backtest    - Run backtest
- WS   /ws/regime          - Real-time updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from typing import Optional, List
import asyncio
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import API models
from api.models import (
    RegimeResponse, RegimeHistoryResponse, RegimeHistoryItem,
    PredictionResponse, FeatureResponse, FeatureValue,
    BacktestRequest, BacktestResponse, BacktestMetrics,
    HealthResponse, RegimeProbabilities, ModelAgreement,
    RegimeEnum
)
from api.websocket import manager, heartbeat_task
from api.scheduler import scheduler, get_scheduler, SCHEDULER_AVAILABLE

# Application state
class AppState:
    """Global application state"""
    def __init__(self):
        self.regime_filter = None
        self.features = None
        self.regimes = None
        self.simplified_regimes = None
        self.ensemble_model = None
        self.last_update = None
        self.start_time = datetime.now()
        self.models_loaded = {
            'regime_filter': False,
            'ensemble': False,
            'xgboost': False,
            'lstm': False,
            'hmm': False
        }

state = AppState()

# Regime recommendations
REGIME_RECOMMENDATIONS = {
    'Bull': "Increase exposure, trend-following strategies work well",
    'Bear': "Reduce exposure, consider defensive positioning",
    'Stress': "Minimize exposure, use wide stops, avoid illiquid names",
    'Choppy': "Range trading, mean reversion strategies, reduce position size",
    'Recovery': "Gradual re-entry, focus on quality names"
}

REGIME_ORDER = ['Bull', 'Bear', 'Stress', 'Choppy', 'Recovery']


def _uniform_probabilities() -> dict:
    """Return a uniform 5-regime probability distribution."""
    p = 1.0 / len(REGIME_ORDER)
    return {regime: p for regime in REGIME_ORDER}


def _update_state_from_scheduler(rf, simplified_regimes):
    """Update API runtime state using objects produced by scheduler."""
    if simplified_regimes is None or rf is None:
        return

    # Keep simplified regimes as Series for scalar access in endpoints.
    if hasattr(simplified_regimes, 'columns') and 'simplified_regime' in simplified_regimes.columns:
        simplified_regimes = simplified_regimes['simplified_regime']

    state.regime_filter = rf
    state.features = rf.features
    state.regimes = rf.regimes
    state.simplified_regimes = simplified_regimes
    state.last_update = datetime.now()
    state.models_loaded['regime_filter'] = True


def _forecast_markov_probabilities(regime_history, start_regime: str, horizon: int) -> dict:
    """
    Forecast regime probabilities using an empirical Markov transition matrix.
    """
    if horizon < 1:
        return _uniform_probabilities()

    if start_regime not in REGIME_ORDER or regime_history is None or len(regime_history) < 2:
        return _uniform_probabilities()

    clean = regime_history.dropna().astype(str)
    transitions = np.zeros((len(REGIME_ORDER), len(REGIME_ORDER)), dtype=float)
    regime_to_idx = {regime: i for i, regime in enumerate(REGIME_ORDER)}

    for current, nxt in zip(clean.iloc[:-1], clean.iloc[1:]):
        if current in regime_to_idx and nxt in regime_to_idx:
            transitions[regime_to_idx[current], regime_to_idx[nxt]] += 1.0

    if transitions.sum() == 0:
        return _uniform_probabilities()

    # Row-normalize; fall back to self-transition when a row has no observations.
    for i in range(len(REGIME_ORDER)):
        row_sum = transitions[i].sum()
        if row_sum > 0:
            transitions[i] /= row_sum
        else:
            transitions[i, i] = 1.0

    state_vec = np.zeros(len(REGIME_ORDER), dtype=float)
    state_vec[regime_to_idx[start_regime]] = 1.0

    future = state_vec @ np.linalg.matrix_power(transitions, horizon)
    total = float(future.sum())
    if total <= 0:
        return _uniform_probabilities()

    future /= total
    return {regime: float(future[i]) for i, regime in enumerate(REGIME_ORDER)}


async def load_models():
    """Load models on startup"""
    print("Loading models...")

    try:
        from regime_filter import RegimeFilter
        from regime_models import SimplifiedRegimeClassifier

        # Initialize regime filter
        state.regime_filter = RegimeFilter()
        state.regime_filter.load_data(fetch_usdtry=True, load_stocks=False)
        state.regime_filter.calculate_features()
        state.regime_filter.classify_regimes()

        state.features = state.regime_filter.features
        state.regimes = state.regime_filter.regimes
        state.models_loaded['regime_filter'] = True

        # Get simplified regimes
        simple_classifier = SimplifiedRegimeClassifier()
        state.simplified_regimes = simple_classifier.classify(
            state.regimes
        )['simplified_regime']

        state.last_update = datetime.now()
        print("Regime filter loaded successfully")

    except Exception as e:
        print(f"Error loading regime filter: {e}")

    # Try to load ensemble model
    try:
        from models.ensemble_regime import EnsembleRegimeModel

        ensemble_path = Path(__file__).parent.parent / "outputs" / "ensemble_model"
        if ensemble_path.exists():
            state.ensemble_model = EnsembleRegimeModel.load(ensemble_path)
            state.models_loaded['ensemble'] = True
            state.models_loaded['xgboost'] = 'xgboost' in state.ensemble_model.available_models
            state.models_loaded['lstm'] = 'lstm' in state.ensemble_model.available_models
            state.models_loaded['hmm'] = 'hmm' in state.ensemble_model.available_models
            print("Ensemble model loaded")
        else:
            print("No pre-trained ensemble model found")

    except Exception as e:
        print(f"Could not load ensemble model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting Regime Filter API...")
    await load_models()

    # Start scheduler if available
    if SCHEDULER_AVAILABLE and scheduler:
        scheduler.start()

        # Set up callbacks for WebSocket broadcasts
        async def on_update(regime, date, regime_changed, features, rf=None, simplified=None):
            # Keep REST API state in sync with scheduled updates.
            if rf is not None and simplified is not None:
                _update_state_from_scheduler(rf, simplified)

            await manager.broadcast_regime_update({
                'regime': regime,
                'date': str(date),
                'regime_changed': regime_changed
            })

        async def on_regime_change(previous_regime, new_regime, date):
            await manager.broadcast_alert('regime_change', {
                'previous': previous_regime,
                'new': new_regime,
                'date': str(date),
                'message': f"Regime changed from {previous_regime} to {new_regime}"
            })

        scheduler.on_update_complete = on_update
        scheduler.on_regime_change = on_regime_change

    # Start heartbeat task
    heartbeat = asyncio.create_task(heartbeat_task(30))

    yield

    # Shutdown
    print("Shutting down...")
    heartbeat.cancel()
    if scheduler:
        scheduler.stop()


# Create FastAPI app
app = FastAPI(
    title="BIST Regime Filter API",
    description="Market regime classification and prediction API for BIST (Turkish Stock Market)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ENDPOINTS ====================

@app.get("/", tags=["Root"])
async def root():
    """API root - returns welcome message"""
    return {
        "message": "BIST Regime Filter API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    uptime = (datetime.now() - state.start_time).total_seconds()

    return HealthResponse(
        status="healthy" if state.models_loaded['regime_filter'] else "degraded",
        version="1.0.0",
        models_loaded=state.models_loaded,
        last_update=state.last_update,
        uptime_seconds=uptime
    )


@app.get("/regime/current", response_model=RegimeResponse, tags=["Regime"])
async def get_current_regime():
    """
    Get current market regime prediction

    Returns the latest regime classification with confidence scores
    and per-model agreement (if ensemble is available).
    """
    if state.simplified_regimes is None:
        raise HTTPException(status_code=503, detail="Regime filter not loaded")

    # Get latest regime
    current_regime = state.simplified_regimes.iloc[-1]
    current_date = state.simplified_regimes.index[-1]

    # Default probabilities (from rule-based classifier)
    probabilities = RegimeProbabilities(
        Bull=1.0 if current_regime == 'Bull' else 0.0,
        Bear=1.0 if current_regime == 'Bear' else 0.0,
        Stress=1.0 if current_regime == 'Stress' else 0.0,
        Choppy=1.0 if current_regime == 'Choppy' else 0.0,
        Recovery=1.0 if current_regime == 'Recovery' else 0.0
    )

    confidence = 0.8  # Default confidence for rule-based
    disagreement = 0.0
    model_agreement = None

    # Use ensemble if available
    if state.ensemble_model and state.features is not None:
        try:
            result = state.ensemble_model.predict_current(
                state.features, state.simplified_regimes
            )
            current_regime = result['prediction']
            confidence = result['confidence']
            disagreement = result['disagreement']

            probabilities = RegimeProbabilities(**result['probabilities'])

            if result.get('model_agreement'):
                model_agreement = ModelAgreement(**{
                    k: RegimeEnum(v) for k, v in result['model_agreement'].items()
                })

        except Exception as e:
            print(f"Ensemble prediction failed, using rule-based: {e}")

    return RegimeResponse(
        date=current_date,
        regime=RegimeEnum(current_regime),
        confidence=confidence,
        probabilities=probabilities,
        model_agreement=model_agreement,
        disagreement=disagreement,
        recommendation=REGIME_RECOMMENDATIONS.get(current_regime, "Use caution")
    )


@app.get("/regime/history", response_model=RegimeHistoryResponse, tags=["Regime"])
async def get_regime_history(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return")
):
    """
    Get historical regime classifications

    Returns regime history for the specified date range.
    """
    if state.simplified_regimes is None or state.regimes is None:
        raise HTTPException(status_code=503, detail="Regime filter not loaded")

    # Filter by date range
    regimes = state.simplified_regimes.copy()

    if start_date:
        regimes = regimes[regimes.index >= str(start_date)]
    if end_date:
        regimes = regimes[regimes.index <= str(end_date)]

    # Limit results
    regimes = regimes.tail(limit)

    # Build history items
    history = []
    for idx, regime in regimes.items():
        item = RegimeHistoryItem(
            date=idx,
            regime=RegimeEnum(regime),
            confidence=0.8  # Default for rule-based
        )

        # Add detailed regimes if available
        if idx in state.regimes.index:
            row = state.regimes.loc[idx]
            item.volatility_regime = row.get('volatility_regime')
            item.trend_regime = row.get('trend_regime')
            item.risk_regime = row.get('risk_regime')

        history.append(item)

    return RegimeHistoryResponse(
        start_date=regimes.index[0] if len(regimes) > 0 else datetime.now(),
        end_date=regimes.index[-1] if len(regimes) > 0 else datetime.now(),
        count=len(history),
        history=history
    )


@app.get("/regime/prediction", response_model=PredictionResponse, tags=["Regime"])
async def get_regime_prediction(
    horizon: int = Query(5, ge=1, le=20, description="Days ahead to predict")
):
    """
    Get N-day ahead regime prediction

    Uses the ensemble model (if available) to predict future regime.
    """
    if state.features is None:
        raise HTTPException(status_code=503, detail="Features not loaded")

    current_date = state.features.index[-1]
    target_date = current_date + timedelta(days=horizon)

    # Build a true horizon-dependent fallback forecast from historical transitions.
    latest_regime = state.simplified_regimes.iloc[-1] if state.simplified_regimes is not None else "Unknown"
    if latest_regime in REGIME_ORDER:
        probs_dict = _forecast_markov_probabilities(state.simplified_regimes, latest_regime, horizon)
        current_regime = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[current_regime]
    else:
        probs_dict = _uniform_probabilities()
        current_regime = "Unknown"
        confidence = max(probs_dict.values())
    probabilities = RegimeProbabilities(**probs_dict)

    # Use ensemble only when its trained horizon matches requested horizon.
    # Otherwise keep the Markov fallback to avoid mislabeling current-state output as forecast.
    if state.ensemble_model:
        try:
            if state.ensemble_model.forecast_horizon == horizon:
                result = state.ensemble_model.predict_current(
                    state.features, state.simplified_regimes
                )
                current_regime = result['prediction']
                confidence = result['confidence']
                probabilities = RegimeProbabilities(**result['probabilities'])
            elif state.simplified_regimes is not None:
                # Improve fallback start-state using ensemble's current prediction.
                current_result = state.ensemble_model.predict_current(
                    state.features, state.simplified_regimes
                )
                start_regime = current_result['prediction']
                probs_dict = _forecast_markov_probabilities(state.simplified_regimes, start_regime, horizon)
                current_regime = max(probs_dict, key=probs_dict.get)
                confidence = probs_dict[current_regime]
                probabilities = RegimeProbabilities(**probs_dict)
        except Exception as e:
            print(f"Prediction failed: {e}")

    return PredictionResponse(
        prediction_date=current_date,
        target_date=target_date,
        forecast_horizon=horizon,
        regime=RegimeEnum(current_regime),
        confidence=confidence,
        probabilities=probabilities
    )


@app.get("/features/current", response_model=FeatureResponse, tags=["Features"])
async def get_current_features(
    features: Optional[List[str]] = Query(
        None,
        description="Specific features to return (all if not specified)"
    )
):
    """
    Get current feature values

    Returns the latest calculated feature values used for regime classification.
    """
    if state.features is None:
        raise HTTPException(status_code=503, detail="Features not loaded")

    current_date = state.features.index[-1]
    current_features = state.features.iloc[-1]

    # Filter features if specified
    if features:
        available = [f for f in features if f in current_features.index]
        if not available:
            raise HTTPException(status_code=400, detail="None of the specified features found")
        feature_list = available
    else:
        feature_list = current_features.index.tolist()[:50]  # Limit to top 50

    # Build feature values
    feature_values = []
    for name in feature_list:
        value = current_features[name]
        if not isinstance(value, (int, float)) or (hasattr(value, 'isna') and value.isna()):
            continue

        feature_values.append(FeatureValue(
            name=name,
            value=float(value),
            percentile=None,  # Could calculate percentile if needed
            description=_get_feature_description(name)
        ))

    return FeatureResponse(
        date=current_date,
        features=feature_values
    )


def _get_feature_description(name: str) -> str:
    """Get human-readable feature description"""
    descriptions = {
        'realized_vol_20d': '20-day realized volatility (annualized)',
        'realized_vol_60d': '60-day realized volatility (annualized)',
        'return_20d': '20-day cumulative return',
        'return_60d': '60-day cumulative return',
        'max_drawdown_20d': 'Maximum drawdown over 20 days',
        'usdtry_momentum_20d': 'USD/TRY 20-day momentum',
        'volume_ratio': 'Current volume vs 20-day average',
        'viop30_proxy': 'VIOP30 (Turkish VIX) proxy',
        'cds_proxy': 'Turkey CDS spread proxy',
        'yield_curve_slope': '10Y-2Y yield curve slope',
        'real_rate': 'Real interest rate (policy - inflation)'
    }
    return descriptions.get(name, name.replace('_', ' ').title())


@app.post("/regime/backtest", response_model=BacktestResponse, tags=["Backtest"])
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest with the specified parameters

    Tests regime-based trading strategies on historical data.
    """
    if state.regime_filter is None:
        raise HTTPException(status_code=503, detail="Regime filter not loaded")

    try:
        from evaluation import RegimeBacktester

        if state.regime_filter.data is None or state.simplified_regimes is None:
            raise HTTPException(status_code=503, detail="Backtest data not available")

        # Backtester expects close-price series and aligned regime labels.
        if 'XU100_Close' not in state.regime_filter.data.columns:
            raise HTTPException(status_code=500, detail="XU100 close prices not available")

        prices = state.regime_filter.data['XU100_Close']
        regimes = state.simplified_regimes

        # Apply requested date window before backtesting.
        end_date = request.end_date or date.today()
        start_str = str(request.start_date)
        end_str = str(end_date)

        prices = prices[(prices.index >= start_str) & (prices.index <= end_str)]
        regimes = regimes[(regimes.index >= start_str) & (regimes.index <= end_str)]

        if len(prices) == 0 or len(regimes) == 0:
            raise HTTPException(status_code=400, detail="No data in requested date range")

        # Initialize backtester
        backtester = RegimeBacktester(
            prices=prices,
            regimes=regimes
        )

        # Run backtest
        if request.strategy == "regime_filter":
            avoid_regimes = [r.value for r in request.avoid_regimes]
            results = backtester.backtest_regime_filter(avoid_regimes=avoid_regimes)
        elif request.strategy == "regime_rotation":
            allocation = request.allocation_weights or {
                'Bull': 1.5, 'Bear': 0.2, 'Stress': 0.0, 'Choppy': 0.5, 'Recovery': 1.0
            }
            results = backtester.backtest_regime_rotation(allocation)
        else:
            results = backtester.backtest_buy_and_hold()

        # Build response
        metrics = BacktestMetrics(
            total_return=results.get('total_return', 0),
            annualized_return=results.get('annual_return', 0),
            sharpe_ratio=results.get('sharpe_ratio', 0),
            sortino_ratio=results.get('sortino_ratio'),
            max_drawdown=results.get('max_drawdown', 0),
            win_rate=results.get('win_rate', 0),
            n_trades=results.get('n_trades', 0)
        )

        return BacktestResponse(
            strategy=request.strategy,
            start_date=request.start_date,
            end_date=end_date,
            metrics=metrics,
            regime_performance={}  # Could add per-regime breakdown
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")


@app.post("/regime/refresh", tags=["Admin"])
async def refresh_data(background_tasks: BackgroundTasks):
    """
    Refresh data and recalculate regimes

    Triggers a manual data update (normally runs daily at market close).
    """
    if SCHEDULER_AVAILABLE and scheduler:
        background_tasks.add_task(scheduler.trigger_manual_update)
        return {"status": "refresh_started", "message": "Data refresh started in background"}
    else:
        # Direct refresh
        try:
            await load_models()
            return {"status": "success", "message": "Data refreshed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Refresh failed: {e}")


@app.get("/scheduler/status", tags=["Admin"])
async def get_scheduler_status():
    """Get scheduler status and next run times"""
    if not SCHEDULER_AVAILABLE or scheduler is None:
        return {"available": False, "message": "Scheduler not installed"}

    return {"available": True, **scheduler.get_status()}


# ==================== WEBSOCKET ====================

@app.websocket("/ws/regime")
async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time regime updates

    Connect to receive:
    - regime_update: When regime is recalculated
    - alert: On regime changes or stress warnings
    - heartbeat: Periodic keepalive
    """
    await manager.connect(websocket, client_id)

    # Send current state on connect
    if state.simplified_regimes is not None:
        current = state.simplified_regimes.iloc[-1]
        await manager.send_personal_message({
            'type': 'initial_state',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'current_regime': current,
                'last_update': state.last_update.isoformat() if state.last_update else None
            }
        }, websocket)

    try:
        while True:
            # Wait for messages (could handle client commands here)
            data = await websocket.receive_text()

            # Echo back for now
            await manager.send_personal_message({
                'type': 'echo',
                'timestamp': datetime.now().isoformat(),
                'data': {'received': data}
            }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==================== FACTORY ====================

def create_app() -> FastAPI:
    """Factory function to create app instance"""
    return app


if __name__ == "__main__":
    import uvicorn

    print("="*60)
    print("Starting BIST Regime Filter API")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /regime/current    - Current regime")
    print("  GET  /regime/history    - Historical regimes")
    print("  GET  /regime/prediction - Future prediction")
    print("  GET  /features/current  - Current features")
    print("  GET  /health            - Health check")
    print("  POST /regime/backtest   - Run backtest")
    print("  WS   /ws/regime         - Real-time updates")
    print("\nDocs available at: http://localhost:8000/docs")
    print("="*60)

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
