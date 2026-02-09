#!/usr/bin/env python3
"""
BIST Regime Filter API - Entry Point

Run the FastAPI server with:
    python run_api.py

Or with uvicorn directly:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Environment Variables:
    TCMB_EVDS_API_KEY - TCMB EVDS API key for Turkish data
    SMTP_HOST - SMTP server host
    SMTP_PORT - SMTP server port
    SMTP_USERNAME - SMTP username
    SMTP_PASSWORD - SMTP password
    ALERT_SENDER_EMAIL - Email address to send alerts from
    ALERT_RECIPIENTS - Comma-separated list of recipient emails
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
except ImportError:
    pass


def check_dependencies():
    """Check runtime dependencies and return missing package groups."""
    dependencies = [
        ('pandas', 'pandas', True),
        ('numpy', 'numpy', True),
        ('xgboost', 'xgboost', False),
        ('hmmlearn', 'hmmlearn', False),
        ('fastapi', 'fastapi', True),
        ('uvicorn', 'uvicorn', True),
        ('torch', 'PyTorch', False),
        ('apscheduler', 'APScheduler', True),
    ]

    missing_required = []
    missing_optional = []

    print("\nDependency Check:")
    for module, name, required in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            if required:
                missing_required.append(name)
            else:
                missing_optional.append(name)

    return missing_required, missing_optional


def main():
    """Run the API server"""
    parser = argparse.ArgumentParser(description="Run BIST Regime Filter API server")
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )
    args = parser.parse_args()

    from config import API_CONFIG

    print("="*70)
    print("BIST REGIME FILTER API")
    print("="*70)
    print(f"\nVersion: 2.0.0")
    print(f"Python: {sys.version}")

    missing_required, missing_optional = check_dependencies()
    if missing_optional:
        print(f"\n  Optional packages missing: {', '.join(missing_optional)}")

    if missing_required:
        print("\nERROR: Required dependencies are missing. API startup aborted.")
        print(f"Missing: {', '.join(missing_required)}")
        print('Install with: pip install -r "Regime Filter/requirements.txt"')
        return 1

    if args.check_deps:
        print("\nDependency check passed.")
        return 0

    # Check TCMB API key
    tcmb_key = os.environ.get('TCMB_EVDS_API_KEY')
    if tcmb_key:
        print(f"\n  TCMB API Key: Configured")
    else:
        print(f"\n  TCMB API Key: Not set (will use proxies)")

    # Check email config
    smtp_user = os.environ.get('SMTP_USERNAME')
    if smtp_user:
        print(f"  Email Alerts: Configured ({smtp_user})")
    else:
        print(f"  Email Alerts: Not configured")

    print("\n" + "="*70)
    print("ENDPOINTS")
    print("="*70)
    print(f"""
  REST API:
    GET  /                    - API info
    GET  /health              - Health check
    GET  /regime/current      - Current regime prediction
    GET  /regime/history      - Historical regimes
    GET  /regime/prediction   - N-day ahead forecast
    GET  /features/current    - Current feature values
    POST /regime/backtest     - Run backtest
    POST /regime/refresh      - Manual data refresh

  WebSocket:
    WS   /ws/regime           - Real-time regime updates

  Documentation:
    GET  /docs                - Swagger UI
    GET  /redoc               - ReDoc
    """)

    print("="*70)
    print(f"Starting server on http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("="*70 + "\n")

    import uvicorn

    # Run server
    uvicorn.run(
        "api.main:app",
        host=API_CONFIG.get('host', '0.0.0.0'),
        port=API_CONFIG.get('port', 8000),
        reload=API_CONFIG.get('reload', False),
        log_level="info"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
