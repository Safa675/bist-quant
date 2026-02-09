"""
APScheduler for daily regime updates

Runs at market close (18:00 Istanbul time) to:
1. Fetch new market data
2. Calculate features
3. Run ensemble prediction
4. Broadcast WebSocket updates
5. Trigger email alerts if configured
"""

import asyncio
from datetime import datetime
from typing import Callable, Optional
import pytz

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

# Import components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RegimeScheduler:
    """
    Scheduler for automated regime updates

    Features:
    - Daily updates at market close
    - Manual trigger support
    - Callback hooks for alerts
    - Error handling and retry
    """

    # Istanbul timezone
    ISTANBUL_TZ = pytz.timezone('Europe/Istanbul')

    def __init__(self):
        if not SCHEDULER_AVAILABLE:
            raise ImportError(
                "APScheduler not installed. Install with: pip install apscheduler"
            )

        self.scheduler = AsyncIOScheduler(timezone=self.ISTANBUL_TZ)
        self.is_running = False

        # Callbacks
        self.on_update_complete: Optional[Callable] = None
        self.on_regime_change: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # State
        self.last_update: Optional[datetime] = None
        self.last_regime: Optional[str] = None
        self.update_count = 0

    def start(self):
        """Start the scheduler"""
        if not self.is_running:
            # Schedule daily update at 18:00 Istanbul time (after market close)
            self.scheduler.add_job(
                self._daily_update,
                CronTrigger(hour=18, minute=0, timezone=self.ISTANBUL_TZ),
                id='daily_regime_update',
                name='Daily Regime Update',
                replace_existing=True
            )

            # Schedule heartbeat every 5 minutes during market hours
            self.scheduler.add_job(
                self._heartbeat,
                CronTrigger(
                    day_of_week='mon-fri',
                    hour='9-18',
                    minute='*/5',
                    timezone=self.ISTANBUL_TZ
                ),
                id='market_hours_heartbeat',
                name='Market Hours Heartbeat',
                replace_existing=True
            )

            self.scheduler.start()
            self.is_running = True
            print(f"Scheduler started at {datetime.now(self.ISTANBUL_TZ)}")
            print("  Daily update scheduled for 18:00 Istanbul time")

    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            print("Scheduler stopped")

    async def _daily_update(self):
        """
        Daily regime update task

        1. Fetch latest data
        2. Calculate features
        3. Run ensemble prediction
        4. Check for regime change
        5. Trigger callbacks
        """
        print(f"\n{'='*60}")
        print(f"DAILY REGIME UPDATE - {datetime.now(self.ISTANBUL_TZ)}")
        print(f"{'='*60}")

        try:
            # Import components (lazy import to avoid circular deps)
            from regime_filter import RegimeFilter
            from regime_models import SimplifiedRegimeClassifier
            from models.ensemble_regime import EnsembleRegimeModel

            # Initialize regime filter
            rf = RegimeFilter()

            # Run pipeline
            print("\n[1/4] Loading data...")
            rf.load_data(fetch_usdtry=True, load_stocks=False)

            print("\n[2/4] Calculating features...")
            rf.calculate_features()

            print("\n[3/4] Classifying regimes...")
            rf.classify_regimes()

            # Get simplified regime
            simple_classifier = SimplifiedRegimeClassifier()
            simplified = simple_classifier.classify(rf.regimes)['simplified_regime']

            # Get current regime
            current_regime = simplified.iloc[-1]
            current_date = simplified.index[-1]

            print(f"\n[4/4] Current Regime: {current_regime}")

            # Check for regime change
            regime_changed = False
            if self.last_regime is not None and current_regime != self.last_regime:
                regime_changed = True
                print(f"  REGIME CHANGE: {self.last_regime} -> {current_regime}")

                if self.on_regime_change:
                    await self._safe_callback(
                        self.on_regime_change,
                        previous_regime=self.last_regime,
                        new_regime=current_regime,
                        date=current_date
                    )

            # Update state
            self.last_regime = current_regime
            self.last_update = datetime.now(self.ISTANBUL_TZ)
            self.update_count += 1

            # Trigger update complete callback
            if self.on_update_complete:
                await self._safe_callback(
                    self.on_update_complete,
                    regime=current_regime,
                    date=current_date,
                    regime_changed=regime_changed,
                    features=rf.features.iloc[-1].to_dict() if rf.features is not None else {},
                    rf=rf,
                    simplified=simplified
                )

            print(f"\nUpdate #{self.update_count} completed successfully")
            return {
                'success': True,
                'regime': current_regime,
                'date': current_date,
                'regime_changed': regime_changed
            }

        except Exception as e:
            print(f"\nERROR in daily update: {e}")

            if self.on_error:
                await self._safe_callback(self.on_error, error=str(e))

            return {'success': False, 'error': str(e)}

    async def _heartbeat(self):
        """Heartbeat during market hours"""
        print(f"Heartbeat: {datetime.now(self.ISTANBUL_TZ).strftime('%H:%M:%S')}")

    async def _safe_callback(self, callback: Callable, **kwargs):
        """Safely execute callback (sync or async)"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(**kwargs)
            else:
                callback(**kwargs)
        except Exception as e:
            print(f"Callback error: {e}")

    async def trigger_manual_update(self):
        """Manually trigger an update (outside scheduled time)"""
        print("Manual update triggered")
        return await self._daily_update()

    def get_status(self) -> dict:
        """Get scheduler status"""
        jobs = []
        if self.scheduler.running:
            for job in self.scheduler.get_jobs():
                jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None
                })

        return {
            'is_running': self.is_running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'last_regime': self.last_regime,
            'update_count': self.update_count,
            'jobs': jobs
        }

    def add_custom_job(self, func: Callable, trigger, job_id: str, **kwargs):
        """
        Add a custom scheduled job

        Args:
            func: Function to call
            trigger: APScheduler trigger (CronTrigger, IntervalTrigger, etc.)
            job_id: Unique job identifier
            **kwargs: Additional job arguments
        """
        self.scheduler.add_job(func, trigger, id=job_id, **kwargs)
        print(f"Added custom job: {job_id}")


# Global scheduler instance
scheduler = RegimeScheduler() if SCHEDULER_AVAILABLE else None


def get_scheduler() -> RegimeScheduler:
    """Get the global scheduler instance"""
    if scheduler is None:
        raise RuntimeError("Scheduler not available. Install apscheduler.")
    return scheduler
