"""
Main ViewModel - connects View with Models
Handles presentation logic, threading, and UI state
"""
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from models.data_manager import DataManager
from models.model_manager import ModelManager
from utils.time_utils import SydneyTimeUtils
from utils.queue_handler import QueueHandler

class MainViewModel:
    """ViewModel for the main window - handles all business logic for the View"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_queue = QueueHandler()
        self.data_manager = DataManager(config, self.log_queue)
        self.model_manager = ModelManager(config, self.log_queue)
        self.time_utils = SydneyTimeUtils()
        
        # State variables (observed by View)
        self.last_data_date = None
        self.countdown = "00:00:00"
        self.is_updating = False
        self.is_training = False
        self.is_predicting = False
        self.auto_run_enabled = False
        self.last_auto_run_date = None
        self.auto_run_hour, self.auto_run_minute = self._parse_time(
            config.get('schedule', {}).get('auto_run_time', '15:30'))
        self.market_close_hour, self.market_close_minute = self._parse_time(
            config.get('schedule', {}).get('market_close_time', '16:00'))
        
        # Callbacks (set by View)
        self.on_state_changed = None
        self.on_log_updated = None
        
        # Initialize
        self._refresh_last_date()
    
    @staticmethod
    def _parse_time(time_str: str) -> tuple:
        """Parse a time string like '15:30' or '3:30pm' into (hour, minute)."""
        time_str = time_str.strip().lower()
        is_pm = time_str.endswith('pm')
        is_am = time_str.endswith('am')
        if is_pm or is_am:
            time_str = time_str[:-2].strip()
        parts = time_str.split(':')
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if is_pm and hour != 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0
        return hour, minute
    
    def save_schedule(self):
        """Persist current schedule times to config.json."""
        import json
        self.config.setdefault('schedule', {})
        self.config['schedule']['auto_run_time'] = f"{self.auto_run_hour:02d}:{self.auto_run_minute:02d}"
        self.config['schedule']['market_close_time'] = f"{self.market_close_hour:02d}:{self.market_close_minute:02d}"
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.log_queue.put(f"⚠ Could not save schedule: {e}", 'warning')
    
    def _refresh_last_date(self):
        """Refresh last data date from storage"""
        date = self.data_manager.get_last_stored_date()
        self.last_data_date = date.strftime('%d/%m/%Y') if date else "No data"
        if self.on_state_changed:
            self.on_state_changed()
    
    def update_countdown(self):
        """Update countdown string"""
        h, m, s = self.time_utils.get_countdown_to(self.market_close_hour, self.market_close_minute)
        self.countdown = f"{h:02d}:{m:02d}:{s:02d}"
        if self.on_state_changed:
            self.on_state_changed()
    
    def check_auto_run(self) -> bool:
        """Check if auto-run should trigger"""
        if not self.auto_run_enabled:
            return False
        
        if self.time_utils.is_auto_run_time(self.auto_run_hour, self.auto_run_minute):
            today = datetime.now().date()
            if self.last_auto_run_date != today:
                self.last_auto_run_date = today
                return True
        return False
    
    # ========== Background Operations ==========
    
    def update_data_async(self, callback=None):
        """Start asynchronous data update"""
        if self.is_updating:
            return
        
        self.is_updating = True
        if self.on_state_changed:
            self.on_state_changed()
        
        def worker():
            try:
                self.log_queue.put("Starting data update...", 'info')
                result = self.data_manager.update_local_data()
                
                if result["success"]:
                    self.log_queue.put(f"Data update completed. {result['message']}", 'success')
                    self._refresh_last_date()
                else:
                    self.log_queue.put(f"Error updating data: {result['message']}", 'error')
            except Exception as e:
                self.log_queue.put(f"Unexpected error: {e}", 'error')
            finally:
                self.is_updating = False
                if self.on_state_changed:
                    self.on_state_changed()
                if callback:
                    callback()
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    def train_model_async(self, callback=None):
        """Start asynchronous model training"""
        if self.is_training:
            return
        
        self.is_training = True
        if self.on_state_changed:
            self.on_state_changed()
        
        def worker():
            try:
                # ── Load previous training snapshot for comparison
                prev = self.model_manager.load_training_snapshot()
                if prev:
                    self.log_queue.put(f"\nPrevious model (trained {prev.get('trained_at', '?')[:16]}):", 'info')
                    self.log_queue.put(
                        f"  Train acc: {prev['train_accuracy']:.3f}  "
                        f"Test acc: {prev['test_accuracy']:.3f}", 'info'
                    )
                    prev_cal = prev.get('calibration')
                    if prev_cal:
                        self.log_queue.put(
                            f"  ECE: {prev_cal['ece']:.4f}  MCE: {prev_cal['mce']:.4f}", 'info'
                        )
                    prev_feats = prev.get('feature_importance', [])
                    if prev_feats:
                        self.log_queue.put("  Top 10 features:", 'info')
                        for feat in prev_feats:
                            self.log_queue.put(
                                f"    {feat['feature']}: {feat['importance']:.4f}", 'info'
                            )
                    self.log_queue.put("", 'info')  # blank line separator

                self.log_queue.put("Loading combined data...", 'info')
                combined = self.data_manager.prepare_combined_data()
                
                if combined.empty:
                    self.log_queue.put("No data available. Please update data first.", 'error')
                    return
                
                self.log_queue.put("Engineering features...", 'info')
                combined = self.model_manager.engineer_features(combined)
                
                if combined.empty:
                    self.log_queue.put("No valid features could be created from the data.", 'error')
                    return
                
                self.log_queue.put("Training model...", 'info')
                result = self.model_manager.train(combined)
                
                if result["success"]:
                    self.log_queue.put(f"Model training completed.", 'success')
                    self.log_queue.put(f"Train accuracy: {result['train_accuracy']:.3f}", 'info')
                    self.log_queue.put(f"Test accuracy: {result['test_accuracy']:.3f}", 'info')
                    
                    # Log top features
                    self.log_queue.put("\nTop 10 features:", 'info')
                    for feat in result["feature_importance"]:
                        self.log_queue.put(f"  {feat['feature']}: {feat['importance']:.4f}", 'info')
                    
                    # Log calibration analysis
                    cal = result.get('calibration')
                    if cal:
                        self.log_queue.put("\nCalibration Analysis (test set):", 'info')
                        self.log_queue.put(f"{'Bin':<12} {'Predicted':>9} {'Actual':>9} {'Count':>6}", 'info')
                        self.log_queue.put(f"{'-'*12} {'-'*9} {'-'*9} {'-'*6}", 'info')
                        for row in cal['calibration_table']:
                            self.log_queue.put(
                                f"{row['bin']:<12} {row['predicted_prob']:>9.4f} "
                                f"{row['actual_freq']:>9.4f} {row['count']:>6}", 'info'
                            )
                        self.log_queue.put(
                            f"ECE: {cal['expected_calibration_error']:.4f}  "
                            f"MCE: {cal['max_calibration_error']:.4f}", 'info'
                        )

                    # ── Show delta comparison with previous model
                    if prev:
                        self.log_queue.put("\n── Changes from previous model ──", 'info')
                        # Accuracy delta
                        d_train = result['train_accuracy'] - prev.get('train_accuracy', 0)
                        d_test = result['test_accuracy'] - prev.get('test_accuracy', 0)
                        self.log_queue.put(
                            f"  Train acc: {d_train:+.3f}  Test acc: {d_test:+.3f}",
                            'success' if d_test > 0 else ('error' if d_test < 0 else 'info'),
                        )
                        # ECE/MCE delta
                        prev_cal = prev.get('calibration')
                        if cal and prev_cal:
                            d_ece = cal['expected_calibration_error'] - prev_cal.get('ece', 0)
                            d_mce = cal['max_calibration_error'] - prev_cal.get('mce', 0)
                            self.log_queue.put(
                                f"  ECE: {d_ece:+.4f}  MCE: {d_mce:+.4f}",
                                'success' if d_ece < 0 else ('error' if d_ece > 0 else 'info'),
                            )
                        # Feature importance comparison
                        prev_feats = {f['feature']: f['importance'] for f in prev.get('feature_importance', [])}
                        new_feats = {f['feature']: f['importance'] for f in result['feature_importance']}
                        if prev_feats:
                            self.log_queue.put("  Feature weight changes (top 10):", 'info')
                            for feat in result['feature_importance']:
                                name = feat['feature']
                                new_imp = feat['importance']
                                old_imp = prev_feats.get(name, 0)
                                delta = new_imp - old_imp
                                arrow = '▲' if delta > 0.005 else ('▼' if delta < -0.005 else '─')
                                self.log_queue.put(
                                    f"    {arrow} {name}: {new_imp:.4f} ({delta:+.4f})", 'info'
                                )
                else:
                    self.log_queue.put(f"Error training model: {result['message']}", 'error')
                    
            except Exception as e:
                self.log_queue.put(f"Unexpected error: {e}", 'error')
            finally:
                self.is_training = False
                if self.on_state_changed:
                    self.on_state_changed()
                if callback:
                    callback()
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    def predict_async(self, callback=None):
        """Start asynchronous prediction"""
        if self.is_predicting:
            return
        
        self.is_predicting = True
        if self.on_state_changed:
            self.on_state_changed()
        
        def worker():
            try:
                # ── Step 1: back-fill any previous predictions with actual results
                updated = self.data_manager.update_prediction_history()
                if updated:
                    self.log_queue.put(f"Back-filled {updated} previous prediction(s) with actual results", 'info')

                # ── Step 1b: show performance stats & drift warning
                perf = self.data_manager.get_prediction_performance(min_predictions=5)
                if perf:
                    self.log_queue.put(
                        f"\nHistory: {perf['correct']}/{perf['total']} correct "
                        f"({perf['accuracy']:.1%})",
                        'success' if perf['accuracy'] >= 0.55 else 'warning',
                    )
                    if perf['recent_10'] is not None:
                        self.log_queue.put(f"  Last 10: {perf['recent_10']:.1%}", 'info')
                    if perf['recent_20'] is not None:
                        self.log_queue.put(f"  Last 20: {perf['recent_20']:.1%}", 'info')
                    if perf['by_confidence']:
                        self.log_queue.put("  By confidence:", 'info')
                        for level, stats in perf['by_confidence'].items():
                            self.log_queue.put(
                                f"    {level:<12} {stats['accuracy']:.1%}  (n={int(stats['n'])})",
                                'info',
                            )
                    self.data_manager.detect_model_drift()

                self.log_queue.put("Loading combined data with live ASX200...", 'info')
                combined = self.data_manager.prepare_combined_data_for_prediction()
                
                if combined.empty:
                    self.log_queue.put("No data available. Please update data first.", 'error')
                    return
                
                combined = self.model_manager.engineer_features(combined, for_prediction=True)
                
                if combined.empty:
                    self.log_queue.put("No valid features could be created from the data.", 'error')
                    return
                
                if not self.model_manager.load_model():
                    self.log_queue.put("No trained model found. Please train first.", 'error')
                    return
                
                # Get decision with confidence analysis
                decision = self.model_manager.get_decision(combined, threshold=0.6)
                
                if decision['probability'] is not None:
                    prob = decision['probability']
                    feature_details = decision['feature_details']
                    
                    latest_date = combined.index[-1]
                    latest_date_str = latest_date.strftime('%Y-%m-%d')
                    latest_return = combined['daily_return'].iloc[-1] * 100
                    latest_price = combined['price'].iloc[-1]
                    
                    # Calculate next trading day (skip weekends)
                    from datetime import timedelta
                    next_day = latest_date + timedelta(days=1)
                    while next_day.weekday() >= 5:  # 5=Sat, 6=Sun
                        next_day += timedelta(days=1)
                    prediction_date_str = next_day.strftime('%Y-%m-%d')
                    
                    predicted_up = 1 if prob > 0.5 else 0
                    
                    self.log_queue.put(f"\n{'='*50}", 'info')
                    self.log_queue.put(f"ASX200 latest price: {latest_price:,.2f} ({latest_date_str})", 'info')
                    self.log_queue.put(f"Latest daily return: {latest_return:+.2f}%", 'info')
                    self.log_queue.put(f"Prediction for: {prediction_date_str}", 'info')
                    self.log_queue.put(f"Probability of POSITIVE return: {prob*100:.1f}%", 
                                      'success' if prob>0.5 else 'info')
                    self.log_queue.put(f"Probability of NEGATIVE return: {(1-prob)*100:.1f}%", 
                                      'error' if prob<0.5 else 'info')
                    
                    # Decision & confidence
                    self.log_queue.put(
                        f"Confidence: {decision['confidence_level']}  "
                        f"(threshold={decision['threshold_used']})", 'info'
                    )
                    dec = decision['decision']
                    dec_level = ('success' if 'POSITIVE' in dec
                                 else 'error' if 'NEGATIVE' in dec
                                 else 'info')
                    self.log_queue.put(f"Decision: {dec}", dec_level)
                    self.log_queue.put('='*50, 'info')
                    
                    # Log top feature inputs
                    self.log_queue.put(f"\nTop 15 features by importance:", 'info')
                    self.log_queue.put(f"{'Feature':<30} {'Value':>12} {'Importance':>10}", 'info')
                    self.log_queue.put(f"{'-'*30} {'-'*12} {'-'*10}", 'info')
                    for fd in feature_details[:15]:
                        val = fd['value']
                        imp = fd['importance']
                        # Format value: percentages for returns, otherwise raw
                        if 'return' in fd['name'] or fd['name'] == 'macd_histogram':
                            val_str = f"{val:>+12.4f}"
                        else:
                            val_str = f"{val:>12.4f}"
                        self.log_queue.put(
                            f"{fd['name']:<30} {val_str} {imp:>10.4f}", 'info'
                        )
                    
                    # ── Step 3: save prediction to history CSV
                    model_ver = self.data_manager.get_model_version()
                    market_reg = self.data_manager.classify_market_regime()
                    self.log_queue.put(f"Market regime: {market_reg}  |  Model version: {model_ver}", 'info')

                    self.data_manager.save_prediction_to_history(
                        prediction_date=prediction_date_str,
                        base_date=latest_date_str,
                        base_price=latest_price,
                        probability=prob,
                        predicted_up=predicted_up,
                        signal=dec,
                        confidence_level=decision['confidence_level'],
                        feature_details=feature_details,
                        model_version=model_ver,
                        market_regime=market_reg,
                    )

                    # ── Step 4: save daily performance snapshot
                    try:
                        self.data_manager.save_performance_snapshot()
                    except Exception as e:
                        self.log_queue.put(f"Could not save performance snapshot: {e}", 'warning')
                else:
                    self.log_queue.put("Prediction failed.", 'error')
                    
            except Exception as e:
                self.log_queue.put(f"Error during prediction: {e}", 'error')
            finally:
                self.is_predicting = False
                if self.on_state_changed:
                    self.on_state_changed()
                if callback:
                    callback()
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    # ========== Performance Dashboard ==========

    def get_performance_data(self) -> dict:
        """Gather all performance analytics for the Performance tab.

        Returns dict with keys:
            perf       – dict from get_prediction_performance() or None
            thresholds – DataFrame from analyze_thresholds() or None
            drift      – bool
        """
        perf = self.data_manager.get_prediction_performance(min_predictions=5)
        thresholds = None
        drift = False
        perf_log = None
        if perf:
            try:
                thresholds = self.data_manager.analyze_thresholds()
            except Exception:
                thresholds = None
            try:
                drift = self.data_manager.detect_model_drift()
            except Exception:
                drift = False
        try:
            perf_log = self.data_manager.get_performance_log()
        except Exception:
            perf_log = None
        return {"perf": perf, "thresholds": thresholds, "drift": drift, "perf_log": perf_log}
