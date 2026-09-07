from collections import deque
import logging
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("model")


class Anomaly_model:
    """Two-stage vibration anomaly / degradation-state model for a single machine.

    Stage 1 (predict_vibration): a Random Forest regressor predicts the
    "healthy" expected vibration for the current RPM/Load, conditioned on a
    rolling RPM mean. The residual between predicted and actual vibration is
    compared against a per-RPM-bin threshold (95th percentile on healthy-only
    data) to flag individual-sample exceedances, and an anomaly is only
    raised once 5 of the last 8 samples exceed threshold (reduces false
    positives from single noisy readings).

    Stage 2 (predict_state): a second Random Forest classifier consumes the
    running anomaly_count (built up in stage 1) to predict a discrete
    degradation state (e.g. 0 = healthy, 1 = degrading, 2 = critical).

    IMPORTANT — call order: predict_state() reads self._df, which is only
    populated by a prior call to predict_vibration() in the same cycle.
    Always call predict_vibration(data) first, then predict_state(), for
    the same data point. Calling predict_state() first will fail because
    self._df is still None.

    One instance of this class holds per-machine state (RPM history,
    exceedance history, running anomaly count) — do NOT share a single
    instance across multiple machines, or their histories will mix.
    """

    def __init__(self, window, rf_model: str, state_rf_model: str):
        """Load both models and initialize per-machine rolling state.

        Args:
            window: number of samples to keep for the rolling RPM mean
                used as a feature by the vibration regressor.
            rf_model: path to the joblib-pickled healthy-vibration
                RandomForestRegressor.
            state_rf_model: path to the joblib-pickled degradation-state
                RandomForestClassifier.
        """

        # ============================================================
        # MODELS
        # ============================================================

        self._rf_model = joblib.load(rf_model)
        self._state_rf_model = joblib.load(state_rf_model)

        # ============================================================
        # WINDOWS
        # ============================================================

        # RPM history for vibration model
        self._window = window
        self._rpm_history = deque(maxlen=self._window)

        # Last 8 exceedance flags, used for the "5 of last 8" anomaly rule.
        # Lives here (not inside predict_vibration) so it persists across calls.
        self._exceedance_history = deque(maxlen=8)

        # 20 samples -> RUL features
        self._rul_window = 20

        # ============================================================
        # RPM BINS
        # ============================================================

        self._bins = [
            (500, 1500),
            (1500, 1650),
            (1650, 1900),
            (1900, 2150),
            (2150, 2670),
            (2670, 3000),
        ]

        # 95th percentile thresholds
        self._bin_thresholds = [
            0.079887,
            0.577672,
            1.593651,
            0.384303,
            0.031468,
            0.105168
        ]

        self._anomaly_count = 0

        # Current dataframe
        self._df = None

    # ================================================================
    # RPM THRESHOLD
    # ================================================================

    def _get_threshold_for_rpm(self, rpm):
        """Return the residual threshold for the RPM bin containing `rpm`.

        Args:
            rpm: current RPM reading.

        Returns:
            The 95th-percentile residual threshold (float) for the matching
            bin, or None if `rpm` falls outside all defined bins.
        """

        for (lo, hi), limit in zip(
            self._bins,
            self._bin_thresholds
        ):

            if lo <= rpm <= hi:
                return limit

        return None

    # ================================================================
    # VIBRATION PREDICTION + ANOMALY DETECTION
    # ================================================================

    def predict_vibration(self, data: dict):
        """Predict expected healthy vibration and flag anomalies.

        Builds a single-row DataFrame from `data`, predicts the expected
        ("healthy") vibration for the current Load/RPM/rolling-RPM-mean,
        computes the residual against the actual vibration reading, and
        checks that residual against the RPM-bin-specific threshold. An
        anomaly is only raised once 5 of the last 8 samples exceed
        threshold. Updates self._df in place — predict_state() depends on
        this having been called first.

        Args:
            data: dict of current sensor readings for one machine. Must
                include 'RPM', 'Load', and 'ActualVibration'.

        Returns:
            dict: the current row (self._df) as a dict, including
                PredictedVibration, Residue, Is_Anomaly, and anomaly_count.
        """

        # ------------------------------------------------------------
        # Create dataframe for current sample
        # ------------------------------------------------------------

        self._df = pd.DataFrame([data])

        # ------------------------------------------------------------
        # RPM history
        # ------------------------------------------------------------

        rpm = float(data['RPM'])

        self._rpm_history.append(rpm)

        rpm_mean = (
            sum(self._rpm_history)
            / len(self._rpm_history)
        )

        self._df['rpm_roll20_mean'] = rpm_mean

        # ------------------------------------------------------------
        # Predict vibration
        # ------------------------------------------------------------

        predicted_vib = self._rf_model.predict(
            self._df[
                [
                    'Load',
                    'RPM',
                    'rpm_roll20_mean'
                ]
            ]
        )[0]

        self._df['PredictedVibration'] = predicted_vib

        # ------------------------------------------------------------
        # Residual
        # ------------------------------------------------------------

        actual_vib = float(
            self._df['ActualVibration'].iloc[0]
        )

        residue = predicted_vib - actual_vib

        self._df['Residue'] = residue

        # ------------------------------------------------------------
        # RPM threshold
        # ------------------------------------------------------------

        try:

            threshold = self._get_threshold_for_rpm(rpm)
            if threshold is not None and threshold > 0:

                # ----------------------------------------------------
                # Exceedance ratio
                # ----------------------------------------------------

                exceedance_ratio = (
                    abs(residue) / threshold
                )

                # ----------------------------------------------------
                # Individual threshold exceedance
                # ----------------------------------------------------
                exceeded = (
                    exceedance_ratio > 1.0
                )
                # ----------------------------------------------------
                # Last 8 samples
                # Used ONLY for anomaly detection
                # ----------------------------------------------------

                self._exceedance_history.append(
                    bool(exceeded)
                )

                # ----------------------------------------------------
                # 5 out of last 8
                # ----------------------------------------------------

                is_anomaly = (
                    len(self._exceedance_history) >= 8
                    and
                    sum(self._exceedance_history) >= 5
                )

                self._df['Is_Anomaly'] = int(
                    is_anomaly
                )

            else:
                self._df['Is_Anomaly'] = 0

            if self._df['Is_Anomaly'].iloc[0] == 1:
                self._anomaly_count += 1

            self._df.loc[self._df.index[0], 'anomaly_count'] = self._anomaly_count

        except Exception as e:

            _logger.critical(f'Error calculating anomaly: {e}')
            self._df['Is_Anomaly'] = 0

        return self._df.to_dict(
            orient='records'
        )[0]

    # ================================================================
    # RUL PREDICTION
    # ================================================================

    def predict_state(self, data: dict):
        """Predict the discrete degradation state from the running anomaly count.

        Reads `anomaly_count` from self._df (set by the most recent
        predict_vibration() call on this instance) and classifies the
        current degradation state. Must be called after predict_vibration()
        in the same cycle, since it depends on self._df already being
        populated.

        Args:
            data: unused. Kept for call-signature symmetry with
                predict_vibration(); all inputs are read from self._df.

        Returns:
            dict: the current row (self._df) as a dict, with
                Pred_DegradeState added (-1 if prediction failed).
        """

        try:
            _features = self._df[['anomaly_count']]
            predicted_state = self._state_rf_model.predict(_features)

            self._df['Pred_DegradeState'] = predicted_state

        except Exception as e:
            _logger.critical(f'Error calculating anomaly: {e}')
            self._df['Pred_DegradeState'] = -1

        return self._df.to_dict(
            orient='records'
        )[0]