import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("influxdb")


class InfluxDB_Client:
    """Thin wrapper around the InfluxDB Python client for writing pipeline data.

    Buffers points built via create_points() in memory and flushes them to
    InfluxDB with write_points(). Credentials and connection settings are
    read from environment variables (loaded from the .env file at
    `env_path`), never from a plaintext config file, so this class should
    never be initialized with hardcoded secrets.

    One instance is shared across all machines in the pipeline — machine
    identity is captured per-point via the "machine" tag in create_points(),
    not via separate client instances.
    """

    def __init__(self, env_path: str):
        """Load InfluxDB connection settings from .env and open the client.

        Args:
            env_path: path to the .env file containing INFLUXDB_URL,
                INFLUXDB_BUCKET, INFLUXDB_ORG, and INFLUXDB_TOKEN.

        Raises:
            Exception: re-raised after logging if the client cannot be
                initialized (bad URL, auth failure, missing env vars, etc.).
        """

        try:
            _logger.info("Reading InfluxDB configuration")

            load_dotenv(dotenv_path=env_path,override=True)

            self._url = os.getenv('INFLUXDB_URL')
            self._bucket = os.getenv('INFLUXDB_BUCKET')
            self._org = os.getenv('INFLUXDB_ORG')
            self._token = os.getenv('INFLUXDB_TOKEN')



            self._client = InfluxDBClient(
                url=self._url,
                token=self._token,
                org=self._org
            )

            # Create Write API only once
            self._write_api = self._client.write_api(
                write_options=SYNCHRONOUS
            )

            self._points = []

            _logger.info("Connected to InfluxDB")

        except Exception as e:
            _logger.exception(f"Failed to initialize InfluxDB client: {e}")
            raise

    def create_points(self, line, data):
        """Build InfluxDB Points from a dict of per-machine readings and buffer them.

        Does not write to InfluxDB — points are appended to self._points and
        only sent on the next write_points() call, so multiple machines'
        readings can be batched into one write.

        Args:
            line: production line name, written as a tag on every point
                (e.g. "Line1").
            data: dict keyed by machine name, each value a flat dict of
                field_name -> value for that machine, e.g.:
                {
                    "Motor_001": {
                        "RPM": 100,
                        "Load": 45
                    },
                    "Motor_002": {
                        "RPM": 120,
                        "Temp": 55
                    }
                }
        """

        for machine, values in data.items():

            point = (
                Point("machine_data")
                .tag("line", line)
                .tag("machine", machine)
                .time(datetime.now(ZoneInfo("Asia/Kolkata")), WritePrecision.NS)
            )

            for field, value in values.items():
                point.field(field, value)

            self._points.append(point)

    def write_points(self):
        """Flush all buffered points to InfluxDB and clear the buffer.

        No-op if there are no buffered points. Errors are logged (with
        traceback) rather than raised, so a failed write doesn't crash the
        caller — but note the buffer is only cleared on success, so a failed
        write's points are silently dropped rather than retried on the next
        call (they remain in self._points only if the exception occurs
        before .clear(), which it does not currently retry).
        """

        if not self._points:
            return

        try:
            _logger.info(f"Writing {len(self._points)} point(s) to InfluxDB")

            self._write_api.write(
                bucket=self._bucket,
                record=self._points
            )

            # Clear buffer after successful write
            self._points.clear()

        except Exception as e:
            _logger.exception(f"Error writing to InfluxDB: {e}")

    def close(self):
        """Close the underlying InfluxDB client connection.

        Should be called once during graceful shutdown (e.g. from the
        caller's disconnect routine) to release the connection cleanly.
        """ 
        _logger.info("Closing InfluxDB client")
        self._client.close()