"""OPC UA subscription client for the ML pipeline.

Connects to an OPC UA server, subscribes to configured machine signals,
runs each machine's readings through its own Anomaly_model instance
(healthy-vibration prediction + degradation-state classification), and
writes the enriched results to InfluxDB (with an optional debug CSV dump).
Includes automatic reconnect with exponential backoff so a dropped OPC UA
connection doesn't require manually restarting the process.
"""

import json
import os
import signal
from pathlib import Path
import sys
import asyncio
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd

from asyncua import Client
from asyncua.common.subscription import DataChangeEvent
from asyncua.ua.uaerrors import UaError

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        RotatingFileHandler('pipeline.log', maxBytes=5_000_000, backupCount=3),
        logging.StreamHandler()  # keep console output too
    ],force=True
)
_logger = logging.getLogger("asyncua")

from data_pipeline.influxdb_dp import InfluxDB_Client
from ML_model.model import Anomaly_model

OPC_CONFIG = os.path.join(project_root, 'Config', 'opcua_config.json')
ENV_PATH = os.path.join(project_root, 'Config', '.env')

RECONNECT_MIN_DELAY = 2
RECONNECT_MAX_DELAY = 60


class ConfigError(Exception):
    """Raised when the OPC UA / InfluxDB config can't be loaded correctly.
    Kept separate from generic Exception so callers know this is a
    startup problem, not a runtime data problem."""
    pass


class opcua_client:
    """Subscribes to OPC UA machine signals, scores them with per-machine
    ML models, and persists the results to InfluxDB.

    On construction, parses opc_config to build the OPC UA node ID map for
    every configured machine, loads a dedicated Anomaly_model instance per
    machine (so rolling state like RPM history and anomaly counts never
    mixes between machines), and opens the InfluxDB write client. The
    actual subscription loop runs via run_forever(), which wraps
    subscription_client() with reconnect + exponential backoff so a
    dropped OPC UA connection (network blip, controller reboot) doesn't
    require restarting the process.
    """

    def __init__(self, opc_config, env_path: str):
        """Load and validate config, then initialize OPC UA/InfluxDB clients and models.

        Args:
            opc_config: path to the OPC UA JSON config file (endpoint,
                subscription settings, plant/machine/signal definitions,
                and ML model paths).
            env_path: path to the .env file with InfluxDB credentials,
                passed through to InfluxDB_Client.

        Raises:
            ConfigError: if opc_config is missing, contains invalid JSON,
                or is missing a required key.
            Exception: re-raised after logging if the OPC UA Client or
                InfluxDB_Client fails to initialize.
        """

        # --- config parsing -------------------------------------------------
        # Original code logged FileNotFoundError/KeyError as critical but then
        # kept going, so self._url etc. could be undefined and the code below
        # (self._client = Client(url=self._url...)) would blow up with a
        # confusing AttributeError instead of the real root cause. Now we
        # raise immediately so the failure is obvious and happens in one place.
        try:
            self._node_ids = {}
            self._signals = {}
            with open(file=opc_config, encoding='utf-8') as f:
                config_data = json.load(f)
                self._url = config_data['opcua']['endpoint']
                self._keepalive_count = config_data['opcua']['subscription']['keepalive_count']
                self._plant = config_data['plant']['name']
                self._machines = list(config_data['plant']['machines'].keys())
                self._base_node = config_data['opcua']['base_node']

                self._write_csv = config_data['opcua'].get('debug', {}).get('write_csv', False)

                base_node_id = ''
                self._model = {machine: Anomaly_model(window=20, rf_model=str(project_root / config_data['plant']['ml']['vibration_model']), \
                                            state_rf_model=str(project_root / config_data['plant']['ml']['state_classifier_model'])) for machine in self._machines}
                self._env_path = env_path

                for machine in self._machines:
                    base_node_id = self._base_node + f'"{self._plant}".' + f'"{machine}".'
                    signals = config_data['plant']['machines'][machine]['signals']
                    self._signals.update({machine: signals})
                    create_node_ids = lambda x: base_node_id + f'"{x}"'
                    self._node_ids.update({machine: list(map(create_node_ids, signals))})

        except FileNotFoundError as e:
            logging.critical(f'{opc_config} not found. specify correct configuration file ; {e}')
            raise ConfigError(f'{opc_config} not found') from e
        except json.JSONDecodeError as e:
            logging.critical(f'{opc_config} contains invalid JSON ; {e}')
            raise ConfigError(f'{opc_config} invalid JSON') from e
        except KeyError as e:
            logging.critical(f'{opc_config} does not contain valid data : {e}')
            raise ConfigError(f'{opc_config} missing key {e}') from e
        else:
            # only log success on the actual success path, not in finally
            logging.info(f'{opc_config} parsed and data recorded')

        # --- client init ------------------------------------------------------
        try:
            self._client = Client(url=self._url, timeout=self._keepalive_count, auto_reconnect=True)
            self._dbclient = InfluxDB_Client(env_path=self._env_path)
        except Exception as e:
            logging.critical(f'Error while initializing client : {e}')
            raise

        self._stopping = False

    def _create_nodes(self):
        """Resolve configured node ID strings into live asyncua Node objects.

        Must be called after self._client.connect() succeeds (get_node()
        requires an active connection). Populates self._nodes (machine ->
        list of Node) and self._nodes_signals (machine -> {Node: signal
        name}) used to map incoming data-change events back to signal names.

        Raises:
            Exception: re-raised after logging if node resolution fails
                (e.g. a configured node ID doesn't exist on the server).
        """
        try:
            self._nodes = {}
            self._nodes_signals = {}
            for machine in self._machines:
                get_nodes = lambda x: self._client.get_node(x)
                self._nodes[machine] = list(map(get_nodes, self._node_ids[machine]))
                self._nodes_signals.update({machine: dict(zip(self._nodes[machine], self._signals[machine]))})
        except Exception as e:
            logging.critical(f'Error while creating nodes from config : {e}')
            raise

    async def subscription_client(self):
        """Connect once and run the subscription loop. Raises on disconnect /
        failure so run_forever() can decide whether/how to retry."""

        await self._client.connect()
        self._create_nodes()
        nodes = []
        data = {}
        prev_data = {}
        dummy = {}
        async with await self._client.create_subscription(1000) as subscription:
            for machine in self._machines:
                nodes += self._nodes[machine]

            await subscription.subscribe_data_change(nodes)
            _logger.info(f'Subscribed to {len(nodes)} nodes across {len(self._machines)} machines')

            async for event in subscription:
                if self._stopping:
                    _logger.info('Stop requested, exiting subscription loop')
                    break
                match event:
                    case DataChangeEvent(node=node, value=value):
                        for machine in self._machines:
                            if self._nodes_signals[machine].get(node) is not None:
                                _logger.info(f"data change  {machine} {self._nodes_signals[machine].get(node)} : {value}")
                                if machine not in data:
                                    data[machine] = {}
                                    prev_data[machine] = {}
                                    dummy[machine] = {}
                                data[machine].update({self._nodes_signals[machine].get(node): value})
                                if len(data[machine].keys()) == len(self._nodes_signals[machine].keys()):
                                    if data[machine] != prev_data[machine]:
                                        loop = asyncio.get_running_loop()
                                        dummy[machine] = data[machine].copy()
                                        vib_result = await loop.run_in_executor(None, self._model[machine].predict_vibration, data[machine])
                                        dummy[machine].update(vib_result)
                                        state_result = await loop.run_in_executor(None, self._model[machine].predict_state, data[machine])
                                        dummy[machine].update(state_result)
                                        # Each of these can fail independently (DB down,
                                        # disk full, permissions, etc). One failing write
                                        # should not crash the whole subscription and lose
                                        # every other machine's data too.
                                        try:
                                            self._dbclient.create_points(line=self._plant, data={machine: dummy[machine]})
                                            self._dbclient.write_points()
                                        except Exception:
                                            logging.exception(f'InfluxDB write failed for {machine}')

                                        try:

                                            if self._write_csv:
                                                df = pd.DataFrame([dummy[machine]])
                                                df.to_csv(f'{machine}.csv', mode='a', header=not os.path.exists(f'{machine}.csv'), index=False)

                                        except Exception:
                                            logging.exception(f'CSV write failed for {machine}')

                                        prev_data[machine] = data[machine].copy()

    async def run_forever(self):
        """Wraps subscription_client with reconnect + backoff so a dropped
        connection (network blip, controller reboot, etc.) doesn't require
        manually restarting the process. This is the main addition: the
        original script only ran subscription_client once."""
        delay = RECONNECT_MIN_DELAY
        while not self._stopping:
            try:
                await self.subscription_client()
                delay = RECONNECT_MIN_DELAY  # reset backoff after a clean run
            except asyncio.CancelledError:
                raise
            except (UaError, OSError, ConnectionError) as e:
                logging.warning(f'OPC UA connection lost/failed : {e}')
            except Exception:
                logging.exception('Unexpected error in subscription loop')
            finally:
                await self.disconnect_client()

            if self._stopping:
                break

            logging.info(f'Reconnecting in {delay}s')
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def disconnect_client(self):
        """Disconnect the OPC UA client, if connected.

        Safe to call even if the client was never connected or is already
        disconnected — guards against raising and masking whatever error
        triggered the disconnect in the first place.
        """
        # Guard so calling this when the client never connected (or is
        # already disconnected) doesn't raise and mask the original error.
        if not hasattr(self, '_client') or self._client is None:
            return
        try:
            logging.info(f'Disconnecting the client {self._url}')
            await self._client.disconnect()
        except Exception:
            logging.exception('Error while disconnecting client (may already be down)')

    def stop(self):
        """Signal the subscription loop to stop after the current event.

        Sets a flag checked inside subscription_client()'s event loop and
        by run_forever() between reconnect attempts; does not forcibly
        interrupt in-flight work.
        """
        logging.info('Stop requested')
        self._stopping = True


# ====================================================================

async def main() -> None:
    """Client-Subscription example with auto-reconnect.

    Run against examples/server-example.py. Kill and restart the server while
    this script is running: the supervisor reconnects transparently and the
    subscription resumes producing notifications without user intervention.
    """
    # client is defined before the try block so the finally clause below
    # can never hit UnboundLocalError if opcua_client(...) itself raises
    # (e.g. bad config) - previously that would mask the real error.
    client = None
    try:
        client = opcua_client(opc_config=OPC_CONFIG, env_path=ENV_PATH)
    except ConfigError as e:
        logging.critical(f'Startup aborted due to config error : {e}')
        return
    except Exception as e:
        logging.critical(f'Startup aborted : {e}')
        return

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, client.stop)
        except NotImplementedError:
            # add_signal_handler isn't available on Windows event loops;
            # KeyboardInterrupt handling below still covers Ctrl+C there.
            pass

    try:
        await client.run_forever()
    finally:
        await client.disconnect_client()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass