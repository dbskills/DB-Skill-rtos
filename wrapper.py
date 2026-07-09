#!/usr/bin/env python3
"""RTOS Skill — single-file long-lived HTTP server wrapper.

Reinforcement Learning with Tree-LSTM for Join Order Selection. Exposes the
RTOS DQN+TreeLSTM join-order selector per the query_optimize skill I/O spec.
State (model weights, replay buffer, alias map, counters) lives in memory for
the lifetime of the server process and is persisted to disk on training events
and on graceful shutdown so a container restart can rehydrate it.

Endpoints: GET /health, POST /optimize, GET /state, POST /shutdown.
"""

import argparse
import contextlib
import copy
import importlib
import io
import json
import os
import pickle
import queue as queue_mod
import signal
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INITIAL_MODEL = "saved_model/job_cost_trained.pth"
JOB_ALIAS_MAP_FILE = "job_alias_map.json"
TRAIN_THRESHOLD_DEFAULT = 10
TRAIN_STEPS_PER_TRIGGER = 4
TARGET_UPDATE_EVERY = 3
# DQN optimize_model() samples a batch of this many transitions. Used to scale
# the gradient steps per --train spawn to the replay buffer size (the fixed 4
# barely trained on a large buffer).
TRAIN_BATCH_SIZE = 128
# Cap on gradient steps per --train spawn so one training run can't run
# unbounded as the buffer grows; TRAIN_STEPS_PER_TRIGGER is the floor.
MAX_TRAIN_STEPS_PER_RUN = 200
# Safe-gate for background training: if the RTOS-chosen plan's cost is more
# than this many times the DP baseline cost, discard it without running
# EXPLAIN ANALYZE / training. The cold-start model occasionally picks a
# catastrophically bad join order (cost tens of millions vs. hundreds of
# thousands); executing it would hang the single background thread for up to
# its statement_timeout, blocking all subsequent training. The cost ratio is
# a cheap foreground-visible proxy for "will this plan take forever".
_MAX_COST_RATIO = 50.0


def _cost_ratio_too_high(chosen_cost, default_cost, max_ratio=_MAX_COST_RATIO):
    """Return True if the chosen plan's cost is catastrophically higher than
    the default plan's (so training should be skipped for this query).
    Unknown / non-positive costs → False (let the background attempt it)."""
    if not chosen_cost or not default_cost:
        return False
    try:
        d = float(default_cost)
        if d <= 0:
            return False
        return float(chosen_cost) / d > max_ratio
    except (TypeError, ValueError, ZeroDivisionError):
        return False


# --------------------------------------------------------------------------- #
# Read-write lock: shared/read for MCTS rollout (parallel), exclusive/write
# for alias-map extension, Memory push, and policy_net reload.
# --------------------------------------------------------------------------- #
class RWLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read(self):
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


def _parse_dsn(dsn):
    import psycopg2.extensions
    parsed = psycopg2.extensions.parse_dsn(dsn)
    return (parsed.get("dbname") or parsed.get("database") or "",
            parsed.get("user") or "", parsed.get("password") or "",
            parsed.get("host") or "127.0.0.1", parsed.get("port") or 5432)


def _fetch_schema_sql(dbname, user, password, host, port):
    import psycopg2
    conn = psycopg2.connect(database=dbname, user=user, password=password,
                            host=host, port=port)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.relname, a.attname, a.atttypid::regtype::text
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid
        WHERE c.relkind = 'r'
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY c.relname, a.attnum
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    tables = {}
    for relname, attname, typname in rows:
        tables.setdefault(relname, []).append((attname, typname))
    parts = []
    for tname, cols in tables.items():
        coldefs = ",\n    ".join(f"{c[0]} {c[1]}" for c in cols)
        parts.append(f"CREATE TABLE {tname} (\n    {coldefs}\n);")
    return "\n\n".join(parts)


def _patch_config(dbname, user, password, host, port):
    """Set up a shared ImportantConfig.config instance with the DSN params, and
    make Config() return it, BEFORE source modules are imported (they call
    Config() at import time). DSN changes mutate this instance directly — no
    module reload needed."""
    import ImportantConfig as _IC
    cfg = _IC.Config()
    cfg.dbName = dbname
    cfg.userName = user
    cfg.password = password
    cfg.ip = host
    cfg.port = port
    cfg.usegpu = False
    cfg.use_hint = True
    # Foreground (request critical path) is ALWAYS cost-only: the module-level
    # pgrunner (PGUtils.py:243) is constructed once from config.isCostTraining,
    # so getLatency -> getCost -> EXPLAIN (no ANALYSE, no execution) in BOTH
    # optimize-only and online-training modes. Actual EXPLAIN ANALYZE for
    # training-data collection runs in the background via a separate connection
    # (see _explain_analyze_latency), never on the critical path.
    cfg.isCostTraining = True
    cfg.latencyRecord = False
    cfg.schemaFile = ""
    _IC.config = cfg
    _IC.Config = lambda: cfg
    return cfg


class _ThreadLocalPGGRunner:
    """Thread-safe proxy over the source ``PGGRunner`` singleton.

    The source ``PGGRunner`` (PGUtils.py) holds a single ``self.con`` /
    ``self.cur`` (one psycopg2 connection, one cursor) and the module-level
    ``pgrunner`` is one shared instance. ``sqlSample``/``DQN`` bind it at
    import time (``from PGUtils import pgrunner``) and every cost-only
    ``EXPLAIN`` in the MCTS rollout (``getCost``/``getLatency``/``getPlan``/
    ``getSelectivity``) goes through that one cursor. Under concurrent
    requests, two threads interleaving ``cur.execute``/``cur.fetchall`` on
    the same cursor raises ``no results to fetch`` (and worse) — this was
    the 4-way-parallel failure.

    This proxy is installed as ``PGUtils.pgrunner`` *before* ``sqlSample``
    and ``DQN`` are imported, so their ``from PGUtils import pgrunner``
    binds the proxy. Each request thread gets its own ``PGGRunner`` (own
    connection + cursor) lazily, cached in ``threading.local()``. A DSN
    change (``_init_for_dsn`` mutates the shared ``ImportantConfig.config``
    in place) is detected via the config key; the per-thread instance is
    rebuilt on the next call so threads pick up the new DSN. The background
    thread uses a separate ``_bg_conn`` and never touches this proxy.
    """

    def __init__(self, PGGRunner_cls, config):
        self._PGGRunner = PGGRunner_cls
        self._config = config
        self._tlocal = threading.local()

    def _dsn_key(self):
        cfg = self._config
        return (cfg.dbName, cfg.userName, cfg.ip, cfg.port)

    def _instance(self):
        key = self._dsn_key()
        inst = getattr(self._tlocal, "inst", None)
        if inst is None or getattr(self._tlocal, "key", None) != key:
            cfg = self._config
            # latencyRecord is False (set in _patch_config) so no file
            # handle is opened per instance; isCostTraining=True keeps the
            # foreground cost-only (EXPLAIN, no ANALYZE).
            inst = self._PGGRunner(
                cfg.dbName, cfg.userName, cfg.password, cfg.ip, cfg.port,
                isCostTraining=cfg.isCostTraining,
                latencyRecord=cfg.latencyRecord,
                latencyRecordFile=cfg.latencyRecordFile)
            self._tlocal.inst = inst
            self._tlocal.key = key
        return inst

    def __getattr__(self, name):
        # Dispatch any PGGRunner method (getCost/getLatency/getPlan/...) to
        # the per-thread instance. Only called for attrs not on the proxy
        # itself (_factory/_tlocal/_config/_dsn_key/_instance are found via
        # normal lookup, so they don't recurse).
        return getattr(self._instance(), name)


def _install_thread_local_pgrunner(PGUtils):
    """Replace the module-level ``pgrunner`` (a single shared PGGRunner with
    one cursor) with a per-thread proxy. Must run after ``import PGUtils``
    but before ``sqlSample``/``DQN`` import so their ``from PGUtils import
    pgrunner`` binds the proxy. Closes the orphaned line-243 instance's
    connection (one wasted conn otherwise)."""
    from ImportantConfig import config as cfg
    orphan = PGUtils.pgrunner
    try:
        if orphan is not None and hasattr(orphan, "con") and orphan.con:
            orphan.con.close()
    except Exception:
        pass
    PGUtils.pgrunner = _ThreadLocalPGGRunner(PGUtils.PGGRunner, cfg)
    return PGUtils.pgrunner


def _load_modules():
    with contextlib.redirect_stdout(io.StringIO()):
        import PGUtils
        _install_thread_local_pgrunner(PGUtils)
        import JOBParser
        import TreeLSTM
        import sqlSample
        import DQN
    return PGUtils, JOBParser, TreeLSTM, sqlSample, DQN


def _reload_modules(modules):
    with contextlib.redirect_stdout(io.StringIO()):
        for m in modules:
            importlib.reload(m)


def _state_paths(state_dir):
    if not os.path.isabs(state_dir):
        state_dir = os.path.join(REPO_DIR, state_dir)
    os.makedirs(state_dir, exist_ok=True)
    return {
        "dir": state_dir,
        "alias_map": os.path.join(state_dir, "alias_map.json"),
        "model": os.path.join(state_dir, "model.pth"),
        "replay": os.path.join(state_dir, "replay_buffer.pkl"),
        "counters": os.path.join(state_dir, "counters.json"),
        "training_result": os.path.join(state_dir, "training_result.json"),
    }


def _atomic_write_bytes(path, data):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    _atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


def _load_alias_map(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    job_map_path = os.path.join(REPO_DIR, JOB_ALIAS_MAP_FILE)
    if os.path.exists(job_map_path):
        with open(job_map_path) as f:
            return json.load(f)
    return {}


def _load_counters(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"samples_seen": 0, "training_steps": 0, "triggers_since_target_sync": 0}


def _load_replay_buffer(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_replay_buffer(path, mem):
    _atomic_write_bytes(path, pickle.dumps(mem))


def _save_model(path, state_dict):
    import torch
    tmp = "%s.tmp.%d" % (path, os.getpid())
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def _merge_alias_map(reloaded, ours):
    merged = dict(reloaded)
    used = set(merged.values())
    next_id = 1
    for alias in sorted(ours.keys()):
        if alias in merged:
            continue
        while next_id <= 39 and next_id in used:
            next_id += 1
        if next_id <= 39:
            merged[alias] = next_id
            used.add(next_id)
            next_id += 1
        else:
            merged[alias] = 0
    return merged


def _build_dqn(dbname, user, password, host, port, model_path, sqlSample, DQN,
               TreeLSTM, JOBParser, PGUtils, replay_buffer=None):
    import torch
    schema_sql = _fetch_schema_sql(dbname, user, password, host, port)
    db_info = JOBParser.DB(schema_sql)

    featureSize = 128
    mask_size = 1640
    device = torch.device("cpu")

    policy_net = TreeLSTM.SPINN(n_classes=1, size=featureSize, n_words=100,
                                mask_size=mask_size, device=device).to(device)
    target_net = TreeLSTM.SPINN(n_classes=1, size=featureSize, n_words=100,
                                mask_size=mask_size, device=device).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(state_dict)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    dqn = DQN.DQN(policy_net, target_net, db_info, PGUtils.pgrunner, device)
    if replay_buffer is not None:
        dqn.Memory.memory = replay_buffer.get("memory", [])
        dqn.Memory.position = replay_buffer.get("position", 0)
        dqn.Memory.bestJoinTreeValue = replay_buffer.get("bestJoinTreeValue", {})
        if len(dqn.Memory.memory) < dqn.Memory.capacity:
            dqn.Memory.position = len(dqn.Memory.memory)
    return dqn, db_info, device


def _build_env(query, db_info, PGUtils, sqlSample, DQN, device):
    with contextlib.redirect_stdout(io.StringIO()):
        sql_sample = sqlSample.sqlInfo(PGUtils.pgrunner, query, "wrapper_input")
        env = DQN.ENV(sql_sample, db_info, PGUtils.pgrunner, device)
    return env


def _run_inference(dqn, env, DQN, collect_training_signal=False):
    from itertools import count
    import torch

    if not hasattr(env.sel, "join_candidate") or not env.sel.from_table_list \
            or len(env.sel.from_table_list) < 2:
        orig = env.sql.sql
        try:
            dp_cost = env.sql.getDPCost()
        except Exception:
            dp_cost = None
        return orig, dp_cost, dp_cost, 0, []

    optimized_sql = None
    rtos_cost = None
    dp_cost = None
    num_steps = 0
    snapshots = []

    with contextlib.redirect_stdout(io.StringIO()):
        for _ in count():
            action_list, chosen_action, all_action = dqn.select_action(env, need_random=False)
            if chosen_action is None:
                break
            if collect_training_signal:
                value_now = env.selectValue(dqn.policy_net)
                next_value = torch.min(action_list).detach()
                env_snapshot = copy.deepcopy(env)
                snapshots.append((value_now, next_value.view(-1, 1), env_snapshot))
            left, right = chosen_action[0], chosen_action[1]
            env.takeAction(left, right)
            num_steps += 1
            reward, done, rtos_cost, sql = env.reward_new("wrapper_input")
            if done:
                optimized_sql = sql
                dp_cost = env.sql.DPLantency if env.sql.DPLantency is not None else None
                break

    if optimized_sql is None:
        optimized_sql = env.sql.sql
    return optimized_sql, dp_cost, rtos_cost, num_steps, snapshots


def _push_training_signal(dqn, env, snapshots, rtos_cost, dp_cost, DQN):
    from math import log
    import torch
    config = DQN.config
    if not dp_cost or not rtos_cost or dp_cost <= 0:
        return
    reward = rtos_cost / dp_cost
    try:
        reward = log(reward + 1)
    except Exception:
        pass
    if reward > config.maxR:
        reward = config.maxR
    reward_t = torch.tensor([reward], dtype=torch.float32).view(-1, 1)
    next_value = torch.tensor([0.0], dtype=torch.float32).view(-1, 1)
    expected = next_value + reward_t.detach()
    final = next_value + reward_t.detach()
    dqn.Memory.push(env, expected, final)
    for value_now, next_val, env_snap in snapshots[:0:-1]:
        if expected > next_val:
            expected = next_val
        dqn.Memory.push(env_snap, expected, final)


def _compute_estimated_impact(dp_cost, rtos_cost):
    if not dp_cost or not rtos_cost or dp_cost <= 0:
        return 0.0
    return max(0.0, (1.0 - rtos_cost / dp_cost) * 100.0)


# --------------------------------------------------------------------------- #
# Training subprocess: reconstruct a DQN for the DSN, load Memory from disk,
# run optimize_model, save atomically. Runs in its own process (separate GIL)
# so it never blocks the server's request threads.
# --------------------------------------------------------------------------- #
def run_training(state_dir, dsn):
    paths = _state_paths(state_dir)
    dbname, user, password, host, port = _parse_dsn(dsn)
    _patch_config(dbname, user, password, host, port)
    PGUtils, JOBParser, TreeLSTM, sqlSample, DQN = _load_modules()

    alias_map = _load_alias_map(paths["alias_map"])
    sqlSample.set_persisted_alias_map(alias_map)
    replay_buffer = _load_replay_buffer(paths["replay"])

    if os.path.exists(paths["model"]):
        model_path = paths["model"]
    else:
        model_path = DEFAULT_INITIAL_MODEL
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            model_path = os.path.join(REPO_DIR, model_path)

    dqn, _, device = _build_dqn(dbname, user, password, host, port, model_path,
                                 sqlSample, DQN, TreeLSTM, JOBParser, PGUtils, replay_buffer)
    counters = _load_counters(paths["counters"])

    # Scale gradient steps to the replay buffer size (floor TRAIN_STEPS_PER_TRIGGER,
    # cap MAX_TRAIN_STEPS_PER_RUN) so a large buffer is actually trained on
    # rather than getting a fixed 4 steps per spawn.
    buffer_len = 0
    try:
        buffer_len = len(dqn.Memory.memory)
    except Exception:
        pass
    steps = max(TRAIN_STEPS_PER_TRIGGER,
                min(buffer_len // TRAIN_BATCH_SIZE if TRAIN_BATCH_SIZE else 0,
                    MAX_TRAIN_STEPS_PER_RUN))
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            dqn.optimize_model()
    triggers = counters.get("triggers_since_target_sync", 0) + 1
    if triggers >= TARGET_UPDATE_EVERY:
        dqn.target_net.load_state_dict(dqn.policy_net.state_dict())
        triggers = 0

    # Atomic save: temp file + os.replace.
    _save_model(paths["model"], dqn.policy_net.state_dict())
    _save_replay_buffer(paths["replay"], {
        "memory": dqn.Memory.memory,
        "position": dqn.Memory.position,
        "bestJoinTreeValue": dqn.Memory.bestJoinTreeValue,
    })
    # Write the advanced training_steps/triggers to a result file for the
    # foreground to apply. The foreground is the SOLE counters.json writer
    # (per-collect persist), so this avoids the cross-process clobber where
    # the foreground's stale in-memory training_steps would overwrite this
    # increment. See RTOSSkill._apply_training_result.
    _atomic_write_json(paths["training_result"], {
        "training_steps": counters.get("training_steps", 0) + steps,
        "triggers_since_target_sync": triggers,
    })
    return 0


# --------------------------------------------------------------------------- #
# Skill: holds all in-memory state; one instance per server process.
# --------------------------------------------------------------------------- #
class RTOSSkill:
    def __init__(self):
        # _model_lock: RW for policy_net. MCTS rollout = read (parallel);
        # background reload after training = write (brief, rare).
        self._model_lock = RWLock()
        # _state_lock: brief mutex for Memory/counters/alias_map (push, env-build,
        # persist snapshot). Never held during disk I/O or MCTS.
        self._state_lock = threading.Lock()
        self.loaded = False
        self.dsn = None
        self.dsn_key = None
        self.state_dir = "state"
        self.initial_model = DEFAULT_INITIAL_MODEL
        self.train_threshold = TRAIN_THRESHOLD_DEFAULT
        self.paths = None
        self.PGUtils = self.JOBParser = self.TreeLSTM = self.sqlSample = self.DQN = None
        self._config_instance = None
        self.dqn = None
        self.db_info = None
        self.device = None
        self.alias_map = {}
        self.counters = {"samples_seen": 0, "training_steps": 0, "triggers_since_target_sync": 0}
        self._model_mtime = 0
        self._training_in_progress = False
        self._train_proc = None  # Popen of the in-flight training subprocess
        self._shutting_down = False  # set by persist(); /optimize then 503s
        self._persist_lock = threading.Lock()  # single-flight: concurrent /shutdown runs persist once
        # Dedicated background connection for EXPLAIN ANALYZE (latency
        # collection). Single background thread → one cached conn per DSN,
        # separate from the foreground pgrunner so background execution never
        # blocks/interferes with foreground cost-only EXPLAIN.
        self._bg_conn = None
        self._bg_conn_dsn = None
        self._bg_queue = queue_mod.Queue()
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def _apply_config(self, config):
        self.state_dir = config.get("state_dir") or self.state_dir
        self.initial_model = config.get("initial_model") or DEFAULT_INITIAL_MODEL
        self.train_threshold = int(config.get("train_trigger", TRAIN_THRESHOLD_DEFAULT))
        self.paths = _state_paths(self.state_dir)

    def _init_for_dsn(self, dsn):
        dbname, user, password, host, port = _parse_dsn(dsn)
        if not self.loaded:
            # First load: patch config + import source modules (one time).
            self._config_instance = _patch_config(dbname, user, password, host, port)
            self.PGUtils, self.JOBParser, self.TreeLSTM, self.sqlSample, self.DQN = _load_modules()
        else:
            # DSN change: mutate the shared config instance in place. The
            # per-thread pgrunner proxy (see _ThreadLocalPGGRunner) detects
            # the new DSN key and rebuilds each thread's PGGRunner lazily on
            # the next call, so no explicit pgrunner swap is needed here.
            cfg = self._config_instance
            cfg.dbName, cfg.userName, cfg.password = dbname, user, password
            cfg.ip, cfg.port = host, port
            # Drop the cached background connection (different DSN).
            if self._bg_conn is not None:
                try:
                    self._bg_conn.close()
                except Exception:
                    pass
                self._bg_conn = None
                self._bg_conn_dsn = None

        self.alias_map = _load_alias_map(self.paths["alias_map"])
        self.sqlSample.set_persisted_alias_map(self.alias_map)
        replay_buffer = _load_replay_buffer(self.paths["replay"])

        if os.path.exists(self.paths["model"]):
            model_path = self.paths["model"]
        else:
            model_path = self.initial_model
            if not os.path.isabs(model_path) and not os.path.exists(model_path):
                model_path = os.path.join(REPO_DIR, model_path)

        self.dqn, self.db_info, self.device = _build_dqn(
            dbname, user, password, host, port, model_path,
            self.sqlSample, self.DQN, self.TreeLSTM, self.JOBParser,
            self.PGUtils, replay_buffer,
        )
        self.counters = _load_counters(self.paths["counters"])
        self._apply_training_result()  # apply any orphaned result from a crashed prior run
        try:
            self._model_mtime = os.path.getmtime(self.paths["model"])
        except OSError:
            self._model_mtime = 0
        self.dsn = dsn
        self.dsn_key = (dbname, user, host, port)
        self.loaded = True

    def ensure_loaded(self, config, dsn):
        self._apply_config(config)
        dbname, user, password, host, port = _parse_dsn(dsn)
        key = (dbname, user, host, port)
        if not self.loaded or key != self.dsn_key:
            with self._model_lock.write():
                if not self.loaded or key != self.dsn_key:
                    self._init_for_dsn(dsn)

    def optimize(self, dsn, query, optimize_only):
        # No model reload on the critical path; the background thread reloads
        # after a training subprocess finishes.
        start = time.time()
        # NOTE: isCostTraining is fixed True at config time (see _patch_config),
        # so the foreground pgrunner only does cost-only EXPLAIN here — never
        # EXPLAIN ANALYZE. The optimize_only flag now only gates whether
        # training-data collection is enqueued for the background thread.

        # env-build mutates the shared alias map → brief state_lock.
        with self._state_lock:
            env = _build_env(query, self.db_info, self.PGUtils,
                             self.sqlSample, self.DQN, self.device)

        # MCTS rollout reads policy_net → model_lock read (parallel). Cost-only
        # EXPLAIN (no execution) is used for plan costs in both modes.
        collect = (not optimize_only)
        with self._model_lock.read():
            optimized_sql, dp_cost, rtos_cost, num_steps, snapshots = _run_inference(
                self.dqn, env, self.DQN, collect_training_signal=collect)

        elapsed = time.time() - start
        estimated_impact = _compute_estimated_impact(dp_cost, rtos_cost)
        num_tables = len(env.sel.from_table_list) if hasattr(env, "sel") else 0

        if collect and optimized_sql and not _cost_ratio_too_high(rtos_cost, dp_cost):
            # Push the env + snapshots + the two SQLs to the background, which
            # runs EXPLAIN ANALYZE on a separate connection to collect the
            # latency-based reward. The response returns immediately. (If the
            # RTOS plan is catastrophically costlier than the DP baseline, the
            # safe-gate skips enqueueing — see _cost_ratio_too_high — so the
            # background thread isn't blocked for minutes on a doomed plan.)
            self._bg_queue.put(("collect", env, snapshots, optimized_sql, query, dsn))

        return {
            "optimized_query": optimized_sql,
            "metadata": {
                "strategy_type": "rl-tree-lstm-join-order",
                "optimization_time": round(elapsed, 4),
                "estimated_impact": round(estimated_impact, 4),
                "num_tables": num_tables,
                "num_join_steps": num_steps,
                "mode": "optimize-only" if optimize_only else "online-training",
                # Cost estimates from cost-only EXPLAIN (foreground never
                # executes). Actual latencies are collected in the background.
                "dp_cost": dp_cost,
                "rtos_cost": rtos_cost,
                "trained_this_run": False,
            },
        }

    def _bg_loop(self):
        while True:
            task = self._bg_queue.get()
            try:
                self._collect_training_data(task)
            except Exception as e:
                print(f"bg training-data collection failed: {e}", file=sys.stderr)

    def _get_bg_conn(self, dsn):
        """One cached background connection per DSN (the background is
        single-threaded). Separate from the foreground pgrunner so background
        EXPLAIN ANALYZE never interferes with foreground cost-only EXPLAIN."""
        import psycopg2
        if self._bg_conn is not None and self._bg_conn_dsn == dsn:
            try:
                cur = self._bg_conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
                return self._bg_conn
            except Exception:
                try:
                    self._bg_conn.close()
                except Exception:
                    pass
                self._bg_conn = None
        dbname, user, password, host, port = _parse_dsn(dsn)
        self._bg_conn = psycopg2.connect(
            database=dbname, user=user, password=password, host=host, port=port)
        self._bg_conn_dsn = dsn
        self._bg_conn.autocommit = True
        return self._bg_conn

    def _explain_analyze_latency(self, dsn, sql_text):
        """Run EXPLAIN ANALYZE on a separate background connection and return
        the Actual Total Time (latency, ms) under the DB's actual default
        settings — no parallelism/join-collapse/GEQO overrides (those distorted
        real latency and disabled parallelism, making training-data collection
        needlessly slow). The Leading hint in hinted SQL is honored via the
        global pg_hint_plan.enable_hint (the source relies on it, not a
        per-session SET). A 300s statement_timeout caps runaway plans. The
        +5ms offset matches the source getLatency post-processing so the
        reward scale aligns with the pre-trained model."""
        from psycopg2 import sql as pgsql
        try:
            conn = self._get_bg_conn(dsn)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 300000;")
            # Use psycopg2.sql.SQL so '%' in LIKE predicates isn't interpreted
            # as a placeholder.
            cur.execute(pgsql.SQL("EXPLAIN (COSTS, FORMAT JSON, ANALYSE) {}").format(
                pgsql.SQL(sql_text)))
            rows = cur.fetchall()
            conn.rollback()
            return float(rows[0][0][0]["Plan"]["Actual Total Time"]) + 5.0
        except Exception as e:
            print(f"bg explain analyze failed: {e}", file=sys.stderr)
            try:
                if self._bg_conn is not None:
                    self._bg_conn.rollback()
            except Exception:
                pass
            return None

    def _collect_training_data(self, task):
        _, env, snapshots, optimized_sql, original_query, dsn = task
        spawn = False
        # Background EXPLAIN ANALYZE on a separate connection — the foreground
        # already returned with a cost-only result. This is the ONLY place the
        # rtos skill executes queries.
        rtos_latency = self._explain_analyze_latency(dsn, optimized_sql)
        dp_latency = self._explain_analyze_latency(dsn, original_query)
        if not rtos_latency or not dp_latency or dp_latency <= 0:
            return
        # Brief state_lock: push to Memory + snapshot state for persist.
        # Never held during EXPLAIN ANALYZE or disk I/O → doesn't block
        # foreground MCTS.
        with self._state_lock:
            _push_training_signal(self.dqn, env, snapshots, rtos_latency, dp_latency, self.DQN)
            self.counters["samples_seen"] = self.counters.get("samples_seen", 0) + 1
            self.alias_map = self.sqlSample.get_persisted_alias_map()
            untrained = self.counters["samples_seen"] - self.counters.get("last_train_samples", 0)
            if untrained >= self.train_threshold and not self._training_in_progress:
                self._training_in_progress = True
                self.counters["last_train_samples"] = self.counters["samples_seen"]
                spawn = True
            # Snapshot for unlocked disk write.
            mem_snapshot = {
                "memory": list(self.dqn.Memory.memory),
                "position": self.dqn.Memory.position,
                "bestJoinTreeValue": dict(self.dqn.Memory.bestJoinTreeValue),
            }
            counters_snapshot = dict(self.counters)
            alias_snapshot = dict(self.alias_map)
        # Persist to disk WITHOUT the lock (atomic writes; foreground unaffected).
        _save_replay_buffer(self.paths["replay"], mem_snapshot)
        _atomic_write_json(self.paths["counters"], counters_snapshot)
        _atomic_write_json(self.paths["alias_map"], alias_snapshot)
        if spawn:
            self._spawn_training_worker()

    def _spawn_training_worker(self):
        state_dir = self.paths["dir"]
        dsn = self.dsn

        def _run():
            try:
                p = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__),
                     "--train", "--state-dir", state_dir, "--dsn", dsn]
                )
                with self._state_lock:
                    self._train_proc = p
                p.wait()
                # After training completes, reload policy_net in the background
                # (brief model_lock write; never on the request critical path).
                self._reload_model_from_disk()
                # Apply the --train subprocess's training_steps/triggers so
                # /state reports them and the per-collect persist doesn't
                # clobber the increment with a stale in-memory value.
                self._apply_training_result()
            except Exception as e:
                print(f"training worker failed: {e}", file=sys.stderr)
            finally:
                with self._state_lock:
                    self._training_in_progress = False
                    self._train_proc = None

        threading.Thread(target=_run, daemon=True).start()

    def _wait_for_training(self, timeout=300):
        """Block until the in-flight training subprocess finishes (its model
        save completes) or the timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                in_progress = self._training_in_progress
            if not in_progress:
                return
            time.sleep(0.5)

    def persist(self):
        if not self._persist_lock.acquire(blocking=False):
            # Another persist() is in flight (concurrent /shutdown or SIGTERM
            # racing a skill_runner stop_wrapper); it will complete the save.
            # Returning here avoids two in-process run_training calls
            # clobbering the same pid-based temp dir.
            return
        try:
                # Don't bypass the model save on shutdown. The Memory/counters/alias
                # map are in-memory; snapshot+save them first. Then wait for any
                # in-flight training subprocess so its model save isn't abandoned, and
                # if samples were collected since the last training, run a final
                # in-process training so the persisted model reflects all data. (The
                # server's in-memory policy_net is a load of the on-disk model, so we
                # only _save_model directly when no final training ran — otherwise
                # run_training is the canonical saver.)
                self._shutting_down = True
                with self._state_lock:
                    if not (self.loaded and self.paths is not None):
                        return
                    mem_snapshot = {
                        "memory": list(self.dqn.Memory.memory),
                        "position": self.dqn.Memory.position,
                        "bestJoinTreeValue": dict(self.dqn.Memory.bestJoinTreeValue),
                    }
                    counters_snapshot = dict(self.counters)
                    alias_snapshot = dict(self.alias_map)
                    untrained = (self.counters.get("samples_seen", 0)
                                 - self.counters.get("last_train_samples", 0))
                _save_replay_buffer(self.paths["replay"], mem_snapshot)
                _atomic_write_json(self.paths["counters"], counters_snapshot)
                _atomic_write_json(self.paths["alias_map"], alias_snapshot)
                self._wait_for_training(timeout=300)
                if untrained > 0 and self.dsn:
                    try:
                        run_training(self.paths["dir"], self.dsn)
                        self._apply_training_result()
                    except Exception as e:
                        print(f"final training on shutdown failed: {e}", file=sys.stderr)
                else:
                    # No new training data; flush the in-memory policy_net (a re-save
                    # of the on-disk model, harmless, but keeps the file fresh).
                    _save_model(self.paths["model"], self.dqn.policy_net.state_dict())
        finally:
            self._persist_lock.release()
            # Only the persist that acquired the lock shuts down the
            # server — a guarded-out persist (concurrent /shutdown)
            # must NOT start SERVER.shutdown, or daemon_threads=True
            # would let main() exit and kill this in-flight save.
            if SERVER is not None:
                threading.Thread(target=SERVER.shutdown, daemon=True).start()

    def _reload_model_from_disk(self):
        with self._model_lock.write():
            try:
                mtime = os.path.getmtime(self.paths["model"])
            except OSError:
                return
            if mtime == self._model_mtime:
                return
            try:
                import torch
                state_dict = torch.load(self.paths["model"], map_location=self.device)
                self.dqn.policy_net.load_state_dict(state_dict)
                self.dqn.target_net.load_state_dict(state_dict)
                self.dqn.target_net.eval()
                self._model_mtime = mtime
            except Exception as e:
                print(f"model reload failed: {e}", file=sys.stderr)

    def _apply_training_result(self):
        """Apply the --train subprocess's advanced training_steps/triggers
        (written to training_result.json) into the in-memory counters and
        persist counters.json. The foreground is the SOLE counters.json writer
        (per-collect persist), so this avoids the cross-process clobber where
        the per-collect persist would overwrite the --train's training_steps
        increment with a stale in-memory value (which left training_steps
        stuck at the load-time value). Also called on _load for orphaned
        results from a crashed previous run."""
        if not self.paths:
            return
        try:
            with open(self.paths["training_result"]) as f:
                res = json.load(f)
        except (FileNotFoundError, Exception):
            return
        with self._state_lock:
            if "training_steps" in res:
                self.counters["training_steps"] = res["training_steps"]
            if "triggers_since_target_sync" in res:
                self.counters["triggers_since_target_sync"] = res["triggers_since_target_sync"]
            snapshot = dict(self.counters)
        _atomic_write_json(self.paths["counters"], snapshot)
        try:
            os.remove(self.paths["training_result"])
        except OSError:
            pass

    def state_summary(self):
        with self._state_lock:
            if not self.loaded:
                return {"loaded": False}
            mem_len = 0
            try:
                mem_len = len(self.dqn.Memory.memory)
            except Exception:
                pass
            return {
                "loaded": True,
                "samples_seen": self.counters.get("samples_seen", 0),
                "training_steps": self.counters.get("training_steps", 0),
                "replay_buffer_len": mem_len,
                "alias_count": len(self.alias_map),
                "state_dir": self.paths["dir"] if self.paths else None,
                "training_in_progress": self._training_in_progress,
            }


# --------------------------------------------------------------------------- #
# HTTP server.
# --------------------------------------------------------------------------- #
SERVER = None
SKILL = RTOSSkill()


def _drain_and_stop():
    try:
        SKILL.persist()  # persist() starts SERVER.shutdown in its finally (only if it owns the lock)
    except Exception as e:
        print(f"persist failed: {e}", file=sys.stderr)
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path.startswith("/state"):
            self._send_json(200, SKILL.state_summary())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/optimize":
            if SKILL._shutting_down:
                self._send_json(503, {"error": "server is shutting down"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(n).decode() if n else "{}"
                req = json.loads(raw) if raw else {}
            except Exception as e:
                self._send_json(400, {"error": f"bad request body: {e}"})
                return
            dsn = req.get("dsn")
            query = req.get("query")
            optimize_only = bool(req.get("optimize_only"))
            cfg = req.get("config")
            if isinstance(cfg, str) and cfg:
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {"initial_model": cfg}
            cfg = cfg or {}
            if not isinstance(cfg, dict):
                self._send_json(400, {"error": "config must be a JSON object"})
                return
            if not dsn or not query:
                self._send_json(400, {"error": "dsn and query are required"})
                return
            try:
                SKILL.ensure_loaded(cfg, dsn)
                result = SKILL.optimize(dsn, query, optimize_only)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            else:
                self._send_json(200, result)
        elif self.path == "/shutdown":
            self._send_json(200, {"status": "shutting down"})
            _drain_and_stop()
        else:
            self._send_json(404, {"error": "not found"})


def _signal_handler(signum, frame):
    _drain_and_stop()


def main():
    parser = argparse.ArgumentParser(description="RTOS join-order optimizer server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "0")))
    parser.add_argument("--train", action="store_true",
                        help="Run a single training cycle and exit (subprocess mode).")
    parser.add_argument("--state-dir", default=None, help="State directory (for --train mode).")
    parser.add_argument("--dsn", default=None, help="Database DSN (for --train mode).")
    args = parser.parse_args()

    if args.train:
        if not args.state_dir or not args.dsn:
            print("error: --train requires --state-dir and --dsn", file=sys.stderr)
            sys.exit(1)
        sys.exit(run_training(args.state_dir, args.dsn))

    if not args.port:
        print("error: --port or PORT env required", file=sys.stderr)
        sys.exit(1)

    global SERVER
    SERVER = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    print(f"RTOS skill server listening on 127.0.0.1:{args.port}", flush=True)
    SERVER.serve_forever()


if __name__ == "__main__":
    main()
