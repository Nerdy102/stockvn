# Production Runbook

## Start services

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

Check health:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metrics | head
curl -s http://localhost:9001/metrics | head
```

## Log rotation + bronze retention

- Worker writes runtime logs to container stdout/stderr: rotate via Docker logging driver (json-file max-size/max-file).
- Bronze retention cleanup job runs daily (`bronze_retention_cleanup`) and deletes old files + DB rows using `BRONZE_RETENTION_DAYS`.
- Manual cleanup:

```bash
python scripts/bronze_cleanup.py --days 30
```

## Redis backlog recovery

Symptoms: stream lag growing, worker delayed.

1. Check lag metrics (`redis_stream_lag`) on worker metrics endpoint.
2. Scale worker replicas temporarily.
3. Run once mode to drain:

```bash
python -m worker_scheduler.main --once
```

4. If consumer group is poisoned, recreate group after backing up pending IDs.

## Common failure modes

### Token 401 (SSI)

- Root cause: expired access token.
- Action: verify `SSI_*` creds, ensure token refresh path reachable, restart `stream_ingestor`.

### Websocket reconnect storm

- Root cause: remote instability or auth mismatch.
- Action: lower subscribe universe, verify token validity, check egress firewall and retry backoff.

### DB lock/contention

- Root cause: long-running transaction or vacuum lag.
- Action: inspect blocking query, reduce batch size, retry worker jobs, run `VACUUM ANALYZE` off-peak.

