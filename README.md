# rushdb-hermes-memory

Native [RushDB](https://rushdb.com) persistent memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

The provider recalls scope-authorized historical context before inference, captures completed turns without blocking the response, and uses a profile-local durable outbox so acknowledged events survive process restarts.

## Requirements

- Python 3.11–3.13
- Hermes Agent with memory-provider entry-point support
- A RushDB project API key from [app.rushdb.com](https://app.rushdb.com), or a standalone RushDB
  engine and an embedding provider (see below)

## Install

Install the provider into the same Python environment as Hermes Agent:

```bash
pip install rushdb-hermes-memory
hermes memory setup
```

Set the RushDB project key in the Hermes environment:

```bash
export RUSHDB_API_KEY="your-project-api-key"
```

Optional environment variables:

- `RUSHDB_API_URL` selects a different RushDB Cloud API endpoint. To self-host, run the RushDB engine and use the settings below.
- `RUSHDB_MEMORY_SCOPE_SALT` makes participant hashes deployment-specific.

### A standalone RushDB engine

Recall is a vector search. RushDB Cloud embeds memories for you; a standalone engine (`rushdb`) does
not, so the provider embeds them itself through an OpenAI-compatible embeddings endpoint:

```bash
export RUSHDB_ENGINE_URL="http://127.0.0.1:3000"
export RUSHDB_ENGINE_TOKEN="your-engine-token"
# export RUSHDB_ENGINE_SCOPE="tenant-1"   # only for an engine started with --multitenant
export RUSHDB_EMBEDDING_API_KEY="sk-..."
export RUSHDB_EMBEDDING_MODEL="text-embedding-3-small"
export RUSHDB_EMBEDDING_DIMENSIONS="1536"
# export RUSHDB_EMBEDDING_BASE_URL="https://api.openai.com/v1"
```

On start the provider creates the external vector indexes it needs; every memory is written with its
vector and recall searches with the embedded query. With RushDB Cloud the embedding settings are
optional. When the configuration is incomplete, `hermes memory status` reports what is missing.

Hermes discovers the `rushdb` provider through the `hermes_agent.memory_providers` Python entry-point group.

## Behavior

- Recall searches `EPISODE.summary` and active `MEMORY_FACT.text` records with mandatory agent, profile, privacy, participant, and sandbox scope filters.
- Recall has a short fail-open timeout and falls back to bounded recent in-process episodes.
- Successful primary-agent turns are captured asynchronously as canonical `EPISODE` records.
- Explicit Hermes memory `add` and `replace` operations are mirrored as active `MEMORY_FACT` records.
- A `replace` (or a `patch`, which Hermes mirrors as a replace) deactivates the fact holding the entry it replaced and records its id in `supersedesFactId`; a `remove` deactivates the removed entry's fact. Both set `validUntil` and keep the old fact for history. The entry is matched by the exact `previous_content` Hermes reports, within the same scope and target; Hermes versions that don't report it keep the earlier fact active.
- Writes first enter an atomic local outbox under `$HERMES_HOME/rushdb-memory/outbox`, then use deterministic IDs and idempotent RushDB upserts.
- Surviving outbox files are retried while the process is running and replayed after restart.

## Privacy and trust

The provider uploads bounded user and assistant text from completed primary-agent turns. It does not automatically upload Hermes' complete `messages` payload, system prompts, tool calls, command output, or local file paths. Subagents do not write memory.

Recalled records are formatted as untrusted historical context, never as instructions or policy. Participant identifiers are stored as hashes rather than raw platform IDs.

## Development

```bash
uv sync --locked --group dev
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

## Release

CI verifies pull requests and every push to `main`. A successful `main` build publishes the version in `pyproject.toml` to PyPI when that exact version is not already present. Configure the `pypi` GitHub environment and its `PYPI_TOKEN` secret. Bump `version` before merging a subsequent release.

## License

Apache-2.0
