# ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A command-line tool that translates natural language descriptions into shell commands using an LLM API. The resulting command is printed to stdout and automatically copied to your clipboard.

```
$ ai list all open ports on this machine
ss -tulnp
```

## Installation

Requires [Rust](https://rustup.rs/) (2021 edition or later).

```bash
cargo install --path .
```

This installs the `ai` binary to `~/.cargo/bin/ai`.

## Quick Start

1. Set your API key:

   ```bash
   export AI_API_KEY="sk-..."
   ```

2. Describe what you want to do:

   ```bash
   ai find all files modified in the last 24 hours
   ```

   The command is printed and copied to your clipboard, ready to paste and run.

## Configuration

Configuration is resolved in this order (later sources override earlier ones):

| Source | Location |
|--------|----------|
| System config | `/etc/ai/config.toml` |
| Install prefix config | `<install_prefix>/etc/ai/config.toml` (e.g., conda env or module install) |
| User config | macOS: `~/Library/Application Support/ai/config.toml`<br>Linux: `~/.config/ai/config.toml` |
| Environment variables | `AI_API_KEY`, `AI_BASE_URL`, `AI_MODEL` |

The install prefix is auto-detected by walking up from the binary's location.

### Config file

```toml
api_key  = "sk-..."
base_url = "https://api.openai.com/v1"   # optional; this is the default
model    = "gpt-4o-mini"                 # optional; this is the default
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `AI_API_KEY` | API key for the LLM provider |
| `AI_BASE_URL` | Base URL of an OpenAI-compatible API |
| `AI_MODEL` | Model name to use |

### Inspect active configuration

```bash
ai --config
```

## Using an alternative LLM provider

Any OpenAI-compatible API works. For example, to use a local [Ollama](https://ollama.com) instance:

```bash
export AI_BASE_URL="http://localhost:11434/v1"
export AI_MODEL="llama3"
export AI_API_KEY="ollama"   # required by the client; value is ignored by Ollama
```

### NOAA HPC systems (hpc-job-analyst proxy)

On NOAA HPC login nodes where the `hpc-job-analyst` proxy is running,
no API key is needed. Load the module and use the tool directly:

```bash
module load ai
ai list all files modified in the last 24 hours
```

The tool is pre-configured to use the local proxy on `127.0.0.1:8742`.
To verify your configuration:

```bash
ai --config
```

If the proxy is not reachable, check its status with:

```bash
analyze-job proxy status
```

## Clipboard support

The generated command is automatically copied to the clipboard:

- **macOS** — uses the native clipboard via the `arboard` library.
- **Linux (X11/Wayland)** — tries `arboard` first, then falls back to `wl-copy`, `xclip`, or `xsel` if available.

If no clipboard mechanism is found, the command is still printed to stdout and a warning is displayed.

## Building from source

```bash
cargo build --release
# binary is at target/release/ai
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| [`clap`](https://crates.io/crates/clap) | CLI argument parsing |
| [`ureq`](https://crates.io/crates/ureq) | HTTP client |
| [`serde`](https://crates.io/crates/serde) / [`serde_json`](https://crates.io/crates/serde_json) | JSON serialization |
| [`toml`](https://crates.io/crates/toml) | Config file parsing |
| [`dirs`](https://crates.io/crates/dirs) | Platform config directory resolution |
| [`arboard`](https://crates.io/crates/arboard) | Clipboard access |
