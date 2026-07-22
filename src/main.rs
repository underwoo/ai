use clap::Parser;
use std::path::PathBuf;
use std::process;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4o-mini";
const SYSTEM_PROMPT: &str =
    "You are a Linux shell command assistant. When the user describes what they want \
     to do, respond with ONLY the shell command that accomplishes it. Do not include \
     any explanation, markdown, code fences, backticks, or extra text. Output only \
     the bare command, ready to paste and run.";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "ai",
    about = "Suggest a shell command from a natural-language description",
    long_about = None
)]
struct Args {
    /// Print active configuration and exit
    #[arg(long)]
    print_config: bool,

    /// Use a custom config file (merged with other config sources)
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config_file: Option<PathBuf>,

    /// Increase output verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Natural-language description of what you want to do
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    prompt: Vec<String>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Default)]
struct ConfigFile {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
}

#[derive(Debug)]
struct Config {
    api_key: Option<String>,
    base_url: String,
    model: String,
}

/// Result of loading configuration, includes which files were loaded.
#[derive(Debug)]
struct ConfigResult {
    config: Config,
    loaded_paths: Vec<PathBuf>,
}

/// Detect the install prefix by walking up from the binary's location.
/// Returns the directory that contains `bin/ai` (e.g., `/opt/ai` if binary is `/opt/ai/bin/ai`).
fn install_prefix() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    let mut p = exe.as_path();
    loop {
        p = p.parent()?;
        if p.join("bin").join("ai").exists() {
            return Some(p.to_path_buf());
        }
        p.parent()?;
    }
}

/// Returns config search paths in priority order (lowest to highest):
/// 1. /etc/ai/config.toml (system-wide)
/// 2. <install_prefix>/etc/ai/config.toml (conda env / module install)
/// 3. ~/.config/ai/config.toml (per-user)
fn config_search_paths() -> Vec<PathBuf> {
    let mut paths = vec![PathBuf::from("/etc/ai/config.toml")];
    if let Some(prefix) = install_prefix() {
        paths.push(prefix.join("etc").join("ai").join("config.toml"));
    }
    if let Some(cfg_dir) = dirs::config_dir() {
        paths.push(cfg_dir.join("ai").join("config.toml"));
    }
    paths
}

/// Legacy helper for error messages — returns the user config path.
fn user_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ai")
        .join("config.toml")
}

/// Load and validate a custom config file specified via -c/--config.
/// Returns an error if the file doesn't exist or contains invalid TOML.
fn load_custom_config(path: &PathBuf) -> Result<ConfigFile, String> {
    if !path.exists() {
        return Err(format!("Config file not found: {}", path.display()));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config file {}: {}", path.display(), e))?;
    toml::from_str(&text)
        .map_err(|e| format!("Invalid TOML in config file {}: {}", path.display(), e))
}

fn load_config(custom_config: Option<&PathBuf>) -> Result<ConfigResult, String> {
    let mut merged = ConfigFile::default();
    let mut loaded_paths = Vec::new();

    // Merge config files in priority order (lowest to highest)
    for path in config_search_paths() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(c) = toml::from_str::<ConfigFile>(&text) {
                if c.api_key.is_some() {
                    merged.api_key = c.api_key;
                }
                if c.base_url.is_some() {
                    merged.base_url = c.base_url;
                }
                if c.model.is_some() {
                    merged.model = c.model;
                }
                loaded_paths.push(path);
            }
        }
    }

    // Custom config file has higher priority than standard paths
    if let Some(custom_path) = custom_config {
        let c = load_custom_config(custom_path)?;
        if c.api_key.is_some() {
            merged.api_key = c.api_key;
        }
        if c.base_url.is_some() {
            merged.base_url = c.base_url;
        }
        if c.model.is_some() {
            merged.model = c.model;
        }
        loaded_paths.push(custom_path.clone());
    }

    // Environment variables have highest priority
    let api_key = std::env::var("AI_API_KEY").ok().or(merged.api_key);
    let base_url = std::env::var("AI_BASE_URL")
        .ok()
        .or(merged.base_url)
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let model = std::env::var("AI_MODEL")
        .ok()
        .or(merged.model)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    Ok(ConfigResult {
        config: Config {
            api_key,
            base_url,
            model,
        },
        loaded_paths,
    })
}

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

#[derive(serde::Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(serde::Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(serde::Deserialize)]
struct ResponseMessage {
    content: String,
}

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------

/// Check if the URL points to a local proxy (127.0.0.1 or localhost).
fn is_local_proxy(base_url: &str) -> bool {
    base_url.starts_with("http://127.0.0.1") || base_url.starts_with("http://localhost")
}

/// Check if the URL is the OpenAI API (requires an API key).
fn is_openai_url(base_url: &str) -> bool {
    base_url.starts_with(DEFAULT_BASE_URL) || base_url.starts_with("https://api.openai.com")
}

fn fetch_command(config: &Config, prompt: &str) -> Result<String, String> {
    // API key is required only for OpenAI URLs; local proxies don't need one
    if config.api_key.is_none() && is_openai_url(&config.base_url) {
        return Err(format!(
            "API key not set. Add api_key to {} or set AI_API_KEY.",
            user_config_path().display()
        ));
    }

    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));

    let body = ChatRequest {
        model: config.model.clone(),
        messages: vec![
            Message {
                role: "system",
                content: SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user",
                content: prompt.to_string(),
            },
        ],
        temperature: 0.0,
        max_tokens: 500,
    };

    // Build request, conditionally adding Authorization header
    let mut req = ureq::post(&url).set("Content-Type", "application/json");
    if let Some(ref key) = config.api_key {
        req = req.set("Authorization", &format!("Bearer {}", key));
    }

    let response = match req.send_json(&body) {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            return Err(format!("API error {}: {}", code, body));
        }
        Err(ureq::Error::Transport(e)) => {
            if is_local_proxy(&config.base_url) {
                return Err(format!(
                    "Cannot reach the hpc-job-analyst proxy at {}.\n\
                     Check that the proxy service is running:\n\
                     \n    analyze-job proxy status",
                    config.base_url
                ));
            }
            return Err(format!("Connection failed: {}", e));
        }
    };

    let chat: ChatResponse = response
        .into_json()
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let raw = chat
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "Model returned an empty response.".to_string())?;

    let cmd = clean_command(&raw);
    if cmd.is_empty() {
        return Err("Model returned an empty response.".to_string());
    }

    Ok(cmd)
}

// Strip any accidental markdown wrapping the model may add despite instructions.
fn clean_command(raw: &str) -> String {
    let s = raw.trim();

    // Triple-backtick code fence (with optional language tag on opening line)
    if s.starts_with("```") {
        let inner: Vec<&str> = s
            .lines()
            .skip(1)
            .take_while(|l| !l.starts_with("```"))
            .collect();
        return inner.join("\n").trim().to_string();
    }

    // Single-backtick wrapping
    if s.starts_with('`') && s.ends_with('`') && s.len() > 2 {
        return s[1..s.len() - 1].trim().to_string();
    }

    s.to_string()
}

// ---------------------------------------------------------------------------
// Clipboard
// ---------------------------------------------------------------------------

/// Execute a closure with stderr temporarily suppressed.
/// Used to silence noisy X11/Wayland library errors like "Error: Can't open display".
#[cfg(unix)]
fn suppress_stderr<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    use std::os::unix::io::AsRawFd;

    // Open /dev/null for writing
    let devnull = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/null")
        .ok();

    // Save current stderr
    let saved_stderr = unsafe { libc::dup(2) };

    // Redirect stderr to /dev/null
    if let Some(ref dn) = devnull {
        unsafe { libc::dup2(dn.as_raw_fd(), 2) };
    }

    // Execute the closure
    let result = f();

    // Restore stderr
    if saved_stderr >= 0 {
        unsafe { libc::dup2(saved_stderr, 2) };
        unsafe { libc::close(saved_stderr) };
    }

    result
}

/// On non-Unix platforms, no stderr suppression is needed.
#[cfg(not(unix))]
fn suppress_stderr<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    f()
}

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    // On Linux, check if a display server is available before trying arboard.
    // This avoids triggering X11 library initialization which prints errors.
    // On macOS, arboard uses native Cocoa APIs that don't require DISPLAY.
    #[cfg(target_os = "linux")]
    let has_display = std::env::var("DISPLAY").is_ok() || std::env::var("WAYLAND_DISPLAY").is_ok();

    #[cfg(not(target_os = "linux"))]
    let has_display = true;

    let mut arboard_error: Option<String> = None;

    if has_display {
        // Suppress stderr to avoid X11 library noise like "Error: Can't open display"
        let result = suppress_stderr(|| match arboard::Clipboard::new() {
            Ok(mut cb) => cb.set_text(text).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        });

        if result.is_ok() {
            return result;
        }

        arboard_error = result.err();
    }

    // Fall back to CLI clipboard tools (also suppress their stderr)
    if suppress_stderr(|| try_cli_clipboard(text)) {
        return Ok(());
    }

    // Build detailed error message for -vvv
    match arboard_error {
        Some(e) => Err(format!(
            "arboard: {}; no fallback tool found (xclip/xsel/wl-copy)",
            e
        )),
        None => {
            Err("no display available; no fallback tool found (xclip/xsel/wl-copy)".to_string())
        }
    }
}

fn try_cli_clipboard(text: &str) -> bool {
    let tools: &[(&str, &[&str])] = &[
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["--clipboard", "--input"]),
    ];
    for (cmd, args) in tools {
        if let Ok(mut child) = std::process::Command::new(cmd)
            .args(*args)
            .stdin(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            use std::io::Write;
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(text.as_bytes());
            }
            if child.wait().map(|s| s.success()).unwrap_or(false) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Group 1: clean_command — pure function
    // -----------------------------------------------------------------------
    mod clean_command_tests {
        use super::*;

        #[test]
        fn plain_passthrough() {
            assert_eq!(clean_command("ls -la"), "ls -la");
        }

        #[test]
        fn trims_surrounding_whitespace() {
            assert_eq!(clean_command("  ls -la  "), "ls -la");
        }

        #[test]
        fn only_whitespace() {
            assert_eq!(clean_command("   "), "");
        }

        #[test]
        fn triple_backtick_no_lang() {
            assert_eq!(clean_command("```\nls -la\n```"), "ls -la");
        }

        #[test]
        fn triple_backtick_with_lang() {
            assert_eq!(clean_command("```bash\nls -la\n```"), "ls -la");
        }

        #[test]
        fn triple_backtick_multiline() {
            assert_eq!(clean_command("```sh\ncmd1\ncmd2\n```"), "cmd1\ncmd2");
        }

        #[test]
        fn triple_backtick_trims_inner_whitespace() {
            assert_eq!(clean_command("```\n  ls -la  \n```"), "ls -la");
        }

        #[test]
        fn triple_backtick_empty_body() {
            assert_eq!(clean_command("```\n```"), "");
        }

        #[test]
        fn whitespace_only_inside_fence() {
            assert_eq!(clean_command("```\n   \n```"), "");
        }

        #[test]
        fn single_backtick_wrapping() {
            assert_eq!(clean_command("`ls -la`"), "ls -la");
        }

        #[test]
        fn single_backtick_trims_inner() {
            assert_eq!(clean_command("`  ls -la  `"), "ls -la");
        }

        #[test]
        fn single_backtick_missing_closing() {
            assert_eq!(clean_command("`ls -la"), "`ls -la");
        }

        #[test]
        fn single_backtick_minimum_length_guard() {
            assert_eq!(clean_command("``"), "``");
        }

        #[test]
        fn single_backtick_three_chars() {
            assert_eq!(clean_command("`x`"), "x");
        }
    }

    // -----------------------------------------------------------------------
    // Group 2: serde round-trip tests
    // -----------------------------------------------------------------------
    mod serde_tests {
        use super::*;

        #[test]
        fn chat_response_single_choice() {
            let json = r#"{"choices":[{"message":{"role":"assistant","content":"ls -la"}}]}"#;
            let resp: ChatResponse = serde_json::from_str(json).unwrap();
            assert_eq!(resp.choices.len(), 1);
            assert_eq!(resp.choices[0].message.content, "ls -la");
        }

        #[test]
        fn chat_response_multiple_choices() {
            let json =
                r#"{"choices":[{"message":{"content":"ls -la"}},{"message":{"content":"dir"}}]}"#;
            let resp: ChatResponse = serde_json::from_str(json).unwrap();
            assert_eq!(resp.choices.len(), 2);
            assert_eq!(resp.choices[0].message.content, "ls -la");
        }

        #[test]
        fn chat_response_empty_choices() {
            let resp: ChatResponse = serde_json::from_str(r#"{"choices":[]}"#).unwrap();
            assert!(resp.choices.is_empty());
        }

        #[test]
        fn chat_response_ignores_extra_fields() {
            let json = r#"{
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ls -la"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }"#;
            let resp: ChatResponse = serde_json::from_str(json).unwrap();
            assert_eq!(resp.choices[0].message.content, "ls -la");
        }

        #[test]
        fn config_file_all_fields() {
            let toml =
                "api_key = \"sk-abc\"\nbase_url = \"https://custom.io/v1\"\nmodel = \"gpt-4\"";
            let cfg: ConfigFile = toml::from_str(toml).unwrap();
            assert_eq!(cfg.api_key.as_deref(), Some("sk-abc"));
            assert_eq!(cfg.base_url.as_deref(), Some("https://custom.io/v1"));
            assert_eq!(cfg.model.as_deref(), Some("gpt-4"));
        }

        #[test]
        fn config_file_partial_fields() {
            let cfg: ConfigFile = toml::from_str("model = \"gpt-4\"").unwrap();
            assert!(cfg.api_key.is_none());
            assert!(cfg.base_url.is_none());
            assert_eq!(cfg.model.as_deref(), Some("gpt-4"));
        }

        #[test]
        fn config_file_empty_toml() {
            let cfg: ConfigFile = toml::from_str("").unwrap();
            assert!(cfg.api_key.is_none());
            assert!(cfg.base_url.is_none());
            assert!(cfg.model.is_none());
        }

        #[test]
        fn config_file_ignores_unknown_keys() {
            let cfg: ConfigFile = toml::from_str("foo = \"bar\"\napi_key = \"sk-abc\"").unwrap();
            assert_eq!(cfg.api_key.as_deref(), Some("sk-abc"));
        }

        #[test]
        fn chat_request_serializes_correctly() {
            let req = ChatRequest {
                model: "gpt-4o-mini".to_string(),
                messages: vec![
                    Message {
                        role: "system",
                        content: "sys".to_string(),
                    },
                    Message {
                        role: "user",
                        content: "hello".to_string(),
                    },
                ],
                temperature: 0.0,
                max_tokens: 500,
            };
            let val = serde_json::to_value(&req).unwrap();
            assert_eq!(val["model"], "gpt-4o-mini");
            assert_eq!(val["temperature"], 0.0);
            assert_eq!(val["max_tokens"], 500);
            assert_eq!(val["messages"][0]["role"], "system");
            assert_eq!(val["messages"][1]["role"], "user");
            assert_eq!(val["messages"][1]["content"], "hello");
        }
    }

    // -----------------------------------------------------------------------
    // Group 3: fetch_command — HTTP behavior (uses mockito)
    // -----------------------------------------------------------------------
    mod fetch_command_tests {
        use super::*;

        fn test_config(base_url: &str) -> Config {
            Config {
                api_key: Some("test-key".to_string()),
                base_url: base_url.to_string(),
                model: "gpt-4o-mini".to_string(),
            }
        }

        fn ok_body(content: &str) -> String {
            serde_json::json!({
                "choices": [{"message": {"content": content}}]
            })
            .to_string()
        }

        #[test]
        fn missing_api_key_returns_err() {
            let config = Config {
                api_key: None,
                base_url: DEFAULT_BASE_URL.to_string(),
                model: DEFAULT_MODEL.to_string(),
            };
            let err = fetch_command(&config, "list files").unwrap_err();
            assert!(err.contains("API key not set"), "got: {err}");
        }

        #[test]
        fn success_plain_response() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("ls -la"))
                .create();
            assert_eq!(
                fetch_command(&test_config(&server.url()), "list files"),
                Ok("ls -la".to_string())
            );
        }

        #[test]
        fn success_strips_markdown_fence() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("```bash\nls -la\n```"))
                .create();
            assert_eq!(
                fetch_command(&test_config(&server.url()), "list files"),
                Ok("ls -la".to_string())
            );
        }

        #[test]
        fn api_returns_401() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(401)
                .with_body("Unauthorized")
                .create();
            let err = fetch_command(&test_config(&server.url()), "list files").unwrap_err();
            assert!(err.starts_with("API error 401"), "got: {err}");
        }

        #[test]
        fn api_returns_500() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(500)
                .with_body("Internal Server Error")
                .create();
            let err = fetch_command(&test_config(&server.url()), "list files").unwrap_err();
            assert!(err.starts_with("API error 500"), "got: {err}");
        }

        #[test]
        fn empty_choices_returns_err() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(r#"{"choices":[]}"#)
                .create();
            let err = fetch_command(&test_config(&server.url()), "list files").unwrap_err();
            assert_eq!(err, "Model returned an empty response.");
        }

        #[test]
        fn whitespace_only_content_returns_err() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("   "))
                .create();
            let err = fetch_command(&test_config(&server.url()), "list files").unwrap_err();
            assert_eq!(err, "Model returned an empty response.");
        }

        #[test]
        fn malformed_json_returns_err() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body("not json")
                .create();
            let err = fetch_command(&test_config(&server.url()), "list files").unwrap_err();
            assert!(err.starts_with("Failed to parse response"), "got: {err}");
        }

        #[test]
        fn trailing_slash_in_base_url_is_normalised() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("ls -la"))
                .create();
            let url_with_slash = format!("{}/", server.url());
            assert_eq!(
                fetch_command(&test_config(&url_with_slash), "list files"),
                Ok("ls -la".to_string())
            );
        }

        #[test]
        fn sends_correct_model_in_body() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .match_body(mockito::Matcher::Regex(
                    r#""model"\s*:\s*"custom-model""#.to_string(),
                ))
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("ls -la"))
                .create();
            let config = Config {
                api_key: Some("test-key".to_string()),
                base_url: server.url(),
                model: "custom-model".to_string(),
            };
            assert!(fetch_command(&config, "list files").is_ok());
        }

        #[test]
        fn sends_authorization_header() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .match_header("authorization", "Bearer test-key")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("ls -la"))
                .create();
            assert!(fetch_command(&test_config(&server.url()), "list files").is_ok());
        }
    }

    // -----------------------------------------------------------------------
    // Group 4: load_config — env var override logic
    // -----------------------------------------------------------------------
    mod load_config_tests {
        use super::*;
        use serial_test::serial;

        fn clear_ai_env() {
            std::env::remove_var("AI_API_KEY");
            std::env::remove_var("AI_BASE_URL");
            std::env::remove_var("AI_MODEL");
        }

        // This test requires no config file at the platform config path
        // (e.g. ~/Library/Application Support/ai/config.toml on macOS).
        // Ignored by default to avoid flakiness when a real config file exists.
        #[test]
        #[serial]
        #[ignore]
        fn defaults_when_nothing_set() {
            clear_ai_env();
            let result = load_config(None).unwrap();
            assert!(result.config.api_key.is_none());
            assert_eq!(result.config.base_url, DEFAULT_BASE_URL);
            assert_eq!(result.config.model, DEFAULT_MODEL);
        }

        #[test]
        #[serial]
        fn env_api_key_is_used() {
            clear_ai_env();
            std::env::set_var("AI_API_KEY", "sk-env");
            let result = load_config(None).unwrap();
            clear_ai_env();
            assert_eq!(result.config.api_key.as_deref(), Some("sk-env"));
        }

        #[test]
        #[serial]
        fn env_base_url_overrides() {
            clear_ai_env();
            std::env::set_var("AI_BASE_URL", "https://custom.io/v1");
            let result = load_config(None).unwrap();
            clear_ai_env();
            assert_eq!(result.config.base_url, "https://custom.io/v1");
        }

        #[test]
        #[serial]
        fn env_model_overrides() {
            clear_ai_env();
            std::env::set_var("AI_MODEL", "gpt-4");
            let result = load_config(None).unwrap();
            clear_ai_env();
            assert_eq!(result.config.model, "gpt-4");
        }

        #[test]
        #[serial]
        fn all_three_env_vars_override() {
            clear_ai_env();
            std::env::set_var("AI_API_KEY", "sk-all");
            std::env::set_var("AI_BASE_URL", "https://all.io/v1");
            std::env::set_var("AI_MODEL", "gpt-all");
            let result = load_config(None).unwrap();
            clear_ai_env();
            assert_eq!(result.config.api_key.as_deref(), Some("sk-all"));
            assert_eq!(result.config.base_url, "https://all.io/v1");
            assert_eq!(result.config.model, "gpt-all");
        }
    }

    // -----------------------------------------------------------------------
    // Group 5: config_search_paths — path discovery logic
    // -----------------------------------------------------------------------
    mod config_search_paths_tests {
        use super::*;

        #[test]
        fn returns_at_least_two_paths() {
            let paths = config_search_paths();
            // At minimum: /etc/ai/config.toml and user config
            assert!(
                paths.len() >= 2,
                "expected at least 2 paths, got {}",
                paths.len()
            );
        }

        #[test]
        fn first_path_is_etc() {
            let paths = config_search_paths();
            assert_eq!(paths[0], PathBuf::from("/etc/ai/config.toml"));
        }

        #[test]
        fn last_path_is_user_config() {
            let paths = config_search_paths();
            let last = paths.last().unwrap();
            // Should end with ai/config.toml
            assert!(
                last.ends_with("ai/config.toml"),
                "unexpected last path: {:?}",
                last
            );
            // Should NOT be /etc
            assert!(
                !last.starts_with("/etc"),
                "last path should be user config, not /etc"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Group 6: is_local_proxy and is_openai_url — helper functions
    // -----------------------------------------------------------------------
    mod url_helper_tests {
        use super::*;

        #[test]
        fn is_local_proxy_127() {
            assert!(is_local_proxy("http://127.0.0.1:8742/api/v1"));
            assert!(is_local_proxy("http://127.0.0.1/api"));
        }

        #[test]
        fn is_local_proxy_localhost() {
            assert!(is_local_proxy("http://localhost:8742/api/v1"));
            assert!(is_local_proxy("http://localhost/api"));
        }

        #[test]
        fn is_local_proxy_false_for_remote() {
            assert!(!is_local_proxy("https://api.openai.com/v1"));
            assert!(!is_local_proxy("http://example.com/api"));
        }

        #[test]
        fn is_openai_url_default() {
            assert!(is_openai_url(DEFAULT_BASE_URL));
        }

        #[test]
        fn is_openai_url_variants() {
            assert!(is_openai_url("https://api.openai.com/v1"));
            assert!(is_openai_url("https://api.openai.com/v2"));
        }

        #[test]
        fn is_openai_url_false_for_others() {
            assert!(!is_openai_url("http://127.0.0.1:8742/api/v1"));
            assert!(!is_openai_url("https://api.anthropic.com/v1"));
        }
    }

    // -----------------------------------------------------------------------
    // Group 7: layered config merge — uses tempfile
    // -----------------------------------------------------------------------
    mod layered_config_tests {
        use super::*;
        use serial_test::serial;

        /// Helper to merge config files without environment variable interference.
        /// Takes a list of (path, content) tuples and returns the merged ConfigFile.
        fn merge_config_files(files: &[(PathBuf, &str)]) -> ConfigFile {
            let mut merged = ConfigFile::default();
            for (path, content) in files {
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, content).unwrap();
            }
            // Now read them back in order
            for (path, _) in files {
                if let Ok(text) = std::fs::read_to_string(path) {
                    if let Ok(c) = toml::from_str::<ConfigFile>(&text) {
                        if c.api_key.is_some() {
                            merged.api_key = c.api_key;
                        }
                        if c.base_url.is_some() {
                            merged.base_url = c.base_url;
                        }
                        if c.model.is_some() {
                            merged.model = c.model;
                        }
                    }
                }
            }
            merged
        }

        #[test]
        fn higher_priority_overwrites_lower() {
            let dir = tempfile::tempdir().unwrap();
            let low = dir.path().join("low").join("config.toml");
            let high = dir.path().join("high").join("config.toml");

            let merged = merge_config_files(&[
                (low, "model = \"low-model\"\nbase_url = \"http://low.io\""),
                (high, "model = \"high-model\""),
            ]);

            assert_eq!(merged.model.as_deref(), Some("high-model"));
            assert_eq!(merged.base_url.as_deref(), Some("http://low.io")); // not overwritten
        }

        #[test]
        fn missing_file_is_skipped() {
            let dir = tempfile::tempdir().unwrap();
            let existing = dir.path().join("exists").join("config.toml");
            let missing = dir.path().join("missing").join("config.toml");

            // Only create the existing file
            std::fs::create_dir_all(existing.parent().unwrap()).unwrap();
            std::fs::write(&existing, "model = \"exists-model\"").unwrap();

            let mut merged = ConfigFile::default();
            for path in [&missing, &existing] {
                if let Ok(text) = std::fs::read_to_string(path) {
                    if let Ok(c) = toml::from_str::<ConfigFile>(&text) {
                        if c.model.is_some() {
                            merged.model = c.model;
                        }
                    }
                }
            }

            assert_eq!(merged.model.as_deref(), Some("exists-model"));
        }

        #[test]
        fn partial_configs_merge_correctly() {
            let dir = tempfile::tempdir().unwrap();
            let f1 = dir.path().join("f1").join("config.toml");
            let f2 = dir.path().join("f2").join("config.toml");
            let f3 = dir.path().join("f3").join("config.toml");

            let merged = merge_config_files(&[
                (f1, "api_key = \"key1\""),
                (f2, "base_url = \"http://url2.io\""),
                (f3, "model = \"model3\""),
            ]);

            assert_eq!(merged.api_key.as_deref(), Some("key1"));
            assert_eq!(merged.base_url.as_deref(), Some("http://url2.io"));
            assert_eq!(merged.model.as_deref(), Some("model3"));
        }

        #[test]
        #[serial]
        fn custom_config_file_is_loaded() {
            // Clear env vars to avoid interference
            std::env::remove_var("AI_API_KEY");
            std::env::remove_var("AI_BASE_URL");
            std::env::remove_var("AI_MODEL");

            let dir = tempfile::tempdir().unwrap();
            let custom = dir.path().join("custom.toml");
            std::fs::write(&custom, "model = \"custom-model\"").unwrap();

            let result = load_config(Some(&custom)).unwrap();
            assert_eq!(result.config.model, "custom-model");
            assert!(result.loaded_paths.contains(&custom));
        }

        #[test]
        fn custom_config_missing_file_returns_error() {
            let missing = PathBuf::from("/nonexistent/path/config.toml");
            let result = load_config(Some(&missing));
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Config file not found"));
        }

        #[test]
        fn custom_config_invalid_toml_returns_error() {
            let dir = tempfile::tempdir().unwrap();
            let bad = dir.path().join("bad.toml");
            std::fs::write(&bad, "this is not valid toml [[[").unwrap();

            let result = load_config(Some(&bad));
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Invalid TOML"));
        }
    }

    // -----------------------------------------------------------------------
    // Group 8: optional api_key tests
    // -----------------------------------------------------------------------
    mod optional_api_key_tests {
        use super::*;

        fn config_without_key(base_url: &str) -> Config {
            Config {
                api_key: None,
                base_url: base_url.to_string(),
                model: "test-model".to_string(),
            }
        }

        fn ok_body(content: &str) -> String {
            serde_json::json!({
                "choices": [{"message": {"content": content}}]
            })
            .to_string()
        }

        #[test]
        fn missing_api_key_non_openai_succeeds() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("ls -la"))
                .create();

            // Mock server URL is http://127.0.0.1:PORT, which is NOT openai
            let config = config_without_key(&server.url());
            let result = fetch_command(&config, "list files");
            assert!(result.is_ok(), "expected success, got: {:?}", result);
        }

        #[test]
        fn missing_api_key_openai_url_returns_err() {
            let config = config_without_key(DEFAULT_BASE_URL);
            let err = fetch_command(&config, "list files").unwrap_err();
            assert!(err.contains("API key not set"), "got: {err}");
        }

        #[test]
        fn no_auth_header_when_key_missing() {
            let mut server = mockito::Server::new();
            let _mock = server
                .mock("POST", "/chat/completions")
                // Explicitly expect NO authorization header
                .match_header("authorization", mockito::Matcher::Missing)
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(ok_body("ls -la"))
                .create();

            let config = config_without_key(&server.url());
            let result = fetch_command(&config, "list files");
            assert!(result.is_ok(), "expected success, got: {:?}", result);
        }

        #[test]
        fn local_proxy_connection_error_shows_friendly_message() {
            // Use a port that's definitely not listening
            let config = Config {
                api_key: None,
                base_url: "http://127.0.0.1:59999/api/v1".to_string(),
                model: "test-model".to_string(),
            };
            let err = fetch_command(&config, "list files").unwrap_err();
            assert!(
                err.contains("hpc-job-analyst proxy"),
                "expected friendly message, got: {err}"
            );
            assert!(
                err.contains("analyze-job proxy status"),
                "expected hint, got: {err}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // Load configuration, handling custom config file errors
    let config_result = match load_config(args.config_file.as_ref()) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    };
    let config = config_result.config;
    let loaded_paths = config_result.loaded_paths;

    // Verbosity level 2+: show which config files were loaded
    if args.verbose >= 2 {
        if loaded_paths.is_empty() {
            eprintln!("note: no config files loaded");
        } else {
            for path in &loaded_paths {
                eprintln!("note: loaded config from {}", path.display());
            }
        }
    }

    // --print-config: show configuration and exit
    if args.print_config {
        println!("Config search paths (lowest -> highest priority):");
        for path in config_search_paths() {
            let status = if path.exists() { "found" } else { "not found" };
            println!("  {}  ({})", path.display(), status);
        }
        if let Some(ref custom) = args.config_file {
            println!("  {}  (custom via -c)", custom.display());
        }
        println!();
        println!("Active values:");
        println!("  base_url  : {}", config.base_url);
        println!("  model     : {}", config.model);
        println!(
            "  api_key   : {}",
            if config.api_key.is_some() {
                "(set)"
            } else {
                "(not set)"
            }
        );
        return;
    }

    // Verbosity level 3+: show full config dump (similar to --print-config but to stderr)
    if args.verbose >= 3 {
        eprintln!("debug: base_url = {}", config.base_url);
        eprintln!("debug: model = {}", config.model);
        eprintln!(
            "debug: api_key = {}",
            if config.api_key.is_some() {
                "(set)"
            } else {
                "(not set)"
            }
        );
    }

    if args.prompt.is_empty() {
        eprintln!("error: provide a description of what you want to do.");
        eprintln!("Usage: ai [OPTIONS] <description...>");
        process::exit(1);
    }

    let prompt = args.prompt.join(" ");

    // Verbosity level 3+: show request details
    if args.verbose >= 3 {
        eprintln!(
            "debug: POST {}/chat/completions",
            config.base_url.trim_end_matches('/')
        );
    }

    match fetch_command(&config, &prompt) {
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
        Ok(cmd) => {
            println!("{}", cmd);
            if let Err(e) = copy_to_clipboard(&cmd) {
                // Tiered clipboard error output based on verbosity
                match args.verbose {
                    0 => {} // Silent
                    1 => eprintln!("note: clipboard unavailable"),
                    2 => eprintln!("note: clipboard unavailable"),
                    _ => eprintln!("note: clipboard unavailable: {}", e),
                }
            }
        }
    }
}
