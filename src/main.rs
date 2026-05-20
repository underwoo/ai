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
    #[arg(short, long)]
    config: bool,

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

struct Config {
    api_key: Option<String>,
    base_url: String,
    model: String,
}

fn config_file_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ai")
        .join("config.toml")
}

fn load_config() -> Config {
    let path = config_file_path();
    let file_cfg: ConfigFile = if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => ConfigFile::default(),
        }
    } else {
        ConfigFile::default()
    };

    let api_key = std::env::var("AI_API_KEY").ok().or(file_cfg.api_key);
    let base_url = std::env::var("AI_BASE_URL")
        .ok()
        .or(file_cfg.base_url)
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let model = std::env::var("AI_MODEL")
        .ok()
        .or(file_cfg.model)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    Config { api_key, base_url, model }
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

fn fetch_command(config: &Config, prompt: &str) -> Result<String, String> {
    let api_key = config.api_key.as_deref().ok_or_else(|| {
        format!(
            "API key not set. Add api_key to {} or set AI_API_KEY.",
            config_file_path().display()
        )
    })?;

    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));

    let body = ChatRequest {
        model: config.model.clone(),
        messages: vec![
            Message { role: "system", content: SYSTEM_PROMPT.to_string() },
            Message { role: "user", content: prompt.to_string() },
        ],
        temperature: 0.0,
        max_tokens: 500,
    };

    let response = match ureq::post(&url)
        .set("Authorization", &format!("Bearer {}", api_key))
        .set("Content-Type", "application/json")
        .send_json(&body)
    {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            return Err(format!("API error {}: {}", code, body));
        }
        Err(ureq::Error::Transport(e)) => {
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

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    // Try arboard (native X11/Wayland)
    match arboard::Clipboard::new() {
        Ok(mut cb) => cb.set_text(text).map_err(|e| e.to_string()),
        Err(arboard_err) => {
            // Fall back to common CLI clipboard tools
            if try_cli_clipboard(text) {
                Ok(())
            } else {
                Err(format!("arboard: {}; no fallback tool found (xclip/xsel/wl-copy)", arboard_err))
            }
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
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();
    let config = load_config();

    if args.config {
        let path = config_file_path();
        println!("Config file : {}", path.display());
        println!("base_url    : {}", config.base_url);
        println!("model       : {}", config.model);
        println!(
            "api_key     : {}",
            if config.api_key.is_some() { "(set)" } else { "(not set)" }
        );
        return;
    }

    if args.prompt.is_empty() {
        eprintln!("error: provide a description of what you want to do.");
        eprintln!("Usage: ai <description...>");
        process::exit(1);
    }

    let prompt = args.prompt.join(" ");

    match fetch_command(&config, &prompt) {
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
        Ok(cmd) => {
            println!("{}", cmd);
            if let Err(e) = copy_to_clipboard(&cmd) {
                eprintln!("warning: clipboard unavailable: {}", e);
            }
        }
    }
}
