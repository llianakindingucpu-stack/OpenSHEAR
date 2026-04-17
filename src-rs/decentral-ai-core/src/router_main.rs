// DecentralAI Router — Standalone binary entry point
// Run with: cargo run --bin router -- --port 8082 --db-path ../data/router.db

use decentral_ai_core::router::{build_router_app, RouterState, start_health_checker};
use std::net::SocketAddr;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    tracing::info!("========================================");
    tracing::info!("  DecentralAI Router v0.1.0");
    tracing::info!("========================================");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let mut port: u16 = 8082;
    let mut db_path = String::from("D:/IdeaProjects/decentral-ai/data/router.db");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or(8082);
                    i += 2;
                } else { i += 1; }
            }
            "--db-path" => {
                if i + 1 < args.len() {
                    db_path = args[i + 1].clone();
                    i += 2;
                } else { i += 1; }
            }
            _ => i += 1,
        }
    }

    tracing::info!("Port:     {}", port);
    tracing::info!("DB:       {}", db_path);

    // Create data dir (use parent of db file)
    if let Some(parent) = std::path::Path::new(&db_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Initialize router state
    let state = RouterState::new(db_path);
    state.init_db();

    // Start background health checker
    let state_clone = state.clone();
    start_health_checker(state_clone).await;

    // Build HTTP app
    let app = build_router_app(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("========================================");
    tracing::info!("  Router HTTP: http://127.0.0.1:{}", port);
    tracing::info!("  Endpoints:");
    tracing::info!("    GET  /health       — 健康检查");
    tracing::info!("    GET  /nodes        — 列出节点");
    tracing::info!("    POST /register     — 注册节点");
    tracing::info!("    DELETE /unregister/:id — 注销节点");
    tracing::info!("    POST /heartbeat    — 节点心跳");
    tracing::info!("    GET  /stats        — 路由统计");
    tracing::info!("    GET  /balance      — 查询余额");
    tracing::info!("    POST /forward      — 路由推理请求");
    tracing::info!("========================================");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
