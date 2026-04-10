from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape

from optimizer import AssetInput, PortfolioConfig, optimize_portfolio_from_tickers


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


@dataclass
class OptimizerPageDefaults:
    capital: str = "100000"
    lookback_years: str = "3"
    auto_max_floor: str = "2"
    auto_max_ceiling: str = "10"
    min_cash_weight: str = "5"
    max_cash_weight: str = "30"
    cash_yield: str = "4"
    treasury_bill_yield: str = "4.5"
    target_expected_return: str = ""
    target_volatility: str = "12"
    simulation_paths: str = "10000"
    simulation_horizon_years: str = "1"
    expected_return_shrinkage: str = "50"
    max_allocation_mode: str = "Manual"
    expected_return_method: str = "historical_mean"
    auto_treasury_bill_yield: bool = True


DEFAULT_ASSETS = [
    {"ticker": "SPY", "max_weight": "45"},
    {"ticker": "QQQ", "max_weight": "35"},
    {"ticker": "XLF", "max_weight": "20"},
    {"ticker": "XLV", "max_weight": "20"},
]


env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


# This turns the percent string into a small decimal ratio for the optimizer.
def parse_ratio(raw: str) -> float:
    value = float(raw)
    if abs(value) >= 1:
        return value / 100.0
    return value


# This lets optional percent strings stay empty or convert to decimals.
def parse_optional_ratio(raw: str | None) -> float | None:
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    return parse_ratio(cleaned)


# This reads the JSON body and makes the asset list and config objects.
def parse_request_payload(payload: dict) -> tuple[list[AssetInput], PortfolioConfig, float]:
    assets_payload = payload.get("assets", [])
    assets: list[AssetInput] = []
    for row in assets_payload:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        assets.append(
            AssetInput(
                ticker=ticker,
                price=0.0,
                expected_return=0.0,
                volatility=0.0,
                max_weight=parse_ratio(str(row.get("max_weight", "0"))),
            )
        )
    if not assets:
        raise ValueError("At least one stock ticker is required.")

    settings = payload.get("settings", {})
    config = PortfolioConfig(
        capital=float(settings.get("capital", "100000")),
        risk_aversion=4.0,
        shrinkage=0.20,
        concentration_penalty=0.05,
        min_cash_weight=parse_ratio(str(settings.get("min_cash_weight", "5"))),
        max_cash_weight=parse_optional_ratio(settings.get("max_cash_weight")),
        cash_yield=parse_ratio(str(settings.get("cash_yield", "4"))),
        treasury_bill_yield=parse_optional_ratio(settings.get("treasury_bill_yield")),
        auto_max_allocation=str(settings.get("max_allocation_mode", "Manual")) == "Auto",
        auto_max_floor=parse_ratio(str(settings.get("auto_max_floor", "2"))),
        auto_max_ceiling=parse_ratio(str(settings.get("auto_max_ceiling", "10"))),
        auto_treasury_bill_yield=bool(settings.get("auto_treasury_bill_yield", True)),
        target_expected_return=parse_optional_ratio(settings.get("target_expected_return")),
        target_volatility=parse_optional_ratio(settings.get("target_volatility")),
        expected_return_method=str(settings.get("expected_return_method", "historical_mean")),
        expected_return_shrinkage=parse_ratio(str(settings.get("expected_return_shrinkage", "50"))),
        hmm_states=2,
        simulation_paths=int(float(settings.get("simulation_paths", "10000"))),
        simulation_horizon_years=float(settings.get("simulation_horizon_years", "1")),
    )
    lookback_years = float(settings.get("lookback_years", "3"))
    return assets, config, lookback_years


# This makes the optimizer output ready to send back in JSON form.
def serialize_result(result: dict) -> dict:
    monte_carlo = result["monte_carlo"]
    return {
        "summary": {
            "expected_return": f"{result['expected_return']:.2%}",
            "expected_volatility": f"{result['expected_volatility']:.2%}",
            "risk_level": f"{result['risk_label']} ({result['risk_score']}/100)",
            "cash": f"${result['cash_dollars']:,.2f} ({result['cash_weight']:.2%})",
            "treasury_bill": f"${result['treasury_bill_dollars']:,.2f} ({result['treasury_bill_weight']:.2%})",
            "defensive": f"${result['defensive_dollars']:,.2f} ({result['defensive_weight']:.2%})",
            "tbill_yield": f"{result['treasury_bill_yield']:.2%}",
            "tbill_source": result["treasury_bill_source"],
            "sample_window": result["sample_window"],
            "mc_expected_value": f"${monte_carlo['expected_terminal_value']:,.0f}",
            "mc_median_value": f"${monte_carlo['median_terminal_value']:,.0f}",
            "mc_var_5": f"${monte_carlo['value_at_5pct']:,.0f}",
            "mc_loss_prob": f"{monte_carlo['probability_of_loss']:.2%}",
            "mc_paths": f"{monte_carlo['paths']:,} @ {monte_carlo['horizon_years']:.1f}y",
            "expected_return_method": result.get("return_model_info", {}).get("method", ""),
            "expected_return_description": result.get("return_model_info", {}).get("description", ""),
        },
        "asset_rows": [
            {
                "ticker": row["ticker"],
                "price": f"${row['price']:,.2f}",
                "expected_return": f"{row['expected_return']:.2%}",
                "volatility": f"{row['volatility']:.2%}",
                "target_weight": f"{row['continuous_weight']:.2%} / cap {row['max_weight']:.2%}",
                "recommended_shares": row["recommended_shares"],
                "invested_dollars": f"${row['invested_dollars']:,.2f}",
                "realized_weight": f"{row['realized_weight']:.2%}",
            }
            for row in result["asset_rows"]
        ],
    }


class PortfolioWebHandler(BaseHTTPRequestHandler):
    server_version = "PortfolioOptimizerWeb/1.0"

    # This suppresses the default HTTP logging so we keep the console clean.
    def log_message(self, format: str, *args) -> None:
        return

    # This serves GET requests for the homepage and health endpoints.
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.render_index()
            return
        if parsed.path.startswith("/static/"):
            self.serve_static(parsed.path)
            return
        if parsed.path == "/health":
            self.send_json({"status": "ok"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # This handles POST requests for optimization requests from the frontend.
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/optimize":
            self.handle_optimize()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # This writes the HTML file back to the browser for the main page.
    def render_index(self) -> None:
        template = env.get_template("index.html")
        html = template.render(
            defaults=asdict(OptimizerPageDefaults()),
            default_assets=DEFAULT_ASSETS,
            expected_return_methods=[
                ("historical_mean", "Historical Mean"),
                ("bayes_stein", "Bayes-Stein"),
                ("market_factor", "Market Factor"),
                ("black_litterman", "Black-Litterman"),
            ],
        )
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # This returns static files like CSS/JS for the frontend.
    def serve_static(self, path: str) -> None:
        relative = path.removeprefix("/static/")
        file_path = (STATIC_DIR / relative).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        content_type = "text/plain; charset=utf-8"
        if file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".svg":
            content_type = "image/svg+xml"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # This runs the optimizer and sends the JSON response for the web UI.
    def handle_optimize(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            assets, config, lookback_years = parse_request_payload(payload)
            result = optimize_portfolio_from_tickers(assets, config, lookback_years=lookback_years)
            self.send_json({"ok": True, "result": serialize_result(result)})
        except Exception as exc:
            self.send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=2),
                },
                status=HTTPStatus.BAD_REQUEST,
            )

    # This writes a JSON response back to the browser with the given status.
    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# This starts the HTTP server when the web frontend is launched.
def main() -> None:
    host = "127.0.0.1"
    port = 8080
    server = ThreadingHTTPServer((host, port), PortfolioWebHandler)
    print(f"Portfolio web app running at http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
