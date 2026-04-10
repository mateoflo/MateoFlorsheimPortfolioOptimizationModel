from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from optimizer import AssetInput, PortfolioConfig, optimize_portfolio_from_tickers


DEFAULT_ROWS = [
    {"ticker": "SPY", "max_weight": "45"},
    {"ticker": "QQQ", "max_weight": "35"},
    {"ticker": "XLF", "max_weight": "20"},
    {"ticker": "XLV", "max_weight": "20"},
]


class PortfolioOptimizerApp:
    # This sets up the whole window and variables so the GUI can run.
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.root.geometry("1400x900")

        self.asset_rows: list[dict] = []
        self.setting_vars: dict[str, tk.StringVar] = {}
        self.boolean_vars: dict[str, tk.BooleanVar] = {}
        self.setting_entries: dict[str, ttk.Entry] = {}
        self.selector_vars: dict[str, tk.StringVar] = {}
        self.summary_vars: dict[str, tk.StringVar] = {}
        self.status_var = tk.StringVar(
            value="Enter stock tickers and portfolio settings. The optimizer will estimate stock statistics from market history and decide the split across stocks, cash, and Treasury bills."
        )

        self.build_layout()
        self.load_example()
        self.root.bind("<Return>", lambda _: self.run_optimization())

    # This lays out all the widgets so the window looks organized for the user.
    def build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(2, weight=1)

        settings = ttk.LabelFrame(outer, text="Portfolio Inputs", padding=10)
        settings.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for idx in range(4):
            settings.columnconfigure(idx, weight=1)

        fields = [
            ("capital", "Capital ($)", "100000"),
            ("lookback_years", "Lookback years", "3"),
            ("auto_max_floor", "Auto max floor %", "2"),
            ("auto_max_ceiling", "Auto max ceiling %", "10"),
            ("min_cash_weight", "Minimum cash %", "5"),
            ("max_cash_weight", "Maximum cash %", "30"),
            ("cash_yield", "Cash yield %", "4"),
            ("treasury_bill_yield", "Fallback T-bill %", "4.5"),
            ("target_expected_return", "Target return % (optional)", ""),
            ("target_volatility", "Target vol % (optional)", "12"),
            ("simulation_paths", "Monte Carlo paths", "10000"),
            ("simulation_horizon_years", "MC horizon years", "1"),
        ]
        for idx, (key, label, value) in enumerate(fields):
            field_frame = ttk.Frame(settings)
            field_frame.grid(row=idx // 4, column=idx % 4, sticky="ew", padx=6, pady=4)
            field_frame.columnconfigure(1, weight=1)
            ttk.Label(field_frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 6))
            var = tk.StringVar(value=value)
            self.setting_vars[key] = var
            entry = ttk.Entry(field_frame, textvariable=var, width=10)
            entry.grid(row=0, column=1, sticky="ew")
            self.setting_entries[key] = entry

        controls_row = ttk.Frame(settings)
        controls_row.grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=(4, 0))
        controls_row.columnconfigure(0, weight=1)
        controls_row.columnconfigure(1, weight=1)

        self.boolean_vars["auto_treasury_bill_yield"] = tk.BooleanVar(value=True)
        auto_tbill = ttk.Checkbutton(
            controls_row,
            text="Auto-fetch T-bill yield when online",
            variable=self.boolean_vars["auto_treasury_bill_yield"],
            command=self.toggle_tbill_fallback_state,
        )
        auto_tbill.grid(row=0, column=0, sticky="w")

        self.selector_vars["max_allocation_mode"] = tk.StringVar(value="Manual")
        max_alloc_frame = ttk.Frame(controls_row)
        max_alloc_frame.grid(row=0, column=1, sticky="e")
        ttk.Label(max_alloc_frame, text="Max allocation mode").grid(row=0, column=0, sticky="w", padx=(0, 6))
        selector = ttk.Combobox(
            max_alloc_frame,
            textvariable=self.selector_vars["max_allocation_mode"],
            values=["Manual", "Auto"],
            state="readonly",
            width=12,
        )
        selector.grid(row=0, column=1, sticky="w")
        selector.bind("<<ComboboxSelected>>", lambda _event: self.toggle_max_weight_mode())

        self.toggle_tbill_fallback_state()
        self.toggle_max_weight_mode()

        asset_frame = ttk.LabelFrame(outer, text="Assets", padding=10)
        asset_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        asset_frame.columnconfigure(0, weight=1)
        asset_frame.rowconfigure(2, weight=1)

        controls = ttk.Frame(asset_frame)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="Add Asset", command=self.add_asset_row).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Load Example", command=self.load_example).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Optimize Portfolio", command=self.run_optimization).pack(side="left")

        header = ttk.Frame(asset_frame)
        header.grid(row=1, column=0, sticky="ew")
        labels = ["Ticker", "Max Weight %", "Remove"]
        widths = [18, 14, 8]
        for col, (label, width) in enumerate(zip(labels, widths)):
            ttk.Label(header, text=label, width=width).grid(row=0, column=col, sticky="w", padx=3)

        asset_canvas = tk.Canvas(asset_frame, highlightthickness=0, height=180, bg=self.root.cget("bg"))
        asset_canvas.grid(row=2, column=0, sticky="nsew")
        asset_scroll = ttk.Scrollbar(asset_frame, orient="vertical", command=asset_canvas.yview)
        asset_scroll.grid(row=2, column=1, sticky="ns")
        asset_canvas.configure(yscrollcommand=asset_scroll.set)
        self.asset_rows_frame = ttk.Frame(asset_canvas)
        asset_canvas.create_window((0, 0), window=self.asset_rows_frame, anchor="nw")
        self.asset_rows_frame.bind("<Configure>", lambda _: asset_canvas.configure(scrollregion=asset_canvas.bbox("all")))

        output = ttk.LabelFrame(outer, text="Optimizer Output", padding=10)
        output.grid(row=2, column=0, sticky="nsew")
        output.rowconfigure(1, weight=1)
        output.columnconfigure(0, weight=1)

        summary = ttk.Frame(output)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(4):
            summary.columnconfigure(idx, weight=1)
        summary_fields = [
            ("expected_return", "Expected Return"),
            ("expected_volatility", "Expected Volatility"),
            ("risk_level", "Risk Level"),
            ("cash", "Cash"),
            ("treasury_bill", "Treasury Bills"),
            ("defensive", "Cash + T-Bills"),
            ("tbill_yield", "T-bill Yield"),
            ("tbill_source", "T-bill Source"),
            ("sample_window", "Data Window"),
            ("mc_expected_value", "MC Mean Value"),
            ("mc_median_value", "MC Median Value"),
            ("mc_var_5", "MC 5th %ile"),
            ("mc_loss_prob", "MC Loss Prob"),
            ("mc_paths", "MC Paths"),
        ]
        for idx, (key, label) in enumerate(summary_fields):
            card = ttk.Frame(summary, padding=(8, 4))
            card.grid(row=idx // 4, column=idx % 4, sticky="ew")
            ttk.Label(card, text=label).pack(anchor="w")
            var = tk.StringVar(value="--")
            self.summary_vars[key] = var
            ttk.Label(card, textvariable=var, font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        table_frame = ttk.Frame(output)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("ticker", "price", "exp_return", "vol", "target_weight", "shares", "dollars", "realized_weight")
        self.output_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        headings = {
            "ticker": "Ticker",
            "price": "Price",
            "exp_return": "Est Ret",
            "vol": "Est Vol",
            "target_weight": "Target Wt",
            "shares": "Shares",
            "dollars": "Invested $",
            "realized_weight": "Realized Wt",
        }
        widths = {
            "ticker": 100,
            "price": 90,
            "exp_return": 90,
            "vol": 90,
            "target_weight": 100,
            "shares": 90,
            "dollars": 130,
            "realized_weight": 100,
        }
        for key in columns:
            self.output_table.heading(key, text=headings[key])
            self.output_table.column(key, width=widths[key], anchor="center")
        self.output_table.grid(row=0, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.output_table.yview)
        output_scroll.grid(row=0, column=1, sticky="ns")
        self.output_table.configure(yscrollcommand=output_scroll.set)

        status = ttk.Label(output, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    # This adds a new row where the user can type another ticker and allocation.
    def add_asset_row(self, values: dict | None = None) -> None:
        values = values or {"ticker": "", "max_weight": "25"}
        row_frame = ttk.Frame(self.asset_rows_frame)
        row_frame.grid(row=len(self.asset_rows), column=0, sticky="ew", pady=2)

        ticker = tk.StringVar(value=str(values["ticker"]))
        max_weight = tk.StringVar(value=str(values["max_weight"]))

        ttk.Entry(row_frame, textvariable=ticker, width=18).grid(row=0, column=0, padx=3)
        max_weight_entry = ttk.Entry(row_frame, textvariable=max_weight, width=14)
        max_weight_entry.grid(row=0, column=1, padx=3)
        ttk.Button(row_frame, text="X", width=4, command=lambda rf=row_frame: self.remove_asset_row(rf)).grid(row=0, column=2, padx=3)

        self.asset_rows.append(
            {
                "frame": row_frame,
                "ticker": ticker,
                "max_weight": max_weight,
                "max_weight_entry": max_weight_entry,
            }
        )
        self.refresh_asset_rows()
        self.toggle_max_weight_mode()

    # This removes a row when the user no longer wants that ticker in play.
    def remove_asset_row(self, row_frame: ttk.Frame) -> None:
        self.asset_rows = [row for row in self.asset_rows if row["frame"] is not row_frame]
        row_frame.destroy()
        self.refresh_asset_rows()

    # This redraws the table whenever rows change so the scrollbar stays in sync.
    def refresh_asset_rows(self) -> None:
        for idx, row in enumerate(self.asset_rows):
            row["frame"].grid_configure(row=idx)

    # This fills in sample data so the user can see how the optimizer works.
    def load_example(self) -> None:
        for row in list(self.asset_rows):
            row["frame"].destroy()
        self.asset_rows.clear()
        for values in DEFAULT_ROWS:
            self.add_asset_row(values)

    # This reads the asset rows and turns them into AssetInput objects for optimization.
    def parse_assets(self) -> list[AssetInput]:
        assets: list[AssetInput] = []
        for row in self.asset_rows:
            assets.append(
                AssetInput(
                    ticker=row["ticker"].get().strip().upper(),
                    price=0.0,
                    expected_return=0.0,
                    volatility=0.0,
                    max_weight=self.parse_ratio(row["max_weight"].get()),
                )
            )
        return assets

    # This gathers the config fields so the optimizer knows the lookback, cash, etc.
    def parse_config(self) -> PortfolioConfig:
        target_raw = self.setting_vars["target_volatility"].get().strip()
        target_volatility = self.parse_ratio(target_raw) if target_raw else None
        target_return_raw = self.setting_vars["target_expected_return"].get().strip()
        target_expected_return = self.parse_ratio(target_return_raw) if target_return_raw else None
        return PortfolioConfig(
            capital=float(self.setting_vars["capital"].get()),
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=self.parse_ratio(self.setting_vars["min_cash_weight"].get()),
            max_cash_weight=self.parse_optional_ratio(self.setting_vars["max_cash_weight"].get()),
            cash_yield=self.parse_ratio(self.setting_vars["cash_yield"].get()),
            treasury_bill_yield=self.parse_optional_ratio(self.setting_vars["treasury_bill_yield"].get()),
            auto_max_allocation=self.selector_vars["max_allocation_mode"].get() == "Auto",
            auto_max_floor=self.parse_ratio(self.setting_vars["auto_max_floor"].get()),
            auto_max_ceiling=self.parse_ratio(self.setting_vars["auto_max_ceiling"].get()),
            auto_treasury_bill_yield=self.boolean_vars["auto_treasury_bill_yield"].get(),
            target_expected_return=target_expected_return,
            target_volatility=target_volatility,
            simulation_paths=int(float(self.setting_vars["simulation_paths"].get())),
            simulation_horizon_years=float(self.setting_vars["simulation_horizon_years"].get()),
        )

    @staticmethod
    # This turns a text percent into a decimal ratio the optimizer understands.
    def parse_ratio(raw: str) -> float:
        value = float(raw)
        if abs(value) >= 1:
            return value / 100.0
        return value

    @classmethod
    # This handles optional percent fields by allowing empty inputs.
    def parse_optional_ratio(cls, raw: str) -> float | None:
        cleaned = raw.strip()
        if not cleaned:
            return None
        return cls.parse_ratio(cleaned)

    # This triggers the entire optimization run and then updates the UI with results.
    def run_optimization(self) -> None:
        self.status_var.set("Estimating market statistics and resolving Treasury bill yield...")
        self.root.update_idletasks()
        try:
            assets = self.parse_assets()
            config = self.parse_config()
            lookback_years = float(self.setting_vars["lookback_years"].get())
            result = optimize_portfolio_from_tickers(assets, config, lookback_years=lookback_years)
        except Exception as exc:
            self.status_var.set("Optimization failed.")
            messagebox.showerror("Optimization Error", str(exc))
            return

        self.summary_vars["expected_return"].set(f"{result['expected_return']:.2%}")
        self.summary_vars["expected_volatility"].set(f"{result['expected_volatility']:.2%}")
        self.summary_vars["risk_level"].set(f"{result['risk_label']} ({result['risk_score']}/100)")
        self.summary_vars["cash"].set(f"${result['cash_dollars']:,.2f} ({result['cash_weight']:.2%})")
        self.summary_vars["treasury_bill"].set(
            f"${result['treasury_bill_dollars']:,.2f} ({result['treasury_bill_weight']:.2%})"
        )
        self.summary_vars["defensive"].set(
            f"${result['defensive_dollars']:,.2f} ({result['defensive_weight']:.2%})"
        )
        self.summary_vars["tbill_yield"].set(f"{result['treasury_bill_yield']:.2%}")
        self.summary_vars["tbill_source"].set(result["treasury_bill_source"])
        self.summary_vars["sample_window"].set(result["sample_window"])
        monte_carlo = result["monte_carlo"]
        self.summary_vars["mc_expected_value"].set(f"${monte_carlo['expected_terminal_value']:,.0f}")
        self.summary_vars["mc_median_value"].set(f"${monte_carlo['median_terminal_value']:,.0f}")
        self.summary_vars["mc_var_5"].set(f"${monte_carlo['value_at_5pct']:,.0f}")
        self.summary_vars["mc_loss_prob"].set(f"{monte_carlo['probability_of_loss']:.2%}")
        self.summary_vars["mc_paths"].set(f"{monte_carlo['paths']:,} @ {monte_carlo['horizon_years']:.1f}y")

        for item in self.output_table.get_children():
            self.output_table.delete(item)
        for row in result["asset_rows"]:
            self.output_table.insert(
                "",
                "end",
                values=(
                    row["ticker"],
                    f"${row['price']:,.2f}",
                    f"{row['expected_return']:.2%}",
                    f"{row['volatility']:.2%}",
                    f"{row['continuous_weight']:.2%} / cap {row['max_weight']:.2%}",
                    row["recommended_shares"],
                    f"${row['invested_dollars']:,.2f}",
                    f"{row['realized_weight']:.2%}",
                ),
            )
        self.status_var.set(
            "Optimization complete. Stock returns and risk were estimated from adjusted close history, and the model allocated across stocks, cash, and Treasury bills before running Monte Carlo."
        )

    # This enables or disables the fallback fetching of T-bill yields when toggled.
    def toggle_tbill_fallback_state(self) -> None:
        entry = self.setting_entries.get("treasury_bill_yield")
        if entry is None:
            return
        if self.boolean_vars["auto_treasury_bill_yield"].get():
            entry.configure(state="normal")
        else:
            entry.configure(state="normal")

    # This shows or hides manual max-weight fields depending on the selected mode.
    def toggle_max_weight_mode(self) -> None:
        manual_mode = self.selector_vars["max_allocation_mode"].get() == "Manual"
        for row in self.asset_rows:
            entry = row.get("max_weight_entry")
            if entry is not None:
                entry.configure(state="normal" if manual_mode else "disabled")


# This starts the Tkinter GUI when you run app.py from the command line.
def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    PortfolioOptimizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
