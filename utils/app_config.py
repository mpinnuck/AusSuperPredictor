"""
Typed configuration dataclasses.

Replaces raw ``dict`` access with validated, IDE-friendly attributes.
Instantiate via ``AppConfig.from_dict(raw)`` and serialise back with
``.to_dict()`` for JSON persistence.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ── Leaf sections ────────────────────────────────────────────────

@dataclass
class DataConfig:
    local_csv_path: str = "australian_super_daily.csv"
    start_date: str = "01/07/2008"
    end_date_offset_days: int = 1
    fund_option: str = "Australian Shares"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        return cls(
            local_csv_path=d.get("local_csv_path", cls.local_csv_path),
            start_date=d.get("start_date", cls.start_date),
            end_date_offset_days=d.get("end_date_offset_days", cls.end_date_offset_days),
            fund_option=d.get("fund_option", cls.fund_option),
        )


@dataclass
class ModelConfig:
    save_path: str = "model.pkl"
    features_save_path: str = "features.pkl"
    n_estimators: int = 100
    max_depth: int = 7
    min_samples_split: int = 10
    min_samples_leaf: int = 15
    random_state: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(
            save_path=d.get("save_path", cls.save_path),
            features_save_path=d.get("features_save_path", cls.features_save_path),
            n_estimators=d.get("n_estimators", cls.n_estimators),
            max_depth=d.get("max_depth", cls.max_depth),
            min_samples_split=d.get("min_samples_split", cls.min_samples_split),
            min_samples_leaf=d.get("min_samples_leaf", cls.min_samples_leaf),
            random_state=d.get("random_state", cls.random_state),
        )


@dataclass
class ScheduleConfig:
    market_close_time: str = "16:00"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScheduleConfig":
        return cls(market_close_time=d.get("market_close_time", cls.market_close_time))


@dataclass
class LoggingConfig:
    level: str = "INFO"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoggingConfig":
        return cls(level=d.get("level", cls.level))


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_addr: str = ""     # "from" in JSON (reserved word)
    to: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmailConfig":
        return cls(
            enabled=d.get("enabled", cls.enabled),
            smtp_server=d.get("smtp_server", cls.smtp_server),
            smtp_port=d.get("smtp_port", cls.smtp_port),
            username=d.get("username", cls.username),
            password=d.get("password", cls.password),
            from_addr=d.get("from", cls.from_addr),
            to=d.get("to", cls.to),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize, mapping ``from_addr`` back to ``from``."""
        d = asdict(self)
        d["from"] = d.pop("from_addr")
        return d


@dataclass
class MarketSource:
    name: str = ""
    ticker: str = ""
    source: str = "yfinance"
    price_field: str = ""
    shift: bool = False
    category: str = ""
    live_source: str = ""
    live_ticker: str = ""
    live_page: str = ""

    # Dict-like access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default) or default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and bool(getattr(self, key))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketSource":
        return cls(
            name=d.get("name", ""),
            ticker=d.get("ticker", ""),
            source=d.get("source", "yfinance"),
            price_field=d.get("price_field", ""),
            shift=d.get("shift", False),
            category=d.get("category", ""),
            live_source=d.get("live_source", ""),
            live_ticker=d.get("live_ticker", ""),
            live_page=d.get("live_page", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Omit empty optional fields for cleaner JSON."""
        d: Dict[str, Any] = {}
        d["name"] = self.name
        d["ticker"] = self.ticker
        if self.source != "yfinance":
            d["source"] = self.source
        if self.price_field:
            d["price_field"] = self.price_field
        d["shift"] = self.shift
        d["category"] = self.category
        if self.live_source:
            d["live_source"] = self.live_source
        if self.live_ticker:
            d["live_ticker"] = self.live_ticker
        if self.live_page:
            d["live_page"] = self.live_page
        return d


@dataclass
class TechnicalIndicator:
    type: str = ""
    # MACD params
    fast: int = 0
    slow: int = 0
    signal: int = 0
    # RSI param
    period: int = 0

    # Dict-like access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        val = getattr(self, key, None)
        return val if val else default

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TechnicalIndicator":
        return cls(
            type=d.get("type", ""),
            fast=d.get("fast", 0),
            slow=d.get("slow", 0),
            signal=d.get("signal", 0),
            period=d.get("period", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.type == "macd":
            d["fast"] = self.fast
            d["slow"] = self.slow
            d["signal"] = self.signal
        elif self.type == "rsi":
            d["period"] = self.period
        return d


# ── Root config ──────────────────────────────────────────────────

@dataclass
class AppConfig:
    """Typed, validated application configuration."""
    data_folder: str = "~/Library/Application Support/AusSuperPredictor/data"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    market_sources: List[MarketSource] = field(default_factory=list)
    technical_indicators: List[TechnicalIndicator] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        """Build a typed config from a raw JSON dict.

        Raises ``KeyError`` / ``TypeError`` immediately for missing or
        malformed sections — fail-fast at startup, not mid-run.
        """
        return cls(
            data_folder=d.get("data_folder", cls.data_folder),
            data=DataConfig.from_dict(d.get("data", {})),
            model=ModelConfig.from_dict(d.get("model", {})),
            schedule=ScheduleConfig.from_dict(d.get("schedule", {})),
            logging=LoggingConfig.from_dict(d.get("logging", {})),
            email=EmailConfig.from_dict(d.get("email", {})),
            market_sources=[MarketSource.from_dict(s) for s in d.get("market_sources", [])],
            technical_indicators=[TechnicalIndicator.from_dict(i) for i in d.get("technical_indicators", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to a plain dict suitable for ``json.dump``."""
        return {
            "data_folder": self.data_folder,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "schedule": asdict(self.schedule),
            "logging": asdict(self.logging),
            "email": self.email.to_dict(),
            "market_sources": [s.to_dict() for s in self.market_sources],
            "technical_indicators": [i.to_dict() for i in self.technical_indicators],
        }

    def resolve_data_paths(self) -> None:
        """Expand ``data_folder`` and derive absolute file paths in-place."""
        folder = os.path.expanduser(self.data_folder)
        if not os.path.isabs(folder):
            folder = os.path.abspath(folder)
        os.makedirs(folder, exist_ok=True)
        self.data_folder = folder
        self.data.local_csv_path = os.path.join(folder, "australian_super_daily.csv")
        self.model.save_path = os.path.join(folder, "model.pkl")
        self.model.features_save_path = os.path.join(folder, "features.pkl")
