"""读取项目全局配置，在每次运行开始时合并参数覆盖值。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

import tomli

ROOT = Path(__file__).resolve().parent


def validate_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    return seed


def parse_ratio(value: str | int | float | Decimal) -> Decimal:
    try:
        ratio = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("b_ratio must be a number in (0, 1].") from error
    if not ratio.is_finite() or not 0 < ratio <= 1:
        raise ValueError("b_ratio must be a number in (0, 1].")
    return ratio


@dataclass(frozen=True)
class Config:
    seed: int
    seeds: tuple[int, ...]
    deterministic: bool
    b_ratio: Decimal

    def __post_init__(self) -> None:
        validate_seed(self.seed)
        for seed in self.seeds:
            validate_seed(seed)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("random.seeds must be a nonempty list of distinct seeds.")
        if type(self.deterministic) is not bool:
            raise ValueError("random.deterministic must be true or false.")
        object.__setattr__(self, "b_ratio", parse_ratio(self.b_ratio))

    @property
    def ratio_name(self) -> str:
        return format(self.b_ratio, "f").rstrip("0").rstrip(".")

    def for_seed(self, seed: int) -> Config:
        return replace(self, seed=seed)

    def snapshot(self) -> dict:
        return {
            "random": {"seed": self.seed, "seeds": list(self.seeds), "deterministic": self.deterministic},
            "data": {"b_ratio": self.ratio_name},
        }


def load_config(
    path: Path | None = None, *, seed: int | None = None, b_ratio: str | Decimal | None = None
) -> Config:
    path = ROOT / "config.toml" if path is None else Path(path)
    with path.open("rb") as stream:
        values = tomli.load(stream, parse_float=Decimal)
    expected = {"random": {"seed", "seeds", "deterministic"}, "data": {"b_ratio"}}
    if set(values) != set(expected):
        raise ValueError(f"{path}: expected [random] and [data] sections.")
    for section, keys in expected.items():
        if not isinstance(values[section], dict) or set(values[section]) != keys:
            raise ValueError(f"{path}: [{section}] must contain {', '.join(sorted(keys))}.")
    random = values["random"]
    if not isinstance(random["seeds"], list):
        raise ValueError("random.seeds must be a list.")
    return Config(
        seed=random["seed"] if seed is None else seed,
        seeds=tuple(random["seeds"]),
        deterministic=random["deterministic"],
        b_ratio=values["data"]["b_ratio"] if b_ratio is None else b_ratio,
    )
