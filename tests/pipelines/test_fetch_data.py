from pathlib import Path

from src.pipelines.fetch_data import FetchTableConfig, fetch_table
from src.pipelines.lightcurves import LightcurveFetchConfig, fetch_lightcurve


def test_fetch_table_dry_run(tmp_path: Path) -> None:
    config = FetchTableConfig(table="koi", output_dir=tmp_path, dry_run=True, overwrite=True)
    output = fetch_table(config)
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "placeholder-001" in content


def test_fetch_lightcurve_dry_run(tmp_path: Path) -> None:
    config = LightcurveFetchConfig(
        object_id="KOI-0001",
        mission="kepler",
        output_dir=tmp_path,
        dry_run=True,
    )
    data = fetch_lightcurve(config)
    assert data.metadata["object_id"] == "KOI-0001"
    assert len(data.times_days) == len(data.flux_normalised)
    assert min(data.flux_normalised) < 1.0
