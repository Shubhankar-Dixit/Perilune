from __future__ import annotations

from pathlib import Path

from src.pipelines.catalogs import CataloguePaths, harmonise_catalogues


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + '\n', encoding='utf-8')


def test_harmonise_catalogues(tmp_path: Path) -> None:
    koi_csv = '''# comment
kepid,koi_pdisposition,koi_period,koi_duration,koi_depth
1001,CONFIRMED,5.0,2.4,500.0
1002,FALSE POSITIVE,10.0,3.0,800.0'''
    k2_csv = '''# comment
epic_hostname,disposition,pl_orbper,pl_trandur,pl_trandep
EPIC 123,CONFIRMED,2.0,1.0,0.05
EPIC 456,REFUTED,3.0,1.5,'''
    toi_csv = '''# comment
toi,tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep
1000.01,PC,4.0,2.0,1200.0
1000.02,FP,6.0,2.5,900.0'''

    koi_path = tmp_path / 'koi.csv'
    k2_path = tmp_path / 'k2.csv'
    toi_path = tmp_path / 'toi.csv'

    _write_csv(koi_path, koi_csv)
    _write_csv(k2_path, k2_csv)
    _write_csv(toi_path, toi_csv)

    df = harmonise_catalogues(CataloguePaths(koi=koi_path, k2=k2_path, toi=toi_path))

    assert set(df.mission.unique()) == {'kepler', 'k2', 'tess'}
    assert df[df['mission'] == 'k2']['depth_ppm'].iloc[0] == 500.0
    assert {'planet-candidate', 'false-positive'} == set(df['label'].unique())
    assert len(df) == 6
