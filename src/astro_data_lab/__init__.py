"""Astro Data Lab: plain functions for training models on TESS light curves.

    1. sync()                                  update catalogues + light-curve manifest  (astro_data_lab.sync)
    2. load_curves(), load_tois(), ...         DataFrames of what exists                  (astro_data_lab.tables)
       -> you build a labelled table: data_uri, label, split
    3. read_lightcurve(data_uri)               one light curve as raw arrays, cached      (astro_data_lab.lightcurves)
    4. LightCurveDataset(table, transform)     PyTorch Dataset; pad_collate for batches   (astro_data_lab.dataset)
       -> you write `transform` (preprocessing), the model and the training loop
    5. report(y_true, y_pred, class_names)     balanced accuracy, confusion, 95% CI       (astro_data_lab.metrics)
"""
from astro_data_lab.dataset import LightCurveDataset, pad_collate
from astro_data_lab.lightcurves import LightCurve, LightCurveUnavailable, cache_info, read_lightcurve
from astro_data_lab.metrics import Report, report
from astro_data_lab.sync import sync
from astro_data_lab.tables import load_confirmed_planets, load_curves, load_tois, split_of

__all__ = [
    "LightCurve", "LightCurveDataset", "LightCurveUnavailable", "Report", "cache_info", "load_confirmed_planets",
    "load_curves", "load_tois", "pad_collate", "read_lightcurve", "report", "split_of", "sync",
]
