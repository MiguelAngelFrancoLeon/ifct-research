#!/usr/bin/env python3
# main.py - CLI para validación IFCT cuaterniónico

import argparse
import importlib
import sys
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def run_validation(deltaG: float, out: str, fmt: str):
    # Importar tu módulo
    if "./" not in sys.path:
        sys.path.append("./")
    mod = importlib.import_module("teoria_matematica_validada")

    # Llamar a la validación
    diagnostics = mod.complete_mathematical_validation(deltaG=deltaG)

    # Aplanar resultados
    flat = {}
    for k, v in diagnostics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{kk}"] = vv
        else:
            flat[k] = v
    flat["deltaG"] = deltaG
    flat["timestamp"] = datetime.utcnow().isoformat()

    # Guardar resultados
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        pd.DataFrame([flat]).to_csv(out_path, index=False)
    elif fmt == "json":
        with open(out_path, "w") as f:
            json.dump(flat, f, indent=2)

    print(f"✅ Resultados guardados en {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación IFCT cuaterniónico")
    parser.add_argument("--deltaG", type=float, required=True, help="Valor de δG (ej: 0.921)")
    parser.add_argument("--out", type=str, default="results/output.csv", help="Ruta del archivo de salida")
    parser.add_argument("--fmt", choices=["csv", "json"], default="csv", help="Formato de salida (csv/json)")

    args = parser.parse_args()
    run_validation(args.deltaG, args.out, args.fmt)
