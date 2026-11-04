"""Inspecciona la estructura de los 4 datasets crudos descargados en data/raw/.

Solo lee encabezados/muestras (rápido). Requiere los datos crudos en el repo.
Uso:  python examples/inspect_raw_datasets.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from routedata.validate import print_summary, validate_all

if __name__ == "__main__":
    reports = validate_all()
    print_summary(reports)
