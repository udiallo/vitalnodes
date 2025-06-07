#!/usr/bin/env python3
"""
CLI für VitalNodes: Berechne Zentralitätsmetriken auf einem Netzwerk.

Beispiel:
  vitalnodes-cli network.edgelist --metrics gc igc erm --parallel --processes 4 --out results.csv
"""
import argparse
import sys
import json
import csv

import networkx as nx

from vitalnodes.orchestrator import compute_metrics, get_metric_names

# Mapping von Dateiendungen zu NetworkX-Ladern
_FORMAT_READERS = {
    ".edgelist":  nx.read_edgelist,
    ".adjlist":   nx.read_adjlist,
    ".gml":       nx.read_gml,
    ".graphml":   nx.read_graphml,
    ".gpickle":   nx.read_gpickle,
}

def _infer_and_load_graph(path: str, fmt: str | None = None) -> nx.Graph:
    """Lädt einen Graphen entweder nach explizitem Format oder anhand der Dateiendung."""
    if fmt:
        reader = {
            "edgelist": nx.read_edgelist,
            "adjlist":  nx.read_adjlist,
            "gml":      nx.read_gml,
            "graphml":  nx.read_graphml,
            "gpickle":  nx.read_gpickle,
        }.get(fmt.lower())
        if reader is None:
            raise ValueError(f"Unbekanntes Format: {fmt}")
        return reader(path)
    # Infer by extension
    for ext, reader in _FORMAT_READERS.items():
        if path.endswith(ext):
            return reader(path)
    # Fallback auf einfache edgelist
    return nx.read_edgelist(path)

def _write_json(data: dict, out_path: str):
    with open(out_path, "w") as fp:
        json.dump(data, fp, indent=2)

def _write_csv(results: dict[str, dict[int, float]], out_path: str):
    """
    CSV mit Zeilen: node, metric1, metric2, ...
    """
    metrics = list(results.keys())
    # Sammle alle Knoten (als Strings)
    nodes = sorted({int(n) for res in results.values() for n in res.keys()})
    with open(out_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        # Header
        writer.writerow(["node"] + metrics)
        # Zeilen
        for node in nodes:
            row = [node]
            for m in metrics:
                row.append(results[m].get(node, ""))
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(
        prog="vitalnodes-cli",
        description="Berechne VitalNodes-Zentralitätsmetriken auf einem Netzwerk."
    )
    parser.add_argument("graph", help="Pfad zur Graph-Datei (edgelist, adjlist, gml, graphml, gpickle)")
    parser.add_argument(
        "--format", "-f",
        choices=["edgelist","adjlist","gml","graphml","gpickle"],
        default=None,
        help="Falls angegeben, zwingend dieses Format verwenden"
    )
    parser.add_argument(
        "--metrics", "-m",
        nargs="+",
        required=True,
        help="Zu berechnende Metriken. Verfügbare: " + ", ".join(get_metric_names())
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Multiprocessing einschalten (default: für |V|>=500 automatisch)"
    )
    parser.add_argument(
        "--processes", "-P",
        type=int, default=None,
        help="Anzahl Worker-Prozesse (None → cpu_count()-1)"
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Ausgabepfad. Endet auf .json oder .csv (stdout, falls nicht gesetzt)"
    )

    args = parser.parse_args()

    try:
        G = _infer_and_load_graph(args.graph, args.format)
    except Exception as e:
        print(f"Fehler beim Laden des Graphen: {e}", file=sys.stderr)
        sys.exit(1)

    # Metrics berechnen
    try:
        results = compute_metrics(
            G,
            args.metrics,
            parallel=args.parallel,
            processes=args.processes
        )
    except KeyError as e:
        print(f"Unbekannte Metrik: {e}", file=sys.stderr)
        print("Verfügbare Metriken:", ", ".join(get_metric_names()), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fehler bei der Metrik-Berechnung: {e}", file=sys.stderr)
        sys.exit(1)

    # Ausgabe
    if args.out:
        if args.out.endswith(".json"):
            _write_json(results, args.out)
        elif args.out.endswith(".csv"):
            _write_csv(results, args.out)
        else:
            # Default JSON
            _write_json(results, args.out)
        print(f"Ergebnisse geschrieben nach {args.out}")
    else:
        # stdout als JSON
        json.dump(results, sys.stdout, indent=2)

if __name__ == "__main__":
    main()
