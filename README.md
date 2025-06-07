# VitalNodes

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/yourusername/vitalnodes/ci.yml)](https://github.com/yourusername/vitalnodes/actions)
[![License](https://img.shields.io/github/license/yourusername/vitalnodes.svg)](LICENSE)

**Vital-Node Identification Algorithms**  
Fast, influence-oriented heuristics for ranking and immunizing nodes in spreading processes.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Quick Start](#quick-start)  
4. [Command-Line Interface (CLI)](#command-line-interface-cli)  
5. [API Reference](#api-reference)  
6. [Available Metrics](#available-metrics)  
7. [Examples](#examples)  
8. [Testing](#testing)  
9. [Citation](#citation)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## Overview

Traditional centrality measures (degree, betweenness, closeness) quantify **static** structural importance.  
**VitalNodes** provides **influence-oriented surrogates**—single-pass graph heuristics that correlate much better with the actual **final outbreak size** in SIR-type spreading processes, while remaining efficient and parallelizable.

---

## Features

- **Gravity-Family**: GC, IGC, DK-IGC, LGC, MCGM  
- **Entropy-Based**: MCDE, MCDWE, ERM, DSR/EDSR, ECRM  
- **Neighbour-Layer**: NINL (configurable layers)  
- **Density-Based**: Density Centrality, CLD  
- **Global-Local Influence**: GLI, GLI-new  
- **Link-Strength**: LS  
- **Parallel execution** for large graphs  
- **Uniform façade** via `vitalnodes.orchestrator`  
- **Simple CLI** for on-the-fly metric computation  

---

## Quick Start

```python
import networkx as nx
from vitalnodes.orchestrator import compute_metric, compute_metrics, get_metric_names

# Create a graph
G = nx.karate_club_graph()

# Compute a single metric
scores_gc = compute_metric(G, "gc")

# Compute several metrics at once
batch = compute_metrics(G, ["gc", "erm", "ninl"], parallel=False)

# List all available metric keys
print(get_metric_names())
```

---

## API Reference

High‑level façade in `vitalnodes.orchestrator`:

- `get_metric_names() -> List[str]`  
- `compute_metric(G, metric, *, parallel=None, processes=None, **kwargs) -> Dict[node, score]`  
- `compute_metrics(G, metrics, *, parallel=None, processes=None, **kwargs) -> Dict[metric, Dict[node, score]]`

All metrics accept `parallel` and `processes` flags and metric-specific keyword arguments.

---

## Available Metrics

```
gravity-family:
  gc, gc+ (neighbor-aggregated), igc, igc+, dk, dk+, lgc, mcgm

entropy-family:
  mcde, mcde_weighted, erm, dsr, edsr, ecrm

NINL:
  ninl, ninl_layer0

density-family:
  density, cld

GLI:
  gli, gli_new

h-index:
  h_index, local_h_index

LS:
  ls
```



---

## License

Distributed under the [MIT License](LICENSE).