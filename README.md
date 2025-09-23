# VitalNodes

**Vital-Node Identification Algorithms**  
Fast, influence-oriented heuristics for ranking and immunizing nodes in spreading processes.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features) 
3. [Installation](#installation) 
4. [Quick Start](#quick-start)  
5. [Command-Line Interface (CLI)](#command-line-interface-cli)  
6. [API Reference](#api-reference)  
7. [Available Metrics](#available-metrics)  
8. [Examples](#examples)  
9. [Testing](#testing)  
10. [Citation](#citation)  
11. [Contributing](#contributing)  
12. [License](#license)  

---

## Overview

Traditional centrality measures (degree, betweenness, closeness) quantify **static** structural importance.  
**VitalNodes** provides **influence-oriented surrogates**—single-pass graph heuristics that correlate much better with the actual **final outbreak size** in SIR-type spreading processes, while remaining efficient and parallelizable.

---

## Features

- **Gravity-Family**: GC, IGC, DK-IGC, LGC, MCGM, TGC  
- **Entropy-Based**: MCDE, MCDWE, ERM, DSR/EDSR, ECRM  
- **Neighbour-Layer**: NINL (configurable layers)  
- **Density-Based**: Density Centrality, CLD  
- **Global-Local Influence**: GLI, GLI-new  
- **Link-Strength**: LS  
- **Parallel execution** for large graphs  
- **Uniform façade** via `vitalnodes.orchestrator`  
- **Simple CLI** for on-the-fly metric computation  

---

## Installation

```
git clone git@github.com:udiallo/vitalnodes.git
cd vitalnodes
pip install -e .
```
The package requires the following dependencies:

- networkx
- numpy
- scipy
- pathos (multiprocessing)
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

If G is a static graph (nx.Graph):
- `compute_metric(G, metric, *, parallel=None, processes=None, **kwargs) -> Dict[node, score]`  
- `compute_metrics(G, metrics, *, parallel=None, processes=None, **kwargs) -> Dict[metric, Dict[node, score]]`

If G is a temporal graph (List[nx.Graph]):
  
  - If `metric == "tgc"`:
    - `compute_metric(G, metric, *, parallel=None, processes=None, **kwargs) -> Dict[node, score]` 
  - otherwise:
    - `compute_metric(G, metric, *, parallel=None, processes=None, **kwargs) -> List[Dict[node, score]]` 

  - `compute_metrics(G, metrics, *, parallel=None, processes=None, **kwargs) -> Dict[time, Dict[metric, Dict[node, score]]]`



All metrics accept `parallel` and `processes` flags and metric-specific keyword arguments.

---

## Available Metrics

```
gravity-family:
  gc, gc+ (neighbor-aggregated), igc, igc+, dk, dk+, lgc, mcgm, tgc

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

Classical metrics:
  degree, closeness, betweenness, eigenvector, pagerank
```



---

## License

Distributed under the [MIT License](LICENSE).