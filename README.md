# Vital-Node Identification Algorithms: Influence-Oriented Heuristics

This document contrasts **classical centrality measures** with **Vital-Node Identification algorithms**, emphasizing the latter’s role as *fast surrogates* for estimating dynamic spread influence (e.g., final outbreak size in an SIR epidemic).

---

## 1. Classical Centrality Measures

- **Degree**: Counts immediate connections.
- **Betweenness**: Counts shortest-path flows.
- **Closeness**: Inverse average distance to all nodes.
- **Eigenvector**: Importance of neighbors’ importance.

> *Limitations*:  
> These metrics capture **static structural importance**, but do **not** directly model how a node drives a spreading process like an epidemic.

---

## 2. Vital-Node Algorithms

Vital-Node heuristics **augment** static structure by:

1. Assigning each node a “mass” (e.g., degree, k-core number, entropy, eigenvector score).  
2. Summing pairwise interactions between node masses, **decaying** with graph distance (often inverse-square or similar).  
3. Producing a **single numeric score** per node that correlates more strongly with:
   - The **final fraction of nodes infected** in an SIR run seeded at that node.  
   - The **effectiveness of immunizing/removing** that node in reducing spread.  

### Families of Vital-Node Measures

- **Gravity-Family** (GC, IGC, DK-IGC, LGC, MCGM)  
- **Entropy-Based** (MCDE, ECRM, ERM, DSR, EDSR)  
- **NINL** (Node-Influence via Neighbour-Layer)  
- **Density-Based** (Density Centrality, CLD)  
- **GLI** (Global-Local Influence)  
- **LS** (Link-Strength)

---

## 3. Influence Approximation

Instead of running **thousands** of Monte Carlo epidemic simulations to rank seeds:

- **Compute** a Vital-Node score **once** (can be parallelized; often \`O(n²)\` or better with pruning).  
- **Rank** nodes by this score.  
- **Select** top-\`k\` nodes for immunization or removal.  

Empirical studies (e.g., Wang *et al.*, Physica A 2016; Liu *et al.*, Sci. Rep. 2022) show these heuristics **outperform** raw degree, betweenness, or random-walk measures in predicting true epidemic impact.

---

## 4. Key Takeaway

> **Vital-Node Identification algorithms** serve as **influence-oriented surrogates**, providing:
> - **Speed**: No nested epidemic loops.
> - **Accuracy**: Closer correlation with SIR outcomes.
> - **Scalability**: Parallelizable, accepts precomputed structures.

They bridge the gap between **graph topology** and **dynamics on the network**, making them ideal for *real-time* intervention planning and *risk assessment* in spreading processes.

---

*Based on ACM paper “Privacy-aware Edge Removal…” and classical epidemic-influence literature.*
