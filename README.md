# Ablation F on Google Loon Pet Project

Pruned v2 cold-start matches v1 (0.425 vs 0.425) — the architecture was never the problem.

The bad warmstart was killing Phase-2 v2, notart, pruned v2 converges to the same ceilingas v1 in ~2800 episodes. The per-preset split (calm 0.97, tropical 0.33, strong-shear 0.31) is also structurally identical to v1 — same strengths, same hard cases.

The implication: any further gains require either better exploration on the hard presets or RL pushing past the heuristic ceiling, not architecture changes.


```
{
    "ablation": "F_pruned_v2_coldstart",
    "winner_worker": 9,
    "best_score": 0.42509002057613166,
    "best_episode": 2799,
    "best_per_preset": {
      "tropical": 0.3298611111111111,
      "strong-shear": 0.3132716049382716,
      "calm": 0.9675925925925926
    },
    "n_workers": 10,
    "workers": [
      {
        "worker_id": 0,
        "best_score": 0.4173739711934157,
        "best_episode": 2699
      },
      {
        "worker_id": 1,
        "best_score": 0.3837448559670782,
        "best_episode": 2399
      },
      {
        "worker_id": 2,
        "best_score": 0.3539094650205761,
        "best_episode": 2699
      },
      {
        "worker_id": 3,
        "best_score": 0.3999485596707819,
        "best_episode": 2699
      },
      {
        "worker_id": 4,
        "best_score": 0.37435699588477367,
        "best_episode": 2699
      },
      {
        "worker_id": 5,
        "best_score": 0.3525591563786008,
        "best_episode": 2799
      },
      {
        "worker_id": 6,
        "best_score": 0.39634773662551437,
        "best_episode": 2799
      },
      {
        "worker_id": 7,
        "best_score": 0.42425411522633744,

        "worker_id": 8,
        "best_score": 0.390625,
        "best_episode": 2699
      },
      {
        "worker_id": 9,
        "best_score": 0.42509002057613166,
        "best_episode": 2799
      }
    ]
  }
```
