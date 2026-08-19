# LungCancerPrognosis

Multiple Instance Learning that recovers **instance-level scores** from **bag-level labels only**,
while jointly learning a **sparse non-negative feature weighting**.

MATLAB R2017b+ with the Statistics and Machine Learning Toolbox (`pdist2`, `mink`, `range`).

For the method description, tuning guide, per-file reference, and known limitations, see
[MANUAL.md](MANUAL.md).

## Install

```matlab
addpath('/path/to/MIL_prognosis');
```

## Usage

```matlab
[Weight, f, loss, f_history] = MIL_TL(data, bag_labels, weight_init, label_init, bag_ids, Para);
```

### Inputs

| Argument | Shape | Description |
|---|---|---|
| `data` | `d × N` | Feature matrix. **Features in rows, instances in columns.** Standardize features before calling. |
| `bag_labels` | `M × 1` | Binary label (`0`/`1`) for each bag. |
| `weight_init` | `d × 1` | Initial feature weights, non-negative. Use `ones(d,1)*Para.tau/d`. |
| `label_init` | `N × 1` | Initial instance scores. Use `rand(N,1)` or `bag_labels(bag_ids)`. Keep in `[0,1]`. |
| `bag_ids` | `N × 1` | Bag membership of each instance. **Must be integers `1..M` indexing `bag_labels`.** |
| `Para` | struct | Parameters, see below. |

Remap non-consecutive bag identifiers first:

```matlab
[unique_bags, ~, bag_ids] = unique(raw_bag_id);   % bag_ids is now 1..M
```

### `Para` fields

| Field | Required | Default | Description |
|---|---|---|---|
| `Niter` | yes | — | Max outer alternating iterations (use `≥ 2`). |
| `lambda1` | yes | — | Smoothness weight relative to the bag loss. |
| `tau` | yes | — | L1 budget: output satisfies `sum(Weight) ≤ tau`. |
| `label_iter` | yes | — | Inner steps for `f`. Scalar, or `[min max]` to ramp across outer iterations. |
| `Weight_iter` | yes | — | Inner steps for `Weight`. Scalar or `[min max]`. |
| `sigma` | no | `1` | Kernel bandwidth. |
| `distance_metric` | no | `'cityblock'` | `'cityblock'` or `'squaredeuclidean'`. |
| `truncate_k` | no | `25` | Neighbours per instance in the k-NN graph. Must be `≥ 1`. |

### Outputs

| Output | Shape | Description |
|---|---|---|
| `Weight` | `d × 1` | Learned feature weights, non-negative, `sum ≤ tau`. Zeros mean the feature was deselected. |
| `f` | `N × 1` | Instance scores, min-max rescaled to `[0,1]`. All zeros if the solution collapsed to a constant. |
| `loss` | `(iter+1) × 1` | Objective per outer iteration; `loss(1)` is the value before optimization. |
| `f_history` | `N × (iter+1)` | `f` after each outer iteration, **un-normalized**. |

Selected features:

```matlab
selected = find(Weight / max(Weight) >= 0.01);
```

