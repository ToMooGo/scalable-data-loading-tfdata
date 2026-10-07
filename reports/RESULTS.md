# Results of the latest training run

Generated 2026-10-05 08:38 UTC by `python run_flow.py`. Config `configs/full_flow_config.yaml`, seed 42, commit `unknown`, data SHA-256 `8a3727f4cf54`.

Every timing is the median of several passes after a warm-up pass on the machine that ran the flow; absolute numbers depend on that machine, ratios are what to compare.

## A. Data API on sharded CSV

The pipeline reads 12,384 training rows and returns the same numbers as pandas (largest difference 2.9e-06).

Reading speed (examples per second, no training step):

| Pipeline | Only the listed steps | tf.data defaults |
|---|---|---|
| 1. one file at a time, no shuffle, serial parse | 13,412 | 21,484 |
| 2. + interleave 5 files | 15,632 | 19,019 |
| 3. + shuffle buffer | 16,039 | 19,189 |
| 4. + 5 parse threads | 18,935 | 25,885 |
| 5. + prefetch(1)  [the book's function] | 20,606 | 19,168 |
| 6. + AUTOTUNE threads and prefetch | 15,525 | 15,104 |

Shuffling sorted data (Spearman correlation of position and target: 1 = untouched, 0 = mixed):

| Shuffling | Spearman | Predicted | Batch diversity |
|---|---|---|---|
| no shuffling | 1.000 |  | 0.006 |
| random file order only | -0.011 |  | 0.058 |
| interleave 5 files | 0.005 |  | 0.920 |
| buffer 100 | 1.000 | 1.000 | 0.033 |
| buffer 1000 | 0.965 | 0.963 | 0.267 |
| buffer 10000 | 0.061 |  | 0.974 |
| interleave 5 + buffer 100 | 0.005 |  | 0.916 |
| interleave 5 + buffer 1000 | 0.006 |  | 0.943 |
| interleave 5 + buffer 10000 | -0.010 |  | 0.987 |

## B. TFRecord

| Format | Bytes | Size vs CSV | Examples/s |
|---|---|---|---|
| csv | 856,778 | 1.00 | 16,602 |
| csv.gz | 287,248 | 0.34 | 16,405 |
| tfrecord, parse one record at a time | 3,576,363 | 4.17 | 17,964 |
| tfrecord, parse a batch at a time | 3,576,363 | 4.17 | 60,155 |
| tfrecord.gz, parse a batch at a time | 483,662 | 0.56 | 62,195 |

Scale test (the training set repeated N times):

| Rows | CSV MB | TFRecord.gz MB | tf.data memory growth MB | pandas memory growth MB | tf.data examples/s |
|---|---|---|---|---|---|
| 12,384 | 0.9 | 0.5 | 11 | 8 | 60,138 |
| 123,840 | 8.5 | 4.7 | 26 | 34 | 64,762 |
| 1,238,400 | 85.4 | 47.2 | 31 | 182 | 60,434 |

## C. Features inside the model

5 seeds per cell, mean validation RMSE in dollars with a 95% interval; the last column is the paired difference from `+ ocean one-hot` (negative = better).

Head: `linear`

| Feature set | Inputs | Parameters | Valid RMSE | low | high | vs baseline |
|---|---|---|---|---|---|---|
| numeric only | 8 | 9 | 69,626 | 69,595 | 69,656 | +770 |
| + ocean one-hot | 14 | 15 | 68,856 | 68,824 | 68,888 |  |
| + ocean embedding | 10 | 23 | 68,902 | 68,860 | 68,943 | +46 |
| + income buckets | 19 | 20 | 67,980 | 67,960 | 68,000 | -876 |
| + age x ocean | 18 | 419 | 68,580 | 68,558 | 68,602 | -276 |
| + grid one-hot (400) | 414 | 415 | 64,226 | 64,171 | 64,282 | -4,630 |
| + grid embedding (400) | 22 | 3,223 | 64,583 | 64,515 | 64,652 | -4,273 |
| + hashed grid (1000) | 22 | 8,023 | 64,675 | 64,524 | 64,825 | -4,182 |
| all features | 31 | 8,432 | 63,211 | 63,130 | 63,292 | -5,645 |

Head: `mlp`

| Feature set | Inputs | Parameters | Valid RMSE | low | high | vs baseline |
|---|---|---|---|---|---|---|
| numeric only | 8 | 4,801 | 52,303 | 51,918 | 52,689 | -312 |
| + ocean one-hot | 14 | 5,185 | 52,616 | 52,192 | 53,039 |  |
| + ocean embedding | 10 | 4,941 | 52,180 | 51,515 | 52,844 | -436 |
| + income buckets | 19 | 5,505 | 53,079 | 52,612 | 53,546 | +463 |
| + age x ocean | 18 | 5,841 | 52,706 | 52,278 | 53,134 | +90 |
| + grid one-hot (400) | 414 | 30,785 | 52,759 | 52,276 | 53,242 | +143 |
| + grid embedding (400) | 22 | 8,897 | 52,230 | 51,539 | 52,921 | -385 |
| + hashed grid (1000) | 22 | 13,697 | 52,002 | 51,597 | 52,407 | -614 |
| all features | 31 | 14,673 | 52,803 | 52,361 | 53,245 | +188 |

## D. TensorFlow Datasets

`tfds.load('mnist', batch_size=32, as_supervised=True)` via **mirror**: 1875 training batches and 313 test batches. Test accuracy after 5 epochs: 0.9757 from `tfds.load`, 0.9764 from TFRecord files.

| Encoding | Compression | Bytes/image | Images/s |
|---|---|---|---|
| raw | none | 838 | 23,461 |
| raw | GZIP | 173 | 26,791 |
| png | none | 331 | 23,474 |
| png | GZIP | 242 | 22,866 |
| tensor | none | 857 | 27,490 |
| tensor | GZIP | 173 | 25,460 |

## E. Deployed model

Feature set `+ hashed grid (1000)`; test RMSE **$50,751** (R² 0.802); linear baseline $67,368; always-predict-the-mean $114,164. Largest difference between the request path and the tf.data path: 0.00e+00 USD.

