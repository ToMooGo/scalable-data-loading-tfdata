# Design notes

This page records what was taken from the reference book, what was changed and why, so that a reader can tell a design decision from an accident.

## Sources

The data-loading and preprocessing content is Chapter 13 of Géron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* (2019): the Data API, the TFRecord format, the Features API and the TensorFlow Datasets project. The California housing data, the stratified split, the median imputation and the t-based confidence interval come from Chapter 2 of the same book. Nothing in the data-loading part comes from another source.

The surrounding MLOps system (Prefect flows, MLflow tracking and registry, the FastAPI service, PostgreSQL, Docker Compose, the quality gates, drift monitoring and the CI workflow) is not in the book. It follows the structure of the public example repository the project was modelled on (`full-stack-on-prem-cv-mlops`): one configuration file, one command, train, evaluate and deploy flows, a monitor flow, and a service that serves the registered model.

## From the book to the code

| Book | Code |
|---|---|
| `csv_reader_dataset` (list files, interleave, shuffle, map, batch, prefetch) | `csv_pipeline.csv_reader_dataset`, with every knob exposed so each step can be switched off and measured |
| Splitting a large file into shards | `data.write_csv_shards` |
| `tf.train.Example`, `Features`, `BytesList`, `FloatList`, `Int64List` | `tfrecord.make_example` |
| `TFRecordWriter` and `TFRecordDataset`, GZIP option | `tfrecord.write_tfrecord_shards`, `tfrecord.tfrecord_reader_dataset` |
| `parse_single_example`, batched `parse_example`, `FixedLenFeature` with a default | `tfrecord.parse_single`, `tfrecord.parse_batch` |
| Bucketised, vocabulary, indicator, crossed and embedding columns | `features.build_preprocessing` (Keras 3 layers, see below) |
| `tfds.load(name="mnist", batch_size=32, as_supervised=True)` | `mnist_tfds.load_tfds_mnist` |
| Passing datasets to `fit()` with `steps_per_epoch` | `modeling.fit_model` |

## Where the project differs from the book, and why

**Keras 3 layers instead of `tf.feature_column`.** The book builds its features with `tf.feature_column` and `DenseFeatures`. TensorFlow has deprecated that API and Keras 3 no longer has `DenseFeatures`, so the same features are built with Keras preprocessing layers (`Normalization`, `Discretization`, `StringLookup`, `HashedCrossing`, `Embedding`). The boundaries, the vocabulary, the 20 x 20 grid and the bucket sizes (100 and 1,000) are the book's numbers. The mapping is in the table in `features.py`.

**Housing data only.** The book's Chapter 13 uses the housing data for the CSV and TFRecord parts. The only data file is the one from Chapter 2, downloaded from the book's own repository and checked against pinned SHA-256 digests.

**MNIST through a verified mirror when needed.** `tfds.load` downloads MNIST from a Google Cloud Storage address. Some firewalls and CI runners cannot reach that address; this project was built in such an environment. Where the address cannot be reached, the four official files are fetched from a public mirror, their MD5 checksums are compared with the official ones, and TFDS's own MNIST builder prepares them. After the download it is the real `tfds.load`. The route used is recorded with each run (`tfds` or `mirror`) and shown in the report.

**Image encodings.** The book stores an image in a TFRecord either as an encoded image (it shows `tf.io.encode_jpeg`) or as a serialised tensor (`tf.io.serialize_tensor`), both as a byte string. Part D compares three encodings of MNIST: the encoded image, with lossless PNG instead of the book's JPEG so that the pixels can be checked for equality; the serialised tensor; and the plain bytes of the pixel array, which is an addition of this project and the one that can be decoded for a whole batch at once. Size and decoding speed are compared.

**Two regimes for the throughput benchmark.** The book describes the effect of each pipeline step. Current TensorFlow rewrites a pipeline's graph with its own default optimisations before it runs it, and on the benchmark machine that hides much of the effect of the individual steps. The benchmark therefore reports two numbers: *only the steps listed* (the optimisations are switched off) and *tf.data defaults*. Both are real; the first answers "what does this step do", the second "what does a user get".

**`AUTOTUNE` as well as the book's constants.** The book's function uses fixed values: five files open at the same time with their lines interleaved (`n_readers=5`), no reading threads (`n_read_threads=None`), five parse threads and `prefetch(1)`. Part A includes these (variants 2 to 5 of the benchmark) and `tf.data.AUTOTUNE` for the reading threads, the parse threads and the prefetch, with the same five interleaved files (variant 6), and reports which is faster on the machine that ran the benchmark. The pipelines that train the models use `AUTOTUNE`.

**Quality-gate thresholds are choices, not book content.** The numbers in `configs/full_flow_config.yaml` (largest allowed test RMSE, smallest R-squared, minimum gain over Chapter 2's linear model, tolerance against the champion) were set after looking at the results of a first run and are meant to catch a broken model, not to define a good one. They sit between the linear model of Chapter 2 (which fails them) and the network that is deployed (which passes with a margin).

**The deployed feature set is chosen by a rule.** The first version of the configuration named `all features`. After Part C showed that this set was not better than the baseline for the neural network, the configuration was changed to `auto`: the set with the lowest mean validation RMSE of the neural network. The test set is not used for the choice, and the deployment was repeated with the rule (the earlier model stayed in the registry as the first champion).

## Rules the code follows

- Statistics for standardising and for replacing missing numbers are computed on the training rows only.
- The test set is used for the final report and for the quality gates. Feature sets are compared by their mean validation error and by paired differences over the same seeds; the confidence intervals are t-intervals as in Chapter 2.
- Missing numbers are replaced by the training median inside the pipeline (`record_defaults` for CSV, the `default_value` of `FixedLenFeature` for TFRecord), so a missing value never reaches the model as `NaN`.
- Preprocessing is part of the Keras model. The API sends the model the raw columns, so training and serving use the same code; a quality gate checks that the request path and the tf.data path give the same predictions to within 0.05 dollars.
- Every random choice takes the seed from the configuration.

## What is not claimed

- The timings are from a two-core cloud machine. Absolute numbers depend on the machine; ratios are what to compare. The tables record the median of several passes after a warm-up pass. Within one run, the same CSV pipeline timed in two experiments differed by about 10 %; the tables of only one complete run are stored.
- Housing is a small data set (20,640 rows, about 1.4 MB as CSV, of which the training rows take about 0.86 MB). The scale test repeats the training set 1, 10 and 100 times to measure how memory and speed grow; this is a test of the pipeline, not new data, and no model is trained on the repeated data.
- On such small data a pandas and NumPy baseline that loads the CSV shards into memory is faster than any of the streaming pipelines (about 355,000 examples per second against at most about 26,000 for the CSV pipelines and 62,195 for TFRecord, with no training step). The benchmark reports that, and the memory growth shows why streaming still matters when the data stops fitting in RAM.
- The quality gates and the comparison with the champion use the test set. The model and the feature set were chosen on validation data only, but the gate thresholds were set after a first run (see above), so the test numbers are a final report, not a fresh, untouched estimate for every decision made after looking at them.
