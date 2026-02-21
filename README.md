# D-FINE for document layout analysis

This is an adaptation//refactor of [ArgoHA's implementation](https://github.com/ArgoHA/custom_d_fine) of the
[D-FINE](https://arxiv.org/abs/2410.13842) object detector intended to put it
in line with community standards in the (historical) document layout analysis
community. It is a foundation for eventual integration into the [kraken ATR engine](https://kraken.re).

## Installation

Kraken is currently being rewritten to allow integration of new methods, such
as the one in this repository, with plug-ins. This repository uses the
architecture introduced by this rework which will eventually become kraken 7.0.
The models produced by this repository are *not* going to be compatible with
earlier kraken versions.

Clone the repository and run:

```bash
$ pip install .
```

This will install a `dfine` that can be used to train models:

```bash
$ dfine ... train ...
```

## Training

The basic syntax is very similar to kraken segmentation training using Page or
ALTO XML files. During the rework many of the segmentation dataset filtering
and transformation options have disappeared, being replaced by dictionaries
mapping class labels to indices. As it is annoying to define mapping on the
command line, mappings that do not assign one index to each class in the source
data need to be defined in YAML experiment configuration files.

To train a basic model for 50 epochs from scratch:

```bash
$ dfine -d cuda:0 train *.xml
```

The default configuration trains lines and regions jointly. If this is not what
you want take a look at the [sample configuration
file](experiments/config.yaml) to see how to disable text line detection in
bbox format.

## Inference

Inference is integrated in kraken. You need to convert the checkpoint into
weights first:

```bash
$ ketos convert -o dfine.safetensors checkpoint.ckpt
```

and then run `kraken ... segment ...` as usual:

```bash
$ kraken -i input.jpg out.xml -a segment -i dfine.safetensors
```

### Pretrained Models

Pretrained region segmentation models trained on the
[LADaS](https://github.com/DEFI-COLaF/LADaS/) dataset using the
[SegmOnto](https://segmonto.github.io/) taxonomy (37 region types) are
available for all model variants. They can be downloaded with `kraken download`:

| Variant     | Size    | mAP@50 | Download command                         |
|-------------|---------|--------|------------------------------------------|
| nano        | ~15 MB  | 0.2960 | `kraken get 10.5281/zenodo.18715384`|
| small       | ~42 MB  | 0.3864 | `kraken get 10.5281/zenodo.18715381`|
| medium      | ~79 MB  | 0.3915 | `kraken get 10.5281/zenodo.18715373`|
| large       | ~126 MB | 0.4308 | `kraken get 10.5281/zenodo.18715364`|
| extra_large | ~252 MB | 0.4164 | `kraken get 10.5281/zenodo.18715367`|

The **large** variant achieves the best performance. The extra_large variant
shows slight regression likely due to overfitting, so the large model is
recommended for best accuracy.

```bash
$ kraken get 10.5281/zenodo.18715373
Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 79.1/79.1 MB 0:00:00 0:00:02
Model dir: /home/mittagessen/.local/share/htrmopo/d58d541d-82cf-5ab2-b27b-a20fe2548229 (model files: ladas_m.safetensors)
$ kraken -i input.jpg out.xml -a segment -i ladas_m.safetensors
...
```
