# H3-OPT

A method for CDR-H3 optimization

## Getting started with this repo 

## Renumbering

H3-OPT requires installation of AbRSA, you can download it through the  [AbRSA](http://aligncdr.labshare.cn/aligncdr/download.html) website.

### Install 

To install H3-OPT, we provide the conda environment of H3-OPT, run the following command: :

```
# conda create --name <env> --file requirements.txt
```

### Schrödinger Python API

The CDR-H3 selection module and structure refinement methods are implemented by the Schrödinger Python API, you can assess  Schrödinger modules following the instructions [here](https://www.schrodinger.com/sites/default/files/s3/public/python_api/2023-2/intro.html#getting-started).

### Template database

We provide the template CDR-H3 database from SAbDab website,. These template structures are made available for use [online](https://huggingface.co/datasets/chdcg/H3-OPT_template_dataset/resolve/main/template.zip). Please unzip this compressed file before using.

## Usage

### Template module

We provide a command line interface that effectively figure out the high confidence CDR-H3 loop by the CBM and graft the template loop onto models prediction by AlphaFold2 using the TGM.

```
usage: selection.py [-h] [--input_structure_dir INPUT_STRUCTURE_DIR]
                    [--output_structure_dir OUTPUT_STRUCTURE_DIR]
                    [--pdbname PDBNAME] [--tmp_dir TMP_DIR]
                    [--template_dir TEMPLATE_DIR] [--cutoff CUTOFF]

optional arguments:
  -h, --help            show this help message and exit
  --input_structure_dir INPUT_STRUCTURE_DIR
                        Path to input PDB file, please use PDB format files as
                        inputs
  --output_structure_dir OUTPUT_STRUCTURE_DIR
                        Path to output PDB directory
  --pdbname PDBNAME     Pdbname of input PDB file
  --tmp_dir TMP_DIR     Path to renumbering files
  --template_dir TEMPLATE_DIR
                        Path to CDR-H3 template files
  --cutoff CUTOFF       specify the cutoff of high confidence
```

### PLM-based structure prediction module

#### Feature extraction

To predict the CDR-H3 loops of input AF2 models, you can run the following command to extract the residue-level features and pair representations of input models.

```
usage: data_prep.py [-h] [--input_structure_dir INPUT_STRUCTURE_DIR]
                    [--feature_dir FEATURE_DIR] [--tmp_dir TMP_DIR]
                    [--pdbname PDBNAME]

optional arguments:
  -h, --help            show this help message and exit
  --input_structure_dir INPUT_STRUCTURE_DIR
                        Path to input PDB file, please use PDB format files as
                        inputs
  --feature_dir FEATURE_DIR
                        Path to output feature directory
  --tmp_dir TMP_DIR     Path to renumbering files
  --pdbname PDBNAME     Pdbname of input PDB file
```

#### Model prediction

To directly predict the 3D coordinates of CDR-H3 loops, we provide the weight of H3-OPT [online](https://huggingface.co/chdcg/H3-OPT/blob/main/best_wt.pth). You can specify the path to model weight file and obtain the csv file which contains the coordinates of all Cα atoms in H3 loop.

```
usage: predict.py [-h] [--feature_dir FEATURE_DIR] [--model_dir MODEL_DIR]
                  [--out_dir OUT_DIR] [--out_name OUT_NAME]
                  [--pdbname PDBNAME]

optional arguments:
  -h, --help            show this help message and exit
  --feature_dir FEATURE_DIR
                        Path to feature files
  --model_dir MODEL_DIR
                        Path to model directory
  --out_dir OUT_DIR     Path to output PDB directory
  --out_name OUT_NAME   filename of predicted cordinate files
  --pdbname PDBNAME     Pdbname of input PDB file
```

#### Structure generation

You can optimize the conformation of input AlphaFold2 model CDR-H3 loop by running the following  command lines.

```
usage: structure_generation.py [-h]
                               [--input_structure_dir INPUT_STRUCTURE_DIR]
                               [--tmp_dir TMP_DIR]
                               [--output_structure_dir OUTPUT_STRUCTURE_DIR]
                               [--pdbname PDBNAME] [--pred_csv PRED_CSV]

optional arguments:
  -h, --help            show this help message and exit
  --input_structure_dir INPUT_STRUCTURE_DIR
                        Path to input PDB file, please use PDB format files as
                        inputs
  --tmp_dir TMP_DIR     Path to renumbering files
  --output_structure_dir OUTPUT_STRUCTURE_DIR
                        Path to output PDB directory
  --pdbname PDBNAME     Pdbname of input PDB file
  --pred_csv PRED_CSV   filename of predicted cordinate files
```

#### Confidence score

You can get a confidence score of our prediction using following command lines.
```
usage: pred_confidence_score.py [-h] [--feature_dir FEATURE_DIR] [--model_dir MODEL_DIR]
                  [--out_dir OUT_DIR] [--out_name OUT_NAME]
                  [--pdbname PDBNAME]

optional arguments:
  -h, --help            show this help message and exit
  --feature_dir FEATURE_DIR
                        Path to feature files
  --model_dir MODEL_DIR
                        Path to model directory
  --out_dir OUT_DIR     Path to output PDB directory
  --out_name OUT_NAME   filename of predicted cordinate files
  --pdbname PDBNAME     Pdbname of input PDB file
```
