# Contrasting Neural Click Models and Pointwise IPS Rankers
Source code for our paper `Contrasting Neural Click Models and Pointwise IPS Rankers`.

## Data
The project automatically downloads public datasets to `~/.ltr-datasets/` on first execution.

## Installation
1. Install dependencies using conda: `conda env create -f environment.yaml`
2. Activate environment: `conda activate ultr-cm-vs-ips`
3. Run experiments in the `/scripts directory`, e.g.: `./scripts/mslr_dataset_size.sh`

## Reference
```
@inproceedings{Hager2023Contrasting,
  author    = {Philipp Hager and Maarten de Rijke and Onno Zoeter},
  title     = {Contrasting Neural Click Models and Pointwise IPS Rankers},
  booktitle = {ECIR 2023: 45th European Conference on Information Retrieval},
  publisher = {Springer},
  year      = {2023},
}
```
## License
This project uses the [MIT license]().
