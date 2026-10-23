# BCR-Mistral: TRUST4 assemblies of MS and control repertoires

B and T cell receptor assemblies behind *Language Model Embedding Classifiers Enable Identification of
Multiple Sclerosis-Associated BCRs and Repertoires* (Peet, Owens, Bennett, Krishnan, Macklin; citation to
be added). The processed data are in this same record (see `README.md`); the code is at
https://github.com/gpeet/bcr-mistral-final.

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The sequence data are released
openly, in keeping with the NIH data-sharing policy under which the in-house repertoires are public.
The code and the model weights are released separately under more restrictive terms: all rights
reserved, pending a licence. See the code repository's LICENSE.

## Files

| File | Contents |
|---|---|
| `trust4-sra.tar.gz` | 3,090 TRUST4 AIRR files (`TRUST_<run>_airr.tsv`), one per NCBI SRA run |
| `trust4-inhouse.tar.gz` | 216 TRUST4 AIRR files from in-house samples (Bennett lab, University of Colorado) |
| `sra-files.txt`, `inhouse-files.txt` | File lists of the two archives |

798 files contain a header only: TRUST4 assembled no receptors from those runs. Sample metadata (disease,
tissue, age, sex, treatment) is in `2024-SRARunTable-MetadataCleaning.xlsx` in this record.

## How they were made

Each run was downloaded from SRA (sra-tools 3.4.1) and assembled with TRUST4 v1.1.5. Most runs used
standard TRUST4 on the raw reads. Two runs that did not finish were rerun after fastp quality filtering and
deduplication with `--repseq`. Four 5' single-cell runs (PRJNA549712) were assembled with barcode/UMI-aware
TRUST4 (`--readFormat bc:0:15,um:16:26`). The per-run modes are in `dataset/trust4_run_modes.csv`, and
`dataset/03_trust4_sra.sh` in the code repository reproduces any run.

`scripts/fetch_data.sh trust4` downloads, verifies and unpacks both archives.
