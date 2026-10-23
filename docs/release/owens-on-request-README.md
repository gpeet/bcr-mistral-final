# BCR-Mistral: Owens et al. antibody sequences (on request)

Heavy-chain sequences of the CSF plasmablast antibodies scored in Fig. 3D of *Language Model Embedding
Classifiers Enable Identification of Multiple Sclerosis-Associated BCRs and Repertoires* (Peet, Owens,
Bennett, Krishnan, Macklin; citation to be added), with the 2B4 isotype control. These sequences are not
public; they are shared on request. Contact Graham Peet (graham.peet@cuanschutz.edu).

| File | Contents |
|---|---|
| `owens_sequences.csv.gz` | IgBLAST-annotated, V-J clipped productive IGH sequences (`workflow/run.sh owens-sequences`) |
| `owens-fasta.tar.gz` | The source FASTA files (`MS Repertoire Fasta Files/`) |
| `SHA256SUMS` | Checksums |

## Use

Put the files you received in one folder and run, from the code repository,
`OWENS_BUNDLE=/path/to/folder scripts/fetch_data.sh owens`. This checks the sums and unpacks them into
`$BCRM_DATA/owens/`; the `embed owens`, `owens` and `figures` stages then make Fig. 3D.
