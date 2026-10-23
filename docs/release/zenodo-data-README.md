# BCR-Mistral: processed data

Data behind *Language Model Embedding Classifiers Enable Identification of Multiple Sclerosis-Associated
BCRs and Repertoires* (Peet, Owens, Bennett, Krishnan, Macklin; citation to be added). With the code
(https://github.com/gpeet/bcr-mistral-final) and the model weights
(https://huggingface.co/grpeet/bcr-mistral),
these files regenerate the computational figure panels, except Fig. 3D, whose antibody sequences
(Owens et al.) are shared on request (graham.peet@cuanschutz.edu), and Fig. 1E, Table S1 and 1I, which
also need the TRUST4 assemblies.

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The sequence data are released
openly, in keeping with the NIH data-sharing policy under which the in-house repertoires are public.
The code and the model weights are released separately under more restrictive terms: all rights
reserved, pending a licence. See the code repository's LICENSE.

This record holds the processed data and the TRUST4 assemblies together, both under CC BY 4.0. The
model weights are not here: they are on the Hugging Face Hub, under the code repository's terms.

## Use

Zenodo stores files without folders. `scripts/fetch_data.sh data` in the code repository downloads them,
checks `SHA256SUMS` and unpacks them into the expected layout (`docs/DATA_LAYOUT.md`). By hand: download
the files, put each in the folder that `SHA256SUMS` gives, then `gunzip` them.

## Files

| File | Contents |
|---|---|
| `2024-SRARunTable-MetadataCleaning.xlsx.gz` | Curated metadata for every repertoire: study, disease, tissue, age, sex, treatment, SRA run |
| `allOPIGdefinedUnpairedMetadata.csv.gz` | Metadata of the OAS (OPIG) unpaired studies used for pretraining |
| `classifier_data_igblast-{train,test}-weights.csv.gz` | Labelled IGH sequences for the sequence classifiers (two outer folds), IgBLAST annotation, sample weights |
| `treatment_sequences_{train,test}.csv.gz` | Pre/post-treatment repertoires (Fig. 3B, S2) |
| `triplet_embedding_test_set_igblast.csv.gz` | Test set for the embedding comparison (Fig. 2C, S1A) |
| `diverse_foundation_data.csv.gz` | Pretraining corpus: diversity-selected OAS IGH sequences |
| `undiversified_foundation_data.csv.gz` | Size-matched random sample of the same pool (ablation) |
| `diverse_foundation_data_protein.csv.gz` | The diverse corpus translated to amino acids (protein model) |
| `junction.csv.gz` | CDR3 junction corpus (junction model) |
| `classifier_data_igh_sequence-val.csv.gz` | Validation sequences for the model-size comparison (Fig. 2B) |
| `preliminary-trust4-quality-sequences.csv.gz` | TRUST4 IGH sequences with consensus count > 2 and repertoire metadata (Fig. 1F-H) |
| `all_high_quality_for_igblast.csv.gz`, `all_hq_igblast_realigned.csv.gz` | Intermediates of the classifier dataset build |
| `comparator-output.tar.gz` | Predictions of the k-mer and protein baseline classifiers (GLM, random forest) compared in Fig. 2E, 2F and S1B |

Repertoires come from public NCBI SRA studies (run accessions in the metadata table) and from in-house
samples of the Bennett lab, University of Colorado Anschutz Medical Campus. The TRUST4 assemblies they
were built from are in this record too, as `trust4-sra.tar.gz` and `trust4-inhouse.tar.gz`;
`README-trust4.md` describes them.
