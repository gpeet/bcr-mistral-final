# The run metadata table (metadata/2024-SRARunTable-MetadataCleaning.xlsx): one row per sequencing
# run, keyed by `Run` (SRA accessions) or `Filename` (in-house TRUST4 files).
#
# The table holds one exact duplicate row (BennettMS TRUST_MS14_08_CSFPlasmablast_R1_airr.tsv,
# Subject 6 CSF plasmablasts). A join of sequences to it would be many-to-many for that file and count
# its sequences twice, so exact duplicate rows are dropped here.
#
# Types are pinned rather than guessed, so that every notebook reads the table the same way: `guess_max`
# spans the whole table, which types the sparse columns (EDSS, Years disease) consistently, and `Age`,
# which is numeric apart from 16 rows reading "UKN", is coerced so that unknown ages are NA.
read_metadata <- function(path = data_path("metadata/2024-SRARunTable-MetadataCleaning.xlsx"), ...) {
  readxl::read_xlsx(path, guess_max = 4000, ...) %>%
    dplyr::distinct() %>%
    dplyr::mutate(Age = suppressWarnings(as.numeric(Age)))
}
