# Repertoires left out of patient- and donor-level analyses because they duplicate another repertoire.
# The sequence classifiers were trained with them; both copies sit in the same fold, so no sequence
# crosses between folds.
#
# Each pair shares a large share of the smaller repertoire's CDR3 amino acid sequences, against a
# median of 0.09% between unrelated donors across studies. The deeper repertoire of each pair is kept,
# counting the sequences that pass the quality filters rather than the raw assemblies. BennettMS "Subject 1" also shares 38% with BennettNMO "ON-07-05", which needs no
# entry of its own: ON-07-05 is already dropped as the DonorID typo copy below.
DUPLICATE_REPERTOIRES <- tibble::tribble(
  ~study,        ~donor,      ~reason,
  "PRJNA549712", "Patient 3", "same RRMS patient as PRJEB32763 '38 Female' (same sorted B-cell subsets in blood and CSF; 24% of its CDR3s shared); the deeper PRJEB32763 repertoire is kept",
  "BennettNMO",  "ON-07-05",  "per-index copies of donor ON07-05's gc1 library, labelled with a DonorID typo",
  "PRJNA549712", "sample 8",  "same patient as PRJNA549712 'sample 26' (both 32; they share 36% of the smaller junction set; the recorded sex differs, 'sample 8' Male and 'sample 26' Female, so one label is wrong); the deeper 'sample 26' repertoire is kept",
  "PRJNA549712", "s21",       "same patient as PRJNA549712 'sample 21' (both 21 Female; they share 22% of the smaller junction set); the deeper 'sample 21' repertoire is kept",
  "BennettNMO",  "TUM09-527", "shares 20.5% of its CDR3-aa set with BennettNMO donor ON07-05; the deeper ON07-05 repertoire is kept"
)

# drop_duplicate_repertoires(df, study, donor): `study` and `donor` name the columns holding the study
# (Author / BioProject) and the donor. The donor column may be a DonorID or a Subject string that ends
# with it ("39 Female Patient 3").
drop_duplicate_repertoires <- function(df, study, donor) {
  s <- df[[study]]
  d <- df[[donor]]
  drop <- rep(FALSE, nrow(df))
  for (i in seq_len(nrow(DUPLICATE_REPERTOIRES))) {
    x <- DUPLICATE_REPERTOIRES[i, ]
    drop <- drop | (!is.na(s) & s == x$study & !is.na(d) & (d == x$donor | endsWith(d, paste0(" ", x$donor))))
  }
  df[!drop, , drop = FALSE]
}
