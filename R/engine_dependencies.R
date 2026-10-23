# Packages that are loaded by name at run time, which renv's dependency scan cannot see.
# Listing them here keeps them in renv.lock. This file is never sourced.

# parsnip model engines, chosen with set_engine("ranger") etc.
library(glmnet)
library(ranger)
library(xgboost)

# ggsave() writes .svg figures through svglite
library(svglite)
