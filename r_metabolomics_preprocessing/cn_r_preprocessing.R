library(tidyverse)
library(ggthemes)
library(labelled)
library(caret)

# install.packages('Hmisc')
# install.packages("../adnimerge_package/ADNIMERGE_0.0.1.tar.gz", repos = NULL,
#                  type = "source")

source('r_metabolomics_preprocessing/preprocessing_func.R')

adni_slim <- read.csv('data/adni1_slim_wo_csf.csv', stringsAsFactors = T)

#table(adni_slim$last_DX)
adni_slim$VID <- NULL
adni_slim$COHORT <- NULL
adni_slim$X <- NULL
adni_slim$SAMPLE.ID <- NULL
#str(adni_slim)

str(adni_slim)

adni_slim$last_DX <- as.factor(adni_slim$last_DX)

cn <- preprocessing(adni_slim, 0.9, clinGroup = 'CN')

table(cn$last_DX)

write.csv(cn, 'data/cn_preprocessed_wo_csf.csv') # writes the resultant data.frame to CSV.
#table(mci$last_DX)

# just including demographics, neuropsych tests, and lipids

# mci <- read.csv('data/mci_preprocessed_wo_csf.csv')

# mci_lipids <- mci %>% 
#   select(-c("Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform",
#             "MidTemp",  "ICV", "DX.CN", "DX.MCI" ))
# 
# names(mci)
