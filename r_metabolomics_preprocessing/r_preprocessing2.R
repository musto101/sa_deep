library(tidyverse)
library(ggthemes)
library(labelled)
library(caret)

# install.packages('Hmisc')
# install.packages("../adnimerge_package/ADNIMERGE_0.0.1.tar.gz", repos = NULL,
#                  type = "source")

lipid <- ADNIMERGE::admclipidomicsmeiklelablong
adni <- ADNIMERGE::adnimerge
str(lipid)


lipid <-remove_labels(lipid) 
adni <-remove_labels(adni) 

names(lipid)[4] <- 'EXAMDATE'
lipid <- lipid %>% 
  mutate(RID = as.numeric(RID), VISCODE = as.character(VISCODE),
         EXAMDATE = as.character(EXAMDATE), ORIGPROT = as.character(ORIGPROT))

adni <- adni %>% 
  mutate(RID = as.numeric(RID), VISCODE = as.character(VISCODE),
         EXAMDATE = as.character(EXAMDATE), ORIGPROT = as.character(ORIGPROT))

adni_lipid <- adni %>% 
  left_join(lipid, by = c('RID', 'VISCODE', 'EXAMDATE', 'ORIGPROT'))

# remove where age = 0
adni_without_zero <- adni_lipid[which(adni_lipid$AGE !=0),]

adni_without_zero$DX <- as.character(adni_without_zero$DX)

adni_ad <- adni_without_zero %>% mutate(DX = if_else(DX == 'Dementia', 'AD', DX))

adni_bl <- adni_ad %>% filter(VISCODE == 'bl')

write.csv(adni_bl, 'data/adni_bl2.csv')


# creating the longitudinal data

adni_last <- adni_ad %>%
  drop_na(DX) %>%
  filter(COLPROT == 'ADNI1') %>% # using ADNI2 for this project.
  group_by(PTID) %>%
  arrange(M) %>%
  summarise(M = as.numeric(last(M))) # finding the last visit - number of months.


adni_essentials <- adni_ad %>%
  filter(COLPROT == 'ADNI1') %>%
  drop_na(DX) %>%
  mutate(M = as.numeric(M)) %>%
  select(DX, PTID, M) # isolating the essential final DX information in order to do the join for last visit.

adni_last_measure <- adni_last %>%
  left_join(adni_essentials) %>%
  drop_na(DX) %>%
  mutate(last_DX = DX, last_visit = M, PTID = as.character(PTID)) %>%
  select(-DX, -M) # joining such that each p has their final diagnosis and last visit associated with their id.

adni_long <- adni_last_measure %>%
  left_join(adni_bl) %>%
  filter(COLPROT == 'ADNI1') # then joining with the last visit info table we created.

#table(adni_long$DX.bl) Sanity check
adni_long$X1 <- NULL # removing artifacts
adni_long$...1 <- NULL

adni_wo_missing <- adni_long %>%
  purrr::discard(~ sum(is.na(.x))/length(.x) * 100 >=50) # removing columns with missing data > 50%

adni_slim <- adni_wo_missing %>%
  select(-RID, -PTID, -VISCODE, -SITE, -COLPROT, -ORIGPROT, -EXAMDATE,
         -DX.bl, -FLDSTRENG, -FSVERSION, -IMAGEUID, -FLDSTRENG.bl, DX,
         -FSVERSION.bl, -Years.bl, -Month.bl, -Month, -M, ICV,
         -ends_with('.bl')) # removing unhelpful or superfluous columns

adni_slim$last_DX <- as.character(adni_slim$last_DX) # making sure last DX is a character and not a factor

adni_slim$ABETA <- NULL # removing predictors derived from csf collection via lumbar puncture.
#THIS SHOULD BE THE ONLY DIFFERENCE BETWEEN THIS AND THE OTHER PRE_PREPROCESSING SCRIPT.
adni_slim$TAU <- NULL
adni_slim$PTAU <- NULL

adni_slim$last_DX <- as.factor(adni_slim$last_DX)

write.csv(adni_slim, 'data/adni1_slim_wo_csf2.csv') # writing to csv file

# splitting into mci and cn groups

source('r_metabolomics_preprocessing/preprocessing_func2.R')

adni_slim <- read.csv('data/adni1_slim_wo_csf2.csv', stringsAsFactors = T)

#table(adni_slim$last_DX)
adni_slim$VID <- NULL
adni_slim$COHORT <- NULL
adni_slim$X <- NULL
adni_slim$SAMPLE.ID <- NULL
adni_slim$GUSPECID <- NULL
#str(adni_slim)

mci <- preprocessing(adni_slim, 0.9, clinGroup = 'MCI')

table(mci$last_DX)

write.csv(mci, 'data/mci_preprocessed_wo_csf2.csv') # writes the resultant data.frame to CSV.
#table(mci$last_DX)

# just including demographics, neuropsych tests, and lipids

mci <- read.csv('data/mci_preprocessed_wo_csf.csv')

# mci_lipids <- mci %>% 
#   select(-c("Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform",
#             "MidTemp",  "ICV", "DX.CN", "DX.MCI" ))
# 
# names(mci)
