#' preprocessing from adni_slim data
#'
#' preprocesses the adni_slim data by removing columns with perc missing values
#' and filters on the relevant clinical group clinGroup. Also dummies and records
#' location of missing values.
#'
#' @param dat is a dataframe
#' @param perc is the percentage of missing values, where if
#' column missing > perc, column is removed.
#' @param clinGroup is the clinical group desired to filter on.
#' @return It returns a dataframe of the preprocessed data
#' @export
#'
preprocessing <- function(dat, perc, clinGroup) {

  dat$last_DX <- as.factor(dat$last_DX)

  # missing.perc <- apply(dat, 2, function(x) sum(is.na(x))) / nrow(dat) # calculating missing perc
  # 
  # dat <- dat[, which(missing.perc < perc)] # removing columns where missingness % > perc. in ADNISurvivalProject this is just a failsafe and should have already been done.
  # dat <- dat[dat$PTMARRY != 'Unknown',] # removing rows where marital status = 'Unknown'
  # dat <- dat[dat$last_visit > 0,] # Removing rows where last visit <= 0
  # 
  # y <- dat %>%
  #   mutate_all(funs(ifelse(is.na(.), 1, 0))) # created a bunch of bool cols indicating the presence and location of NA values in dat.
  # 
  # blah <- y %>%
  #   select_if(function(col) is.numeric(col) && sum(col) == 0) # sanity check
  # 
  # y <- y %>%
  #   select_if(negate(function(col) is.numeric(col) && sum(col) == 0)) # selecting only columns that indicate at least one na value
  # 
  # names(y) <- paste0(names(y), '_na') # affixing the suffix _na to denote that they are na indicator cols.
  # 
  # dat <- cbind(dat, y) # affixing the na indicator columns to the original dataset.

  dummies <- dummyVars(last_DX ~., data = dat) # training the dummy variables model. done before splitting the data, as recommended here https://stats.stackexchange.com/questions/355293/creating-dummy-variables-before-or-after-splitting-to-train-test-datasets
  data_numeric <- predict(dummies, newdata= dat) # changing nominal to dummy in the original dataset
  data_numeric <- as.data.frame(data_numeric) # ensuring the data is in a data.frame format.
  data_numeric <-data.frame(dat$last_DX, data_numeric) # binding the outcome to the dummied data.
  names(data_numeric)[1] <- 'last_DX' # ensuring the outcome is called 'last_DX'
  data_numeric$X <- NULL # removing artifact.

  ## The below if statement checks the user defined variable clinGroup and isolates the relevant clinical group and standardises the outcome to a binary format.
  if (clinGroup == 'CN') {

    cn_progress <- data_numeric[data_numeric$DX.CN == 1,]
    cn_progress$last_DX <- factor(ifelse(cn_progress$last_DX == 'CN',
                                         'CN', 'MCI_AD'),
                                  levels = c('CN', 'MCI_AD'))

  } else if (clinGroup == 'MCI') {

    cn_progress <- data_numeric[data_numeric$DX.MCI == 1,]
    cn_progress$last_DX <- factor(ifelse(cn_progress$last_DX == 'AD',
                                         'Dementia', 'CN_MCI'),
                                  levels = c('CN_MCI', 'Dementia'))



  } else {
    stop('clinGroup needs to be either CN or MCI. Please try again.')
  }

  table(cn_progress$last_DX) # sanity check

  cn_progress$DXCN <- NULL # removes all indicators of baseline diagnosis.
  cn_progress$DXDementia <- NULL
  cn_progress$DXMCI <- NULL

  return(cn_progress) # returns the reshaped data.frame
}
