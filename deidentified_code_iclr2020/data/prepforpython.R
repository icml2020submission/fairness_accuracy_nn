library(dplyr)
library(tidyverse)
library(ggplot2)
raw_data <- read.csv("./compas-scores-two-years.csv")
nrow(raw_data)

# Propublica scrubbing
#   If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
#   We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
#   In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
#   We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.

df <- dplyr::select(raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out, 
                    juv_other_count, juv_fel_count, juv_misd_count) %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A')
nrow(df)

# Also combined the covariate choices of Johndrow and Lum
# age, prior_count, jov_other_count, juv_fel_count, juv_misd_count, sex
# protected variable race
# two year recid is target

dfJohndrowLum <- select(df, age, priors_count, juv_other_count, juv_fel_count, juv_misd_count, sex, race, two_year_recid)
nrow(dfJohndrowLum)

write_csv(dfJohndrowLum, path = "compas-scores-two-years-JohndrowLum.csv")

## only do propublica scrubbing but keep more columns in raw_data

scrubbed_morecovariate  = filter(raw_data, days_b_screening_arrest <= 30, 
         days_b_screening_arrest >= -30, is_recid != -1, 
         c_charge_degree != "O", score_text != 'N/A')
scrubbed_morecovariate$length_of_stay <- as.numeric(as.Date(scrubbed_morecovariate$c_jail_out) - as.Date(scrubbed_morecovariate$c_jail_in))

scrubbed_morecovariate  = select(scrubbed_morecovariate, age, c_charge_degree, age_cat, sex, priors_count, 
                                 days_b_screening_arrest, length_of_stay, c_days_from_compas,
                                 juv_other_count, juv_fel_count, juv_misd_count, race, two_year_recid)
write_csv(scrubbed_morecovariate, path = "compas-scores-two-years-JohndrowLumMore.csv")
