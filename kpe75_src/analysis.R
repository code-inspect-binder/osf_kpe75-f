# Install packages (if you don't have them already)
#install.packages("readxl")
#install.packages("metafor") #for Meta-Analysis

## Load packages
library("readxl")
library(metafor)

## Set folder and load the data
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #if not using RStudio, set this to the current folder manually
data1 <- read_excel("data.xlsx")

## Overall effect
rma(Z_value,Z_var, data=data1, measure = "ZCOR", method = "ML")

# Regression test for funnel plot asymmetry
res <- rma(Z_value,Z_var, data=data1, measure = "ZCOR", method = "ML")
regtest(res, predictor = "vi")

# Overall effect but only for larger datasets
rma(Z_value[Num_subjects>20],Z_var[Num_subjects>20], data=data1, measure = "ZCOR", method = "ML")

## Mixed-effects models
# Main test
rma(measure = "ZCOR", Z_value, Z_var, mods = cbind(Memory, Other, Feedback_binary,
      Mean_trials_per_subject, Conf_simultaneous_with_decision, ContScale_Nchoice,
      Estimation_cetagorization), data = data1)

# Repeat to get the Memory/Other effect
rma(measure = "ZCOR", Z_value, Z_var, mods = cbind(Perception, Other, Feedback_binary,
      Mean_trials_per_subject, Conf_simultaneous_with_decision, ContScale_Nchoice,
      Estimation_cetagorization), data = data1)

# Repeat to get the effect of confidence scales above 4-point vs. 4-point or less
rma(measure = "ZCOR", Z_value, Z_var, mods = cbind(Memory, Other, Feedback_binary,
      Mean_trials_per_subject, Conf_simultaneous_with_decision, Over4_under5,
      Estimation_cetagorization), data = data1)