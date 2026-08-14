#!/usr/bin/env Rscript
# Reference-implementation side of the DIFair cross-validation.
#
# Computes Mantel-Haenszel, standardization, Breslow-Day and logistic-regression
# DIF statistics with difR's own R sources, plus base R's stats::mantelhaen.test
# as a second, independent reference for the MH chi-square and common odds
# ratio.
#
# difR's core computational functions depend only on base `stats`, so they are
# sourced directly rather than installing the package and its deep import tree.
#
# Usage:
#   Rscript crossvalidate_difR.R <difR_R_dir> <data_dir> <out_csv>

args <- commandArgs(trailingOnly = TRUE)
difr_dir <- args[1]
data_dir <- args[2]
out_csv <- args[3]

for (f in c("mantelHaenszel.R", "stdPDIF.R", "breslowDay.R", "Logistik.R")) {
  suppressWarnings(source(file.path(difr_dir, f)))
}

files <- sort(list.files(data_dir, pattern = "^dataset_.*\\.csv$", full.names = TRUE))
out <- NULL

for (path in files) {
  tag <- sub("^dataset_", "", sub("\\.csv$", "", basename(path)))
  df <- read.csv(path)
  member <- df$member                       # 0 = reference, 1 = focal
  data <- as.matrix(df[, setdiff(names(df), "member")])
  J <- ncol(data)

  mh <- mantelHaenszel(data, member, correct = TRUE)
  st <- stdPDIF(data, member)
  bd <- breslowDay(data, member, BDstat = "BD")
  lg <- Logistik(data, member, type = "both", criterion = "LRT")

  # Independent base-R reference for the MH statistic, item by item.
  base_chi <- base_or <- rep(NA_real_, J)
  score <- rowSums(data, na.rm = TRUE)
  for (j in seq_len(J)) {
    keep <- rep(TRUE, length(score))
    tab <- table(
      factor(member[keep], levels = c(0, 1)),
      factor(data[keep, j], levels = c(1, 0)),
      factor(score[keep])
    )
    # Drop strata that carry no between-group information.
    good <- apply(tab, 3, function(m) all(rowSums(m) > 0) && all(colSums(m) > 0))
    tab <- tab[, , good, drop = FALSE]
    if (dim(tab)[3] >= 1) {
      tt <- try(mantelhaen.test(tab, correct = TRUE, exact = FALSE), silent = TRUE)
      if (!inherits(tt, "try-error")) {
        base_chi[j] <- unname(tt$statistic)
        base_or[j] <- unname(tt$estimate)
      }
    }
  }

  out <- rbind(out, data.frame(
    dataset = tag,
    item = colnames(data),
    difR_mh_chi2 = mh$resMH,
    difR_alpha_mh = mh$resAlpha,
    difR_var_lambda = mh$varLambda,
    difR_std_pdif = st$resStd,
    difR_bd_stat = bd$res[, 1],
    difR_logistik_chi2 = lg$stat,
    difR_logistik_delta_r2 = lg$deltaR2,
    baseR_mh_chi2 = base_chi,
    baseR_alpha_mh = base_or,
    stringsAsFactors = FALSE
  ))
  cat(sprintf("  %-28s %d items\n", tag, J))
}

write.csv(out, out_csv, row.names = FALSE)
cat("R reference values ->", out_csv, "\n")
cat("R version:", R.version.string, "\n")
