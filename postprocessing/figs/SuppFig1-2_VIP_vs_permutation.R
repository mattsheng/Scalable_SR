setwd("~/Dropbox/PAN_SR/postprocessing/figs/")
set.seed(123)
library(arrow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggsci)

# Define a function to calculate the bootstrap confidence interval using mean_cl_boot
calculate_boot_ci <- function(data, metric) {
  data %>%
    group_by(n, SNR, method) %>%
    summarize(
      boot_stats = list(mean_cl_boot(!!sym(metric), na.rm = TRUE, B=1e4)),  # Bootstrap stats as a list
      .groups = 'drop'
    ) %>%
    mutate(
      mean_value = sapply(boot_stats, function(x) as.numeric(x[1])),
      lower_CI = sapply(boot_stats, function(x) as.numeric(x[2])),
      upper_CI = sapply(boot_stats, function(x) as.numeric(x[3]))
    ) %>%
    select(-boot_stats)  # Remove the temporary list column
}

feynman_SR_plot <- function(df, xlab = "", ylab = "", title = "") {
  ggplot(df, aes(x = SNR, y = mean_value, color = method, group = method)) +
    geom_point() +
    geom_line() + 
    geom_errorbar(aes(ymin = lower_CI, ymax = upper_CI), width = 0.2) +
    labs(x = xlab, y = ylab, color = "", title = title) +
    facet_wrap(~n, scales = "fixed", ncol = 2,
               labeller = labeller(n = function(x) paste0("n = ", x))) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          axis.title = element_text(size = 14),
          axis.text = element_text(size = 12),
          plot.title = element_text(size = 16),
          legend.text = element_text(size = 12),
          legend.title = element_text(size = 14), 
          strip.text = element_text(size = 14)
          ) +
    scale_color_aaas()
}

# Load results
df_BART_permute <- read_feather("../../results_feynman/feynman_BART_perm.feather")
df_BART_VIP <- read_feather("../../results_feynman/feynman_BART_VIP_withidx.feather")

# Keep useful columns
df_BART_permute <- df_BART_permute %>%
  select(dataset_name, random_state, n, p0, p, SNR, idx_local, idx_gse, idx_gmax)
df_BART_VIP <- df_BART_VIP %>%
  select(dataset_name, random_state, n, p0, p, SNR, idx_hclst)

# Combine two dfs
df_BART_all <- df_BART_permute %>%
  left_join(df_BART_VIP, by = c("dataset_name", "random_state", "n", "SNR", "p0", "p")) %>%
  filter(n %in% c(500, 1000, 1500, 2000))
rm(df_BART_VIP, df_BART_permute)

# Convert SNR to factor to maintain the order
df_BART_all <- df_BART_all %>%
  mutate(SNR = ifelse(SNR == 0, "noiseless", as.character(SNR)))
df_BART_all$SNR <- factor(df_BART_all$SNR, levels = c("noiseless", "20", "15", "10", "5", "2", "1", "0.5"))

# Combine selected variable indices of different methods to a single column
df_BART_all <- df_BART_all %>%
  pivot_longer(cols = c(idx_local, idx_gse, idx_gmax, idx_hclst), 
               names_to = "method", 
               values_to = "idx")
df_BART_all <- df_BART_all %>%
  mutate(method = case_match(method,
                             "idx_local" ~ "Local",
                             "idx_gse" ~ "G.SE",
                             "idx_gmax" ~ "G.MAX",
                             "idx_hclst" ~ "VIP Rank"))

# Convert method to factor to maintain the order
df_BART_all$method <- factor(df_BART_all$method, levels = c("VIP Rank", "Local", "G.SE", "G.MAX"))

# Calculate TP, FP, etc.
df_BART_all$TP <- mapply(function(x, y) sum(x < y), df_BART_all$idx, df_BART_all$p0, SIMPLIFY = FALSE)
df_BART_all$TP <-  unlist(df_BART_all$TP)
df_BART_all$FP <- mapply(function(x, y) sum(x >= y), df_BART_all$idx, df_BART_all$p0, SIMPLIFY = FALSE)
df_BART_all$FP <-  unlist(df_BART_all$FP)
df_BART_all <- df_BART_all %>%
  mutate(FN = p0 - TP,
         TN = p - TP - FP - FN,
         F1 = 2*TP / (2*TP + FP + FN),
         TPR = TP / p0 * 100,
         FPR = FP / (p - p0) * 100,
         FNR = FN / p0 * 100)

# Average over `random_state`
summary_df <- df_BART_all %>%
  group_by(dataset_name, n, SNR, method) %>%
  summarize(mean_TPR = mean(TPR), 
            mean_FPR = mean(FPR),
            mean_FNR = mean(FNR),
            .groups = 'drop')

# TPR
tpr_summary <- calculate_boot_ci(summary_df, "mean_TPR")
p1 <- feynman_SR_plot(tpr_summary, xlab = "SNR", ylab = "TPR (%)", title = "True Positive Rate")
pdf(file = "suppfig1.pdf", width = 8, height = 6)
p1
dev.off()

# FPR
FPR_summary <- calculate_boot_ci(summary_df, "mean_FPR")
p2 <- feynman_SR_plot(FPR_summary, xlab = "SNR", ylab = "FPR (%)", title = "False Positive Rate")
pdf(file = "suppfig2.pdf", width = 8, height = 6)
p2
dev.off()


