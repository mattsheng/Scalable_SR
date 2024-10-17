setwd("~/Dropbox/PAN_SR/postprocessing/figs/")
set.seed(123)
library(arrow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggsci)

# Define a function to calculate the bootstrap confidence interval using mean_cl_boot
calculate_boot_ci <- function(data, metric) {
  data %>%
    group_by(n, SNR, vs_method) %>%
    summarize(
      boot_stats = list(mean_cl_boot(!!sym(metric), na.rm = TRUE, B=1e4)),
      .groups = 'drop'
    ) %>%
    mutate(
      mean_value = sapply(boot_stats, function(x) as.numeric(x[1])),
      lower_CI = sapply(boot_stats, function(x) as.numeric(x[2])),
      upper_CI = sapply(boot_stats, function(x) as.numeric(x[3]))
    ) %>%
    select(-boot_stats)
}

feynman_operon_plot <- function(df, xlim = NULL, xlab = "SNR", ylab = "", title = "") {
  ggplot(df, aes(x = SNR, y = mean_value, color = vs_method, group = vs_method)) +
    geom_point() +
    geom_line() + 
    geom_errorbar(aes(ymin = lower_CI, ymax = upper_CI), width = 0.2) +
    labs(title = title, x = xlab, y = ylab, color = "") +
    coord_cartesian(xlim = xlim) +
    theme_minimal() +
    facet_wrap(~n, scales = "fixed", ncol = 2, 
               labeller = labeller(n = function(x) paste0("n = ", x))) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          axis.title = element_text(size = 14),
          axis.text = element_text(size = 12),
          plot.title = element_text(size = 16),
          legend.text = element_text(size = 12),
          legend.title = element_text(size = 14),
          strip.text = element_text(size = 14),
          legend.position = "bottom",
          legend.margin = margin(t = -5, unit = "pt")
    ) +
    scale_color_aaas()
}

# Load data
df_SR <- read_feather("../../results_feynman/feynman_results.feather")
df_BART <- read_feather("../../results_feynman/feynman_BART_VIP_withidx.feather")

# Keep only Operon for analysis
df_SR <- df_SR %>%
  filter(algorithm == "Operon")

# Rename selection method
df_SR$vs_method[df_SR$vs_method == "none"] <- "SR"
df_SR$vs_method[df_SR$vs_method == "hclst_v2"] <- "PAN+SR"
df_BART$vs_method <- "PAN"

# Extract BART runtime
df_BART_time <- df_BART %>% select(dataset_name, random_state, time_time, SNR, n)
colnames(df_BART_time) <- c("dataset_name", "random_state", "BART_time", "SNR", "n")

# Merge BART runtime with df_SR
df_SR <- df_SR %>% 
  left_join(df_BART_time, by = c("dataset_name", "random_state", "SNR", "n"))
df_SR$total_time <- ifelse(df_SR$vs_method == "PAN+SR", df_SR$time_time + df_SR$BART_time, df_SR$time_time)

# If simplified complexity is NA, fill with original complexity
df_SR <- df_SR %>% mutate(complexity_simplified = coalesce(complexity_simplified, model_size))

# Either is correct solution
df_SR$solution <- (df_SR$constant_diff | df_SR$constant_ratio)
df_SR <- df_SR %>%
  mutate(solution = replace_na(solution, FALSE))
df_SR$solution <- ifelse(df_SR$solution, 1, 0)

# Change SNR 0 to noiseless
df_BART <- df_BART %>%
  mutate(SNR = ifelse(SNR == 0, "noiseless", as.character(SNR)))
df_SR <- df_SR %>%
  mutate(SNR = ifelse(SNR == 0, "noiseless", as.character(SNR)))

# Convert SNR to factor to maintain the order
df_BART$SNR <- factor(df_BART$SNR, levels = c("noiseless", "20", "15", "10", "5", "2", "1", "0.5"))
df_SR$SNR <- factor(df_SR$SNR, levels = c("noiseless", "20", "15", "10", "5", "2", "1", "0.5"))

# Find TP
df_BART$TP <- mapply(function(x, y) sum(x < y), df_BART$idx_hclst, df_BART$p0, SIMPLIFY = FALSE)
df_BART$TP <- unlist(df_BART$TP)

# Find FP
df_BART$FP <- mapply(function(x, y) sum(x >= y), df_BART$idx_hclst, df_BART$p0, SIMPLIFY = FALSE)
df_BART$FP <- unlist(df_BART$FP)

# Find FN, TN, F1, TPR, FPR
df_BART <- df_BART %>% 
  mutate(FN = p0 - TP,
         TN = p - TP - FP - FN,
         F1 = 2*TP / (2*TP + FP + FN),
         TPR = TP / p0 * 100,
         FPR = FP / (p - p0) * 100,
         FNR = FN / p0 * 100)
df_SR <- df_SR %>% 
  mutate(TPR = TP / p0 * 100,
         FPR = FP / (p - p0) * 100,
         FNR = FN / p0 * 100)

df <- bind_rows(df_SR, df_BART)
df <- subset(df, n %in% c(500, 1000, 1500, 2000))
df$vs_method <- factor(df$vs_method, levels = c("PAN+SR", "SR", "PAN"))

# Summarize data
summary_df <- df %>%
  group_by(dataset_name, n, SNR, vs_method) %>%
  summarize(mean_TPR = mean(TPR, na.rm = TRUE), 
            mean_FPR = mean(FPR, na.rm = TRUE),
            mean_FNR = mean(FNR, na.rm = TRUE),
            mean_FP = mean(FP, na.rm = TRUE),
            mean_F1 = mean(F1, na.rm = TRUE),
            mean_r2 = mean(r2_zero_test, na.rm = TRUE),
            mean_time = mean(total_time, na.rm = TRUE),
            mean_complexity = mean(complexity_simplified, na.rm = TRUE),
            .groups = 'drop')

# FPR
fpr_summary <- calculate_boot_ci(summary_df, "mean_FPR")
p_fpr <- feynman_operon_plot(fpr_summary, ylab = "FPR (%)", title = "False Positive Rate")
pdf(file = "fig3a.pdf", width = 8, height = 6)
p_fpr
dev.off()

# FNR
fnr_summary <- calculate_boot_ci(summary_df, "mean_FNR")
p_fnr <- feynman_operon_plot(fnr_summary, ylab = "FNR (%)", title = "False Negative Rate")
pdf(file = "fig3b.pdf", width = 8, height = 6)
p_fnr
dev.off()

# R2
summary_df_r2 <- summary_df %>%
  filter(vs_method != "PAN")
r2_summary <- calculate_boot_ci(summary_df_r2, "mean_r2")
p_r2 <- feynman_operon_plot(r2_summary, ylab = expression(R^2), title = expression(R^2~"Test"))
pdf(file = "suppfig5a.pdf", width = 8, height = 6)
p_r2
dev.off()

# Complexity
complexity_summary <- calculate_boot_ci(summary_df_r2, "mean_complexity")
p_complexity <- feynman_operon_plot(df = complexity_summary, ylab = "Model Size", title = "Model Size") 
pdf(file = "suppfig5b.pdf", width = 8, height = 6)
p_complexity
dev.off()

