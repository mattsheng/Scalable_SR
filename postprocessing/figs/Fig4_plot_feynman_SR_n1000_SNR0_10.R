setwd("~/Dropbox/PAN_SR/postprocessing/figs/")
set.seed(123)
library(arrow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggsci)

feynman_SR_plot <- function(df, metric, xlab = "", ylab = "", title = "") {

  aaas_colors <- ggsci::pal_aaas()(2)  # 2 unique colors for 2 vs_methods
  df$combined <- interaction(df$vs_method, df$SNR, sep = ", ")
  
  # Define shape and color values for each combination of vs_method and SNR
  shape_values <- c(1, 2, 1, 2)  # Circle and triangle for both methods
  color_values <- c(aaas_colors[1], aaas_colors[1], aaas_colors[2], aaas_colors[2])
  
  p <- ggplot(df, aes(x = !!sym(metric), y = algorithm, color = combined, shape = combined)) +
    labs(x = xlab, y = ylab, color = "", shape = "", title = title) +
    theme_minimal() +
    theme(axis.title = element_text(size = 14),
          axis.text = element_text(size = 12),
          plot.title = element_text(size = 16),
          legend.text = element_text(size = 10),
          legend.title = element_text(size = 14),
          strip.text = element_text(size = 14)
    ) +
    scale_color_manual(values = color_values) +
    scale_shape_manual(values = shape_values) +
    guides(shape = guide_legend(override.aes = list(size = 1)))
  
  p <- p + 
    stat_summary(aes(x = !!sym(metric), y = algorithm),
                 fun = mean, fun.data = mean_cl_boot,
                 fun.args = list("B" = 1e4),
                 position = position_dodge(width = 0.5),
                 na.rm = TRUE, size = 0.8)
  
  return(p)
}

# Load data
df_feynman <- read_feather("../../results_feynman/feynman_results.feather")

# Only analyze n=1000 and SNR=0,10 case
df_feynman <- df_feynman %>%
  filter(n == 1000, SNR %in% c(0, 10), algorithm %in% c("Operon", "AFP", "GP-GOMEA", "DSR", "EHC", "uDSR", "FEAT"))

# Rename selection method
df_feynman$vs_method[df_feynman$vs_method == "none"] <- "SR"
df_feynman$vs_method[df_feynman$vs_method == "hclst_v2"] <- "PAN+SR"

# Extract BART runtime
df_BART <- read_feather("../../results_feynman/feynman_BART_VIP_withidx.feather")
df_BART <- df_BART %>%
  filter(n == 1000, SNR %in% c(0, 10))
df_BART <- df_BART %>% select(dataset_name, random_state, SNR, time_time)
colnames(df_BART) <- c("dataset_name", "random_state", "SNR", "BART_time")

# Merge BART runtime with df_feynman
df_feynman <- df_feynman %>% 
  left_join(df_BART, by = c("dataset_name", "random_state", "SNR"))
df_feynman$total_time <- ifelse(df_feynman$vs_method == "PAN+SR", df_feynman$time_time + df_feynman$BART_time, df_feynman$time_time)

# Change SNR 0 to noiseless
df_feynman <- df_feynman %>%
  mutate(SNR = ifelse(SNR == 0, "noiseless", as.character(SNR)))

# Convert SNR to factor to maintain the order
df_feynman$SNR <- factor(df_feynman$SNR, levels = c("noiseless", "10"), 
                         labels = c("noiseless", "SNR=10"))

# Calculate TPR, FPR, FNR
df_feynman <- df_feynman %>% 
  mutate(TPR = TP / p0 * 100,
         FPR = FP / (p - p0) * 100,
         FNR = FN / p0 * 100)

# If simplified complexity is NA, fill with original complexity
df_feynman <- df_feynman %>% mutate(complexity_simplified = coalesce(complexity_simplified, model_size))

# Either is correct solution
df_feynman$solution <- (df_feynman$constant_diff | df_feynman$constant_ratio)
df_feynman <- df_feynman %>%
  mutate(solution = replace_na(solution, FALSE))
df_feynman$solution <- ifelse(df_feynman$solution, 1, 0) * 100 # turn into percentage

# Fail rate
df_feynman$failed <- ifelse(df_feynman$failed == "no error", 0, 1) * 100

# Average over `random_state`
summary_feynman <- df_feynman %>%
  group_by(dataset_name, algorithm, vs_method, SNR) %>%
  summarize(mean_r2 = mean(r2_zero_test, na.rm = TRUE),
            mean_rmse = mean(sqrt(mse_test), na.rm = TRUE),
            mean_mae = mean(mae_test, na.rm = TRUE),
            mean_time = mean(total_time, na.rm = TRUE),
            mean_TPR = mean(TPR, na.rm = TRUE),
            mean_FPR = mean(FPR, na.rm = TRUE),
            mean_FNR = mean(FNR, na.rm = TRUE),
            mean_F1 = mean(F1, na.rm = TRUE),
            mean_complexity = mean(complexity_simplified, na.rm = TRUE),
            .groups = 'drop') %>%
  group_by(algorithm, vs_method, SNR) 

# Average over `dataset_name`
summary_feynman_compact <- summary_feynman %>%
  group_by(algorithm, vs_method, SNR) %>%
  summarize(
    mean_r2 = mean(mean_r2, na.rm = TRUE),
    mean_rmse = mean(mean_rmse, na.rm = TRUE),
    mean_mae = mean(mean_mae, na.rm = TRUE),
    mean_time = mean(mean_time, na.rm = TRUE),
    mean_TPR = mean(mean_TPR, na.rm = TRUE),
    mean_FPR = mean(mean_FPR, na.rm = TRUE),
    mean_FNR = mean(mean_FNR, na.rm = TRUE),
    mean_F1 = mean(mean_F1, na.rm = TRUE),
    mean_complexity = mean(mean_complexity, na.rm = TRUE),
    .groups = 'drop'
  )

# Create a variable to order methods by their average R-squared in SR mode
order_feynman <- summary_feynman_compact %>%
  filter(vs_method == "SR", SNR == "noiseless") %>%
  arrange(desc(mean_r2)) %>%
  pull(algorithm)

# Reorder the method factor based on SR's mean R-squared
summary_feynman <- summary_feynman %>%
  mutate(algorithm = factor(algorithm, levels = rev(order_feynman)))
df_feynman <- df_feynman %>%
  mutate(algorithm = factor(algorithm, levels = rev(order_feynman)))

# R2 test
p_r2 <- feynman_SR_plot(df = summary_feynman, metric = "mean_r2", title = expression(R^2~"Test"))

# Solution rate
p_solu_rate <- feynman_SR_plot(df_feynman, metric = "solution", title = "Solution Rate (%)")
p_solu_rate <- p_solu_rate + theme(axis.title.y = element_blank(), axis.text.y = element_blank())

combined_plot <- (p_r2 | plot_spacer() | p_solu_rate) +
  plot_layout(guides = 'collect', width = c(1, 0.1, 1)) &
  theme(legend.position = "bottom",
        legend.margin = margin(t = -15, l = -45, unit = "pt"),
        plot.margin = margin(t = 5, r = 5, b = 5, l = -10, unit = "pt"))
pdf(file = "fig4.pdf", width = 6.5, height = 5)
combined_plot
dev.off()




