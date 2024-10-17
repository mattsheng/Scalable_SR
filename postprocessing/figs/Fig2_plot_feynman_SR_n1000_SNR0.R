setwd("~/Dropbox/PAN_SR/postprocessing/figs/")
set.seed(123)
library(arrow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggsci)

feynman_SR_plot <- function(df, metric, xlab = "", ylab = "", title = "") {
  xlim <- if (metric == "mean_r2") c(0, 1) else NULL
  ggplot(df, aes(x = !!sym(metric), y = algorithm, color = vs_method)) +
    stat_summary(fun = mean, fun.data = mean_cl_boot, # Points with bootstrap CI
                 fun.args = list("B" = 1e4),
                 position = position_dodge(width = 0.5),
                 na.rm = TRUE) +  
    labs(x = xlab, y = ylab, color = "", title = title) +
    coord_cartesian(xlim = xlim) +
    theme_minimal() +
    theme(axis.title = element_text(size = 14),
          axis.text = element_text(size = 12),
          plot.title = element_text(size = 16),
          legend.text = element_text(size = 12),
          legend.title = element_text(size = 14),
          strip.text = element_text(size = 14)
    ) +
    scale_color_aaas()
}

# Load data
df_feynman <- read_feather("../../results_feynman/feynman_results.feather")

# Only analyze n=1000 and SNR=0 case
df_feynman <- df_feynman %>%
  filter(n == 1000, SNR == 0)

# Rename selection method
df_feynman$vs_method[df_feynman$vs_method == "none"] <- "SR"
df_feynman$vs_method[df_feynman$vs_method == "hclst_v2"] <- "PAN+SR"

# Extract BART runtime
df_BART <- read_feather("../../results_feynman/feynman_BART_VIP_withidx.feather")
df_BART <- df_BART %>%
  filter(n == 1000, SNR == 0)
df_BART <- df_BART %>% select(dataset_name, random_state, time_time)
colnames(df_BART) <- c("dataset_name", "random_state", "BART_time")

# Merge BART runtime with df_feynman
df_feynman <- df_feynman %>% 
  left_join(df_BART, by = c("dataset_name", "random_state"))
df_feynman$total_time <- ifelse(df_feynman$vs_method == "PAN+SR", df_feynman$time_time + df_feynman$BART_time, df_feynman$time_time)

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

# Average over `random_state`
summary_feynman <- df_feynman %>%
  group_by(dataset_name, algorithm, vs_method) %>%
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
  group_by(algorithm, vs_method) 

# Average over `dataset_name`
summary_feynman_compact <- summary_feynman %>%
  group_by(algorithm, vs_method) %>%
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
  filter(vs_method == "SR") %>%
  arrange(desc(mean_r2)) %>%
  pull(algorithm)

# Reorder the method factor based on SR's mean R-squared
summary_feynman <- summary_feynman %>%
  mutate(algorithm = factor(algorithm, levels = rev(order_feynman)))
df_feynman <- df_feynman %>%
  mutate(algorithm = factor(algorithm, levels = rev(order_feynman)))

# R2 test
p_r2 <- feynman_SR_plot(df = summary_feynman, metric = "mean_r2", title = expression(R^2~"Test"))

# Training time
p_runtime <- feynman_SR_plot(df = summary_feynman, metric = "mean_time", title = "Training Time (s)") 

# Complexity
p_complexity <- feynman_SR_plot(df = summary_feynman, metric = "mean_complexity", title = "Model Size") 

# Solution rate
p_solu_rate <- feynman_SR_plot(df_feynman, metric = "solution", title = "Solution Rate (%)")

# Modify the second plot to remove y-axis text and ticks
p_runtime <- p_runtime + theme(axis.title.y = element_blank(), axis.text.y = element_blank())
p_complexity <- p_complexity + theme(axis.title.y = element_blank(), axis.text.y = element_blank())
p_solu_rate <- p_solu_rate + theme(axis.title.y = element_blank(), axis.text.y = element_blank())

combined_plot <- (p_r2 | p_solu_rate | p_complexity | p_runtime) +
  plot_layout(guides = 'collect') &
  theme(legend.position = "bottom",
        legend.margin = margin(t = -15, unit = "pt"))
pdf(file = "fig2.pdf", width = 12, height = 6)
combined_plot
dev.off()




