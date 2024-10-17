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
df_pmlb <- read_feather("../../results_blackbox/pmlb_results.feather")

# Rename selection method
df_pmlb$vs_method[df_pmlb$vs_method == "none"] <- "SR"
df_pmlb$vs_method[df_pmlb$vs_method == "hclst_v2"] <- "PAN+SR"

# Extract BART runtime
df_BART <- read_feather("../../results_blackbox/pmlb_BART_VIP_withidx.feather")
df_BART <- df_BART %>% select(dataset_name, random_state, time_time)
colnames(df_BART) <- c("dataset_name", "random_state", "BART_time")

# Merge BART runtime with df_pmlb
df_pmlb <- df_pmlb %>% 
  left_join(df_BART, by = c("dataset_name", "random_state"))
df_pmlb$total_time <- ifelse(df_pmlb$vs_method == "PAN+SR", df_pmlb$time_time + df_pmlb$BART_time, df_pmlb$time_time)

# If simplified complexity is NA, fill with original complexity
df_pmlb <- df_pmlb %>% mutate(complexity_simplified = coalesce(complexity_simplified, model_size))

# Average over `random_state`
summary_pmlb <- df_pmlb %>%
  group_by(dataset_name, algorithm, vs_method) %>%
  summarize(mean_r2 = mean(r2_zero_test, na.rm = TRUE),
            mean_rmse = mean(sqrt(mse_test), na.rm = TRUE),
            mean_mae = mean(mae_test, na.rm = TRUE),
            mean_time = mean(time_time, na.rm = TRUE),
            mean_complexity = mean(complexity_simplified, na.rm = TRUE),
            .groups = 'drop') %>%
  group_by(algorithm, vs_method) 

# Average over `dataset_name`
summary_pmlb_compact <- summary_pmlb %>%
  group_by(algorithm, vs_method) %>%
  summarize(
    mean_r2 = mean(mean_r2, na.rm = TRUE),
    mean_rmse = mean(mean_rmse, na.rm = TRUE),
    mean_mae = mean(mean_mae, na.rm = TRUE),
    mean_time = mean(mean_time, na.rm = TRUE),
    mean_complexity = mean(mean_complexity, na.rm = TRUE),
    .groups = 'drop'
  )

# Create a variable to order methods by their average R-squared in SR mode
order_pmlb <- summary_pmlb_compact %>%
  filter(vs_method == "SR") %>%
  arrange(desc(mean_r2)) %>%
  pull(algorithm)

# Reorder the method factor based on SR's mean R-squared
summary_pmlb <- summary_pmlb %>%
  mutate(algorithm = factor(algorithm, levels = rev(order_pmlb)))

# R2 test
p_r2 <- feynman_SR_plot(df = summary_pmlb, metric = "mean_r2", title = expression(R^2~"Test"))

# Training time
p_runtime <- feynman_SR_plot(df = summary_pmlb, metric = "mean_time", title = "Training Time (s)")

# Complexity
p_complexity <- feynman_SR_plot(df = summary_pmlb, metric = "mean_complexity", title = "Model Size") 

# Modify the second plot to remove y-axis text and ticks
p_runtime <- p_runtime + theme(axis.title.y = element_blank(), axis.text.y = element_blank())
p_complexity <- p_complexity + theme(axis.title.y = element_blank(), axis.text.y = element_blank())

combined_plot <- (p_r2 | p_complexity | p_runtime) +
  plot_layout(guides = 'collect') &
  theme(legend.position = "bottom",
        legend.margin = margin(t = -15, unit = "pt"))
pdf(file = "fig1.pdf", width = 12, height = 6)
combined_plot
dev.off()

