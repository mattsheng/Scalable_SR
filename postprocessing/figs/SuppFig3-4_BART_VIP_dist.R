setwd("~/Dropbox/PAN_SR/postprocessing/figs/")
set.seed(123)
library(arrow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)
library(grid)
library(ggsci)

plot_rank_hist <- function(vip_rank, cluster, cluster_means, mu, title, dist = "euclidean", method = "average") {
  df_rank <- data.frame(rank = vip_rank, cluster = as.factor(cluster))
  df_rank$cluster <- factor(df_rank$cluster, levels = c(1, 2), labels = c("low-mean", "high-mean"))
  p_hist <- ggplot(df_rank, aes(x = rank, fill = cluster)) +
    geom_histogram(binwidth = 5, position = "identity", alpha = 0.6) +
    geom_vline(xintercept = cluster_means, color = "black", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = mu, color =  "red", linewidth = 1, alpha = 0.6) +
    coord_cartesian(xlim = c(0, 180), ylim = c(0, 30)) +
    labs(x = "", y = "", title = title, fill = "Cluster") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12))
  
  return(p_hist)
}

plot_rank_clst <- function(vip_rank, cluster, p, p0, title, dist = "euclidean", method = "average") {
  true_label <- c(rep(1, p0), rep(2, p - p0))
  true_label <- factor(true_label, levels = c(1, 2), labels = c("True Feature", "Irrelevant Feature"))
  
  df_rank <- data.frame(rank = vip_rank, cluster = as.factor(cluster))
  df_rank$cluster <- factor(df_rank$cluster, levels = c(1, 2), labels = c("low-mean", "high-mean"))
  
  p_clust <- ggplot(df_rank, aes(x = rank, y = rank, color = cluster, shape = true_label)) +
    geom_point(size = 3) +
    labs(title = title, x = "", y = "", color = "Cluster", shape = "True Label") +
    coord_cartesian(xlim = c(0, 190), ylim = c(0, 190)) +
    scale_shape_manual(values = c(19, 17)) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12))
  
  return(p_clust)
}

hier_clust <- function(vip_rank, dist = "euclidean", method = "average") {
  # Hierarchical clustering on vip_rank
  vip_rank_dist <- dist(vip_rank, method = dist)
  hclust_result <- hclust(vip_rank_dist, method = method)
  
  # Cut the tree to obtain 2 clusters
  clusters <- cutree(hclust_result, k = 2)
  return(clusters)
}


df_BART <- read_feather("../../results_feynman/feynman_BART_VIP_withidx.feather")
df_BART <- df_BART %>%
  filter(n == 1000, dataset_name == "feynman_I_38_12", random_state == 11284)
df_BART <- df_BART %>%
  mutate(SNR = ifelse(SNR == 0, "noiseless", as.character(SNR)))
df_BART$SNR <- factor(df_BART$SNR, levels = c("noiseless", "20", "15", "10", "5", "2", "1", "0.5"))
df_BART <- df_BART[order(df_BART$SNR), ]

df_BART$cluster <- lapply(df_BART$vip_rank, hier_clust)
df_BART$cluster_means <- lapply(1:nrow(df_BART), function(i) tapply(df_BART$vip_rank[[i]], df_BART$cluster[i], mean))
df_BART$cluster_means <- lapply(df_BART$cluster_means, unname)
df_BART$mu <- lapply(1:nrow(df_BART), function(i) c((1 + df_BART$p0[i]) / 2, (df_BART$p0[i] + df_BART$p[i]) / 2))


hist_plots <- list()
for (i in 1:nrow(df_BART)) {
  hist_plots[[i]] <- with(df_BART, plot_rank_hist(vip_rank[[i]], cluster[[i]], cluster_means[[i]], mu[[i]], paste0("SNR = ", SNR[i])))
}
hist_plots_all <- do.call(ggarrange, c(hist_plots, ncol = 2, nrow = 4, common.legend = TRUE, legend = "bottom"))
pdf(file = "suppfig3.pdf", width = 12, height = 16)
annotate_figure(hist_plots_all, left = text_grob("Frequency", rot = 90, vjust = 1, size = 14),
                bottom = text_grob("Avg VIP Ranking", vjust = -4, size = 14),
                top = text_grob("BART Average VIP Ranking Distribution", face = "bold", size = 18))
dev.off()

clst_plots <- list()
for (i in 1:nrow(df_BART)) {
  clst_plots[[i]] <- with(df_BART, plot_rank_clst(vip_rank[[i]], cluster[[i]], p[i], p0[i], paste0("SNR = ", SNR[i])))
}
clst_plots_all <- do.call(ggarrange, c(clst_plots, ncol = 2, nrow = 4, common.legend = TRUE, legend = "bottom"))
pdf(file = "suppfig4.pdf", width = 12, height = 16)
annotate_figure(clst_plots_all, left = text_grob("Avg VIP Ranking", rot = 90, vjust = 1, size = 14),
                bottom = text_grob("Avg VIP Ranking", vjust = -4, size = 14),
                top = text_grob("BART Average VIP Ranking Cluster", face = "bold", size = 18))
dev.off()
