options(java.parameters = "-Xmx20g")
library(bartMachine)

runBART <- function(dat, seed, rep){
  # Set seed for reproducibility (optional)
  set.seed(seed)
  
  # Prepare input data to bartMachine
  y <- dat[, 1]
  X <- as.data.frame(dat[, -1])
  
  bm <- bartMachine(X = X,
                    y = y,
                    num_trees = 20,
                    num_burn_in = 10000,
                    num_iterations_after_burn_in = 10000,
                    run_in_sample = FALSE,
                    serialize = FALSE,
                    seed = seed,
                    verbose = FALSE)
  bart_machine_arr <- bartMachineArr(bm, R = rep)
  
  # Variable inclusion proportion
  vip <- lapply(bart_machine_arr, function(x) get_var_props_over_chain(x))
  vip <- do.call(rbind, vip) # return list as matrix
  vip_avg <- colMeans(vip, na.rm = TRUE)
  
  # Calculate avg VIP rankings
  vip_rank <- t(apply(vip, 1, function(x) rank(-x)))
  vip_rank_avg <- colMeans(vip_rank, na.rm = TRUE)

  # Hierarchical clustering on avg VIP rankings
  rank_dist <- dist(vip_rank_avg, method = "euclidean")
  hclust_result <- hclust(rank_dist, method = "average")

  # Cut the tree to obtain 2 clusters
  clusters <- cutree(hclust_result, k = 2)
  cluster_means <- tapply(vip_rank_avg, clusters, mean)

  # Find data corresponding to the cluster with the least cluster mean
  pos_cls_id <- as.numeric(names(which.min(cluster_means)))
  idx_hclst <- as.integer(which(clusters == pos_cls_id) - 1) # 0-index for python
  
  # Return results as a list
  return(list(vip_avg = vip_avg, vip_rank_avg = vip_rank_avg, idx_hclst = idx_hclst))
}
