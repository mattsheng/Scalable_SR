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
                    num_burn_in = 5000,
                    num_iterations_after_burn_in = 5000,
                    run_in_sample = FALSE,
                    serialize = FALSE,
                    seed = seed,
                    verbose = FALSE)
  var_sel <- tryCatch(
    {
      var_selection_by_permute(bm,
                               num_reps_for_avg = 10,
                               num_permute_samples = rep,
                               num_trees_for_permute = 20,
                               plot = FALSE)
    },
    error = function(e) {
      message("BART-G.SE variable selection failed")
      message("Here's the original error message:")
      message(conditionMessage(e))
      
      list(important_vars_local_col_nums = NA,
           important_vars_global_max_col_nums = NA,
           important_vars_global_se_col_nums = NA)
    }
  )
  
  # Print the selected column indices to stdout
  idx_local <- var_sel$important_vars_local_col_nums
  idx_gmax <- var_sel$important_vars_global_max_col_nums
  idx_gse <- var_sel$important_vars_global_se_col_nums

  if (!anyNA(idx_local)) {
    if (length(idx_local) == 1) {
      idx_local <- NA
    } else {
      idx_local <- sort(idx_local) - 1
    }
  } else {
    idx_local <- NA
  }

  if (!anyNA(idx_gmax)) {
    if (length(idx_gmax) == 1) {
      idx_gmax <- NA
    } else {
      idx_gmax <- sort(idx_gmax) - 1
    }
  } else {
    idx_gmax <- NA
  }

  if (!anyNA(idx_gse)) {
    if (length(idx_gse) == 1) {
      idx_gse <- NA
    } else {
      idx_gse <- sort(idx_gse) - 1
    }
  } else {
    idx_gse <- NA
  }

  return(list(
    idx_local = as.integer(idx_local),
    idx_gmax = as.integer(idx_gmax),
    idx_gse = as.integer(idx_gse)
  ))
}