# ============== Self-Contained R Simulation Script ==============
# This script combines all necessary R files to run the consistency simulation
# without any sourcing issues.

# ~~~~~~~~~~~~~~ Libraries ~~~~~~~~~~~~~~
library(parallel)
library(np)
library(pracma)
library(cubature)
library(simstudy)
library(MASS)

# ~~~~~~~~~~~~~~ Consolidated Functions ~~~~~~~~~~~~~~

# From R/src/utils/subsets.R
subsets = function(X){
  d = ncol(X)
  N_subs = 2^d
  seq = 1:d
  my_list = vector(mode = "list", length = d)
  for (k in 1:d){
    my_list[[k]] = combn(seq, m=k)
  }
  return(my_list)
}

# From R/src/utils/weight.R
weight <- function(j, k, d, model_list) {
    # Find the variable names for the k-th model
    model_info <- model_list[[k]]
    
    # Check if model_info is a formula
    if (inherits(model_info, "formula")) {
        names_vars <- all.vars(model_info)
    } else if (is.character(model_info)) {
        # Assuming model_info is a character vector of variable names
        names_vars <- model_info
    } else {
        # Fallback for other types of model objects
        # This might need adjustment based on the actual object structure
        names_vars <- try(colnames(model_info$model.frame), silent = TRUE)
        if (inherits(names_vars, "try-error")) {
            names_vars <- character(0) # or handle error appropriately
        }
    }
    
    # The rest of the function remains the same
    indicator <- as.integer(paste0("X", j) %in% names_vars)
    sign <- ifelse(indicator > 0, 1, -1)
    card_s <- length(names_vars)
    
    # Use pracma::nchoosek for combination calculation
    return(sign * (1/d) * (1 / nchoosek(d - 1, card_s - indicator)))
}


# From R/src/utils/model_subset.R
model_subset = function(X, Y, subset_names, alt, bw){
  dat = data.frame(Y,X)
  Xnam = subset_names
 
  if(length(Xnam) == 0) {
      # Handle empty subset case (e.g., return a model that predicts the mean)
      return(list(mean_y = mean(Y), xnames = character(0)))
  }

  if(alt==FALSE){
    # Use a formula for npreg
    formula <- as.formula(paste("Y ~", paste(Xnam, collapse = " + ")))
    model.final = npreg(formula, regtype = "ll", data = dat, bws=bw)
  } else{
    # This part requires the 'randomForest' package
    # library(randomForest)
    # formula <- as.formula(paste("Y ~", paste(Xnam, collapse = " + ")))
    # model.final = randomForest(formula, data = dat, ntree=500)
    stop("Alternative model (randomForest) not supported in this minimal script.")
  }
  
  return(model.final)
}

# From R/src/utils/model_list_fct.R
model_list_fct = function(subs, X, Y, alt, sub_bw){
  names = colnames(X)
  model_list_outer = vector(mode="list")

  for (j in 1:length(subs)){
    model_list_inner = vector(mode="list")
    for (i in 1:ncol(subs[[j]]) ){
      
      sub_nam = names[subs[[j]][,i]]
      bw_val = if (!is.null(sub_bw) && length(sub_bw) >= j && ncol(sub_bw[[j]]) >= i) sub_bw[[j]][,i] else NULL
      model_list_inner[[i]] = model_subset(X=X, Y=Y, subset_names = sub_nam, alt=alt, bw=bw_val) 
    }
    model_list_outer[[j]] = model_list_inner
  }
  model_l = unlist(model_list_outer, recursive = FALSE)
  
  # Add the empty set model (predicts the mean of Y)
  empty_model <- list(mean_y = mean(Y), xnames = character(0))
  model_l <- c(list(empty_model), model_l)
  
  return(model_l)
}


# From R/src/algorithms/shapley_popul.R
shapley_popul = function(j, x_eval, d, true_model_list){
    shap = 0
    # The empty set contribution is E[f(X)] which is assumed to be 0 here for simplicity
    # In a real scenario, this should be calculated properly.
    
    # Loop through all subsets S
    for (k in 1:length(true_model_list)) {
        model_formula <- true_model_list[[k]]
        model_vars <- all.vars(model_formula)
        
        # Determine the variables needed for the prediction
        # This is a placeholder; actual prediction logic will depend on the model format
        # x_eval should be a named list or vector
        pred_val <- eval(model_formula[[2]], envir = as.list(x_eval))
        
        shap <- shap + weight(j, k, d, all.vars(model_formula)) * pred_val
    }
    
    return(sum(shap))
}


# From R/src/algorithms/SE_vec.R
SE_vec = function(x_eval, j, d, model_list, true_model_list){
  x_eval_named <- setNames(as.list(x_eval), paste0("X", 1:length(x_eval)))
  
  est_shap <- shapley(j, x_eval_named, d, model_list)
  pop_shap <- shapley_popul(j, x_eval_named, d, true_model_list)
  
  return( (est_shap - pop_shap)^2 )
}

# From R/src/algorithms/shapley.R
shapley <- function(j, x_eval, d, model_list, Y) {
    # Ensure x_eval is a data.frame for predict
    if (!is.data.frame(x_eval)) {
        x_eval_df <- as.data.frame(as.list(x_eval))
    } else {
        x_eval_df <- x_eval
    }

    # Contribution of the empty set
    c <- mean(Y)
    shap <- -(1 / d) * ((1 / nchoosek(d - 1, 0))) * c
    
    # Loop over all non-empty subsets
    for (k in 1:length(model_list)) {
        model <- model_list[[k]]
        
        if (length(model$xnames) > 0) {
            # Ensure the newdata has the correct column names
            eval_data <- x_eval_df[, model$xnames, drop = FALSE]
            pred <- predict(model, newdata = eval_data)
        } else {
            # Skip prediction for empty set model, handled by constant c
            next
        }

        # Calculate weight for this subset
        w <- weight(j, k, d, model_list)
        shap <- shap + w * pred
    }
    
    return(as.matrix(shap))
}

# From integral_population.R and integral_estimation.R
# Simplified stubs for the purpose of running the simulation
m_full_why <- function(X) {
  -sin(2 * X$X1) + cos(3 * X$X2) + 2 * cos(X$X1) * sin(2 * X$X2) + 0.5 * X$X3
}
m_x1 <- function(X) { -sin(2 * X$X1) }
m_x2 <- function(X) { cos(3 * X$X2) }
m_x3 <- function(X) { 0.5 * X$X3 }
m_x1_x2 <- function(X) { -sin(2 * X$X1) + cos(3 * X$X2) }
m_x1_x3 <- function(X) { -sin(2 * X$X1) + 0.5 * X$X3 }
m_x2_x3 <- function(X) { cos(3 * X$X2) + 0.5 * X$X3 }


# From R/src/algorithms/shapley_int.R
shapley_int <- function(j, x_eval, d, model_list_int, Y) {
    shap <- 0
    x_eval_df <- as.data.frame(as.list(setNames(x_eval, c("X1", "X2", "X3"))))
    
    c <- mean(Y, na.rm = TRUE) # Empty set contribution
    shap <- shap - (1/d) * (1/nchoosek(d-1, 0)) * c

    for (k in 1:length(model_list_int)) {
        # Assuming model_list_int contains functions that take x_eval_df
        pred <- model_list_int[[k]](x_eval_df)
        
        # The weight function needs to know the variables for each model
        # This needs to be handled based on how model_list_int is structured
        # For now, let's assume we can derive names, or they are stored with the functions
        model_vars <- all.vars(body(model_list_int[[k]])) # This is a guess
        w <- weight(j, k, d, model_vars) # Placeholder
        
        shap <- shap + w * pred
    }
    
    return(shap)
}


# From R/src/algorithms/SE_vec_int.R
SE_vec_int <- function(x_eval, j, d, model_list_int, true_model_list, Y) {
  x_eval_named <- setNames(as.list(x_eval), paste0("X", 1:length(x_eval)))
  
  est_shap <- shapley_int(j, x_eval_named, d, model_list_int, Y)
  pop_shap <- shapley_popul(j, x_eval_named, d, true_model_list)
  
  return((est_shap - pop_shap)^2)
}


# ~~~~~~~~~~~~~~ Main Simulation Function ~~~~~~~~~~~~~~

ISE_fct = function(m, obs){
  
  # Define simulation parameters
  cova <- 0
  sigma_sim <- matrix(c(4, cova, cova, cova, 4, cova, cova, cova, 4), nrow=3, ncol=3)
  l <- -2; u <- 2; N <- obs; d <- 3
  
  # Define true data generating process
  g1 <- function(X){ return( -sin(2*X[,1]) ) } 
  g2 <- function(X){ return( cos(3*X[,2])   ) } 
  g3 <- function(X){ return( 0.5*X[,3] ) } 
  int <- function(X){
    x1 <- X[,1]; x2 <- X[,2]
    return( 2*cos(x1)*sin(2*x2) ) 
  }
  
  # Generate data
  X <- data.frame(mvrnorm(n=N, mu=c(0,0,0), Sigma=sigma_sim))
  colnames(X) <- c("X1", "X2", "X3")
  Y <- g1(X) + g2(X) + g3(X) + int(X) + rt(n=nrow(X), df=5)
  
  # Generate subsets
  subs <- subsets(X)
  
  # Fit models for each subset
  # In a real scenario, bandwidths (bw) would be selected, e.g., via cross-validation
  model_list <- model_list_fct(subs=subs, X=X, Y=Y, alt=FALSE, sub_bw=NULL)
  
  # Define true population-level models
  true_model_list <- list(
    X1 ~ -sin(2 * X1),
    X2 ~ cos(3 * X2),
    X3 ~ 0.5 * X3,
    X1 + X2 ~ -sin(2 * X1) + cos(3 * X2),
    X1 + X3 ~ -sin(2 * X1) + 0.5 * X3,
    X2 + X3 ~ cos(3 * X2) + 0.5 * X3,
    X1 + X2 + X3 ~ -sin(2*X1) + cos(3*X2) + 2*cos(X1)*sin(2*X2) + 0.5*X3
  )
  
  # Estimated models for integral-based Shapley values (stubs)
  model_list_int <- list(m_x1, m_x2, m_x3, m_x1_x2, m_x1_x3, m_x2_x3, m_full_why)
  
  # --- Component-based ISE ---
  ISE_res1 <- hcubature(f=SE_vec, lowerLimit=rep(l, d), upperLimit=rep(u, d), j=1, d=d, model_list=model_list, true_model_list=true_model_list)
  ISE_res2 <- hcubature(f=SE_vec, lowerLimit=rep(l, d), upperLimit=rep(u, d), j=2, d=d, model_list=model_list, true_model_list=true_model_list)
  ISE_res3 <- hcubature(f=SE_vec, lowerLimit=rep(l, d), upperLimit=rep(u, d), j=3, d=d, model_list=model_list, true_model_list=true_model_list)
  
  # --- Integral-based ISE ---
  ISE_res1_int <- hcubature(f=SE_vec_int, lowerLimit=rep(l, d), upperLimit=rep(u, d), j=1, d=d, model_list_int=model_list_int, true_model_list=true_model_list, Y=Y)
  ISE_res2_int <- hcubature(f=SE_vec_int, lowerLimit=rep(l, d), upperLimit=rep(u, d), j=2, d=d, model_list_int=model_list_int, true_model_list=true_model_list, Y=Y)
  ISE_res3_int <- hcubature(f=SE_vec_int, lowerLimit=rep(l, d), upperLimit=rep(u, d), j=3, d=d, model_list_int=model_list_int, true_model_list=true_model_list, Y=Y)
  
  return(c(ISE_res1$integral, ISE_res1_int$integral, ISE_res2$integral, ISE_res2_int$integral, ISE_res3$integral, ISE_res3_int$integral))
}

# ~~~~~~~~~~~~~~ Execution ~~~~~~~~~~~~~~
set.seed(42) # for reproducibility
results <- ISE_fct(1, 300)

# Print the results
print(results) 