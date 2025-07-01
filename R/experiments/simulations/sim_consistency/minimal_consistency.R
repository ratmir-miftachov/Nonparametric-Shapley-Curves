library(parallel)

# Function to safely source a file
try_source <- function(file_path) {
  tryCatch({
    source(file_path)
    return(TRUE)
  }, error = function(e) {
    message("Error sourcing file: ", file_path)
    message("Original error: ", e$message)
    message("Current working directory: ", getwd())
    return(FALSE)
  })
}

# Set working directory to the project root for consistency
# This assumes the script is run from the project root.
# If not, this path needs to be adjusted.
# setwd("/Users/ratmir/Nonparametric-Shapley-Curves")


ISE_fct = function(m, obs){

  library(np)
  library(pracma)
  library(cubature)
  library(simstudy)
  library(MASS)

  # Source all dependencies with full paths from project root
  try_source("R/src/utils/functions.R")
  try_source("R/src/algorithms/integral_population.R")
  try_source("R/src/algorithms/integral_estimation.R")
  try_source("R/src/algorithms/shapley_int.R")
  try_source("R/src/algorithms/SE_vec.R")
  try_source("R/src/algorithms/SE_vec_int.R")
  try_source("R/src/utils/subsets.R") # This was missing
  try_source("R/src/utils/model_subset.R") # This was likely also needed
  try_source("R/src/utils/model_list_fct.R") # This was likely also needed
  try_source("R/src/utils/weight.R") # This was likely also needed
  
  cova<<-0
  sigma_sim<<-matrix(c(4, cova, cova,
                   cova, 4, cova,
                   cova, cova, 4), nrow=3, ncol=3)
  
  g1 <<- function(X){ return( -sin(2*X[,1]) ) } 
  g2 <<- function(X){ return( cos(3*X[,2])   ) } 
  g3 <<- function(X){ return( 0.5*X[,3] ) } 
  int = function(X){
    x1 = X[,1]
    x2 = X[,2]
    return( 2*cos(x1)*sin(2*x2) ) 
  }
  
  
  l <<- -2; u <<- 2; N<<- obs
  l_int <<- l; u_int <<- u
  d <<- 3
  
  
  X<<-data.frame(mvrnorm(n=N, mu=c(0,0,0), Sigma=sigma_sim))
  
  #DGP
  Y <<- g1(X) + g2(X) + g3(X) + int(X) + rt(n=nrow(X), df=5)
  
  #All possible subsets
  subs <<- subsets(X)
  
  #Get model fits and sort them in a list
  model_list <<- model_list_fct(subs=subs, X=X, Y=Y) 
  
  
  # Component-based
  ISE_res1=hcubature(f=SE_vec, rep(l_int, d), rep(u_int, d), tol=3e-1, j=1)
  ISE_res2=hcubature(f=SE_vec, rep(l_int, d), rep(u_int, d), tol=3e-1, j=2)
  ISE_res3=hcubature(f=SE_vec, rep(l_int, d), rep(u_int, d), tol=3e-1, j=3)
  
  ISE1 = ISE_res1$integral 
  ISE2 = ISE_res2$integral 
  ISE3 = ISE_res3$integral 
  
  
  # Integral-based
  ISE_res1_int=hcubature(f=SE_vec_int, rep(l_int, d), rep(u_int, d), tol=3e-1, j=1)
  ISE_res2_int=hcubature(f=SE_vec_int, rep(l_int, d), rep(u_int, d), tol=3e-1, j=2)
  ISE_res3_int=hcubature(f=SE_vec_int, rep(l_int, d), rep(u_int, d), tol=3e-1, j=3)
  
  ISE1_int = ISE_res1_int$integral 
  ISE2_int = ISE_res2_int$integral  
  ISE3_int = ISE_res3_int$integral
  
  
  return(c(ISE1, ISE1_int, ISE2, ISE2_int, ISE3, ISE3_int))
}

# Run a single instance with 300 observations
set.seed(42) # for reproducibility
results = ISE_fct(1, 300)

# Print the results
print(results) 