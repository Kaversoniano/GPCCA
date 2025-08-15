#' A Package for GPCCA Implementation
#' 
#' This package provides functions to implement Generalized Probabilistic 
#'  Canonical Correlation Analysis (GPCCA) on multi-modal dataset. 
#'  GPCCA is an unsupervised method designed to perform data integration and 
#'  joint dimensionality reduction on a dataset consisting of multiple modalities. 
#'  It bears the potential to effectively learn both shared and complementary 
#'  information across modalities, generally leading to stable statistical results.
#' 
#' @details
#' GPCCA is an extension of the probabilistic CCA (PCCA) with several key merits:
#' \enumerate{
#'   \item It generalizes PCCA from two to more than two modalities.
#'   \item It learns joint low-dimensional embeddings probabilistically.
#'   \item It imputes missing values inherently during its parameter estimation.
#' }
#' 
#' @section Package Functions:
#' \itemize{
#'   \item \code{\link{GPCCAmodel}}: The primary function to fit the 
#'     GPCCA model on a general multi-modal dataset for end users.
#'   \item \code{\link{GPCCAselect}}: A built-in method of model selection on 
#'     target dimension (hyper-parameter) for end users.
#'   \item \code{\link{GPCCAmodel2}}: The primary function to fit the 
#'     GPCCA extension models on a general multi-modal dataset of 
#'     counts only or binary (0/1) observations only for end users.
#' }
#'
#' @author Tianjian Yang, Wei Vivian Li
#' 
#' @name GPCCA
"_PACKAGE"
