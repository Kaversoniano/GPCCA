#' Generate Example Multi-modal Dataset
#'
#' Simulation of a synthetic multi-modal dataset with three modalities. 
#'   The number of features in each modality is \eqn{m_1 = 30}, \eqn{m_2 = 60} and \eqn{m_3 = 90}, 
#'   respectively. There are six groups of samples where each group has a equal 
#'   sample size of \eqn{50}. In total, there are \eqn{n = 300} samples, 
#'   with ground truth labels provided by \code{example_MultiModalData_labels}. 
#'   Artificial missing values are introduced with MCAR mechanism. 
#'   The simulated datasets should vary under different random seeds.
#'
#' @param dataType One of \code{"continuous"}, \code{"count"} and \code{"binary"}, 
#'   specifying the data type of matrices to be generated. (Default: \code{"continuous"})
#' @param missVal a logical. 
#'   Should artificial missing values (\code{NA}s) be introduced? (Default: \code{TRUE})
#' @param propMCAR a numeric scalar. 
#'   The overall missing rate due to MCAR, between \code{0} and \code{0.5}. 
#'   It affects nothing if \code{missVal = FALSE}. (Default: \code{0.2})
#'
#' @return A list that consists of 3 numeric matrices is returned:
#' \describe{
#'   \item{M1}{Modality 1: a matrix of size \eqn{m_1 \times n}, 
#'     i.e. 30 features (rows) by 300 samples (columns).}
#'   \item{M2}{Modality 2: a matrix of size \eqn{m_2 \times n}, 
#'     i.e. 60 features (rows) by 300 samples (columns).}
#'   \item{M3}{Modality 3: a matrix of size \eqn{m_3 \times n}, 
#'     i.e. 90 features (rows) by 300 samples (columns).}
#' }
#'
#' @seealso \code{\link{example_MultiModalData_labels}}
#'
#' @examples
#' \dontrun{
#' 
#' ## I: Generate a 3-modality continuous dataset with no missing value
#' DAT <- example_MultiModalData_sim(dataType = "continuous", missVal = FALSE)
#' 
#' ## II: Generate a 3-modality count dataset with 10% MCARs
#' DAT <- example_MultiModalData_sim(dataType = "count", missVal = TRUE, propMCAR = 0.1)
#' 
#' ## III: Generate a 3-modality binary dataset with 50% MCARs
#' DAT <- example_MultiModalData_sim(dataType = "binary", missVal = TRUE, propMCAR = 0.5)
#' }

example_MultiModalData_sim <- function(dataType = "continuous", missVal = TRUE, propMCAR = 0.2) {
  
  ##############################################################################
  ### Argument: dataType
  stopifnot(!is.na(dataType), is.character(dataType), length(dataType) == 1)
  if (!(dataType == "continuous" | dataType == "count" | dataType == "binary")) {
    stop("The 'dataType' of matrices to be generated must be one of 'continuous', 'count' and 'binary'.")
  }
  
  ### Argument: missVal
  stopifnot(!is.na(missVal), is.logical(missVal), length(missVal) == 1)
  
  ### Argument: propMCAR
  stopifnot(!is.na(propMCAR), is.numeric(propMCAR), length(propMCAR) == 1)
  if (!(propMCAR >= 0 & propMCAR <= 0.5)) {
    stop("The overall missing rate due to MCAR 'propMCAR' must be in [0, 0.5].")
  }
  
  ##############################################################################
  if(!missVal) padMsg <- "with no missing value."
  if(missVal) padMsg <- paste0("with ", round(100 * propMCAR, 2), "% MCARs.")
  
  if (dataType == "continuous") {
    message(paste0("A synthetic continuous dataset of three modalities is generated, ", padMsg))
    message("Please check the returned list that consists of three numeric matrices.")
    simDAT <- sim_MultiModalData_base(missVal = missVal, propMCAR = propMCAR)
  }
  
  if (dataType == "count") {
    message(paste0("A synthetic count dataset of three modalities is generated, ", padMsg))
    message("Please check the returned list that consists of three numeric matrices.")
    simDAT <- sim_MultiModalData_ext(missVal = missVal, propMCAR = propMCAR, dataType = dataType)
  }
  
  if (dataType == "binary") {
    message(paste0("A synthetic binary dataset of three modalities is generated, ", padMsg))
    message("Please check the returned list that consists of three numeric matrices.")
    simDAT <- sim_MultiModalData_ext(missVal = missVal, propMCAR = propMCAR, dataType = dataType)
  }
  
  return(simDAT)
}


