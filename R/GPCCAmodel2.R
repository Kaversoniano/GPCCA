#' Fit the GPCCA Extension Models
#'
#' Implementation of GPCCA extension models: GPCCA-Pois and GPCCA-Bern. 
#'   GPCCA-Pois is designed for count data only and 
#'   GPCCA-Bern is designed for binary data only, 
#'   while GPCCA itself is developed for general continuous data. 
#'   Like GPCCA, the extension models perform both data integration and 
#'   joint dimensionality reduction on a multi-modal dataset. 
#'   The key output is the resulting projections, i.e. 
#'   joint low-dimensional embeddings. The reconstructed data representations 
#'   learnt by both models can be directly used for 
#'   further downstream analysis of users' preference. 
#'   Model selection for the extension models is under development. 
#'   See \code{\link{GPCCAmodel}} for the original GPCCA model.
#'   
#' For high-dimensional data, feature selection is strongly recommended so that 
#'   only the highly informative features are included in the model fit. This helps 
#'   to reduce the computational resources required and boost the computational speed.
#'
#' @param Y.list a list of numeric matrices. A multi-modal dataset of \eqn{R} 
#'   modalities, where each matrix \eqn{\mathbf{Y}^{(r)}} is a data modality, 
#'   with \eqn{m_r} rows denoting features and \eqn{n} columns denoting samples 
#'   (\eqn{1 \le r \le R}). All modalities must have the same sample size \eqn{n}, 
#'   with the ordering of samples matched to each other. Artificial modality-wise 
#'   missing values should be introduced by the user if non-matching samples are present.
#' @param dataType One of \code{"count"} and \code{"binary"}, 
#'   specifying the data type of matrices in \code{Y.list}. 
#'   It must be explicitly specified since its default is \code{NULL}. 
#'   The case \code{dataType = "continuous"} is implemented in 
#'   a separate framework by \code{\link{GPCCAmodel}}.
#' @param d an integer. The size of target dimension \eqn{d}, i.e. the number of 
#'   latent factors in low-dimensional subspace. (Default: \code{2})
#' @param lambda a numeric scalar. The ridge regularization parameter \eqn{\lambda}, 
#'   in the range of \eqn{[0, \infty)}. A larger \eqn{\lambda} corresponds to heavier 
#'   penalty, which leads to a more unique solution of the loading matrix \eqn{\mathbf{W}} 
#'   and correspondingly more unique joint low-dimensional embeddings. 
#'   \eqn{\lambda = 0} means no ridge regularization is applied. (Default: \code{0.1})
#' @param tol a numeric scalar. Tolerance of the RMSE measuring the difference 
#'   in the matrix of latent factors \eqn{\mathbf{Z}} between two consecutive 
#'   iterations. This tolerance threshold \eqn{\tau} determines the stopping rule 
#'   of the EM algorithm, and the RMSE is used to monitor the convergence.
#'   (Default: \code{0.0001})
#' @param maxiter an integer. The maximum number of iterations allowed in the 
#'   EM algorithm. (Default: \code{50})
#' @param EarlyStop a logical. Should early stop be used? If set to \code{TRUE}, 
#'   it stops the EM algorithm early. With early stop turned on, only sub-optimal 
#'   solution may be attained, but it helps avoid overfitting. (Default: \code{TRUE})
#' @param niter.ES an integer. The number of consecutive increase of RMSE allowed 
#'   to decide for early stop. It affects nothing if \code{EarlyStop = FALSE}. 
#'   (Default: \code{5})
#' @param W.init One of \code{"PCA"} and \code{"rand.std.norm"}, 
#'   indicating which method is used for initializing the loading matrix 
#'   \eqn{\mathbf{W}}. \code{"rand.std.norm"} initializes all elements of 
#'   \eqn{\mathbf{W}} with i.i.d. standard normal random variates. \code{"PCA"} 
#'   uses Randomized Principal Component Analysis (RPCA) if all data modalities 
#'   are complete without missingness or Probabilistic Principal Component Analysis 
#'   (PPCA) if any missing value is present in the multi-modal data \code{Y.list}. 
#'   (Default: \code{"PCA"})
#' @param verbose a binary. If set to \code{0}, it prevents the printing of 
#'   runtime and RMSE in every iteration. (Default: \code{1})
#' @param plot_RMSE a logical. After the EM algorithm is completed, should the 
#'   RMSEs in iterations be plotted? (Default: \code{TRUE})
#' @param missVal a logical. Does the input multi-modal data \code{Y.list} 
#'   contain any missing value \code{NA}? If \code{Y.list} is known to contain 
#'   no missing value, setting \code{missVal = FALSE} can reduce runtime 
#'   significantly. (Default: \code{TRUE})
#'
#' @details
#' \bold{GPCCA extension models:} 
#'   Suppose there are \eqn{n} subjects on which \eqn{R} 
#'   modalities of data are collected.
#'   
#'   The model of GPCCA-Pois on multi-modal count data is specified as:
#'   \deqn{\mathbf{Y}_{ik}|\mathbf{X}_{ik}\overset{\text{ind.}}{\sim}\text{Poisson}(e^{\mathbf{X}_{ik}}) 
#'   \text{ and } \mathbf{X}_{\cdot k} = \boldsymbol{\mu} + \mathbf{W}\mathbf{Z}_{\cdot k}}
#'   The model of GPCCA-Bern on multi-modal binary data is specified as:
#'   \deqn{\mathbf{Y}_{ik}|\mathbf{X}_{ik}\overset{\text{ind.}}{\sim}\text{Bernoulli}((1 + e^{-\mathbf{X}_{ik}})^{-1}) 
#'   \text{ and } \mathbf{X}_{\cdot k} = \boldsymbol{\mu} + \mathbf{W}\mathbf{Z}_{\cdot k}}
#' \itemize{
#'   \item Data matrix, consisting of \eqn{R} modalities: 
#'     \eqn{\mathbf{Y} = [{\mathbf{Y}^{(1)}}^\text{T}, \dots, 
#'     {\mathbf{Y}^{(R)}}^\text{T}]^\text{T} \in \text{R}^{m \times n}} 
#'     where \eqn{\mathbf{Y}^{(r)} \in \text{R}^{m_r \times n}}
#'   \item Mean vector: 
#'     \eqn{\boldsymbol{\mu} \in \text{R}^{m \times 1}}, constant parameter
#'   \item Loading matrix: 
#'     \eqn{\mathbf{W} \in \text{R}^{m \times d}}, constant parameter
#'   \item Matrix of latent factors: 
#'     \eqn{\mathbf{Z} \in \text{R}^{d \times n}} where 
#'     \eqn{\{\mathbf{Z}_{\cdot k}\}_{1 \le k \le n} \overset{\text{i.i.d.}}{\sim} 
#'     \mathcal{N}_d(\mathbf{0}, \mathbf{I})}, random
#'   \item \eqn{n} is the sample size, \eqn{m = \sum_{r=1}^{R}m_r} is the total 
#'     number of features, and for the number of latent factors \eqn{d}, 
#'     we require \eqn{1 \le d \le \min\{m_r\}_{1 \le r \le R}}
#' }
#' \bold{Stopping rule:} 
#'   The EM algorithm stops when the difference in \eqn{\mathbf{Z}} 
#'   between two consecutive iterations (\eqn{t} and \eqn{t+1}) is smaller than 
#'   a threshold \eqn{\tau} (\eqn{\tau > 0}): 
#'   \deqn{\text{RMSE}_{(t)}^{(t+1)} = 
#'   \sqrt{\dfrac{1}{nd}\sum_{i=1}^d\sum_{j=1}^n(\mathbf{Z}^{(t+1)}_{i,j} - 
#'   \mathbf{Z}^{(t)}_{i,j})^2} < \tau}
#' \bold{Ridge regularization:} 
#'   In the M-step of the EM algorithm, we maximize the following regularized and 
#'   normalized Monte Carlo approximation of the expected complete log-likelihood: 
#'   \deqn{\mathcal{Q}_{\lambda}^{\text{MC}}(\boldsymbol{\Theta}|\boldsymbol{\Theta}^{(t)}) = 
#'   \dfrac{1}{n}\mathcal{Q}^{\text{MC}}(\boldsymbol{\Theta}|\boldsymbol{\Theta}^{(t)}) - 
#'   \lambda\text{tr}(\mathbf{W}^{\text{T}}\mathbf{W})} 
#'   \deqn{\text{ with } \text{E}[\ell_{\text{c}}(\boldsymbol{\Theta})] 
#'   \propto \mathcal{Q}(\boldsymbol{\Theta}) 
#'   \approx \mathcal{Q}^{\text{MC}}(\boldsymbol{\Theta})} 
#'   where \eqn{\lambda \in [0, \infty)} 
#'   and \eqn{\boldsymbol{\Theta} = (\boldsymbol{\mu}, \mathbf{W})} denotes 
#'   the model parameter set. The ridge regularization parameter \eqn{\lambda} 
#'   resolves the identifiability issue and improves numerical stability, 
#'   whose optimal value can vary across datasets. Hence, multiple \eqn{\lambda} 
#'   values should be tested in exploratory analysis 
#'   on a small subset to find the most suitable value.
#'
#' @return A list of 3 is returned, including:
#' \describe{
#'   \item{\code{Z}}{a numeric matrix; a matrix of size \eqn{d \times n} that 
#'     stores the fitted latent factors, i.e. the joint low-dimensional 
#'     embeddings of the original multi-modal data.}
#'   \item{\code{mu}}{a numeric vector; the estimated mean vector 
#'     \eqn{\boldsymbol{\mu}} of length \eqn{m}.}
#'   \item{\code{W}}{a numeric matrix; the estimated loading matrix 
#'     \eqn{\mathbf{W}} of size \eqn{m \times d}.}
#' }
#'
#' @seealso \code{\link{GPCCAmodel}}, 
#'   \code{\link{EM_completeDAT_Pois}}, \code{\link{EM_missingDAT_Pois}}, 
#'   \code{\link{EM_completeDAT_Bern}}, \code{\link{EM_missingDAT_Bern}}
#'
#' @examples
#' \dontrun{
#' 
#' ## I: Fit GPCCA-Pois model to the incomplete example multi-modal dataset
#' DAT <- example_MultiModalData_sim(dataType = "count", missVal = TRUE)
#' GPCCA.fit <- GPCCAmodel2(Y.list = DAT, d = 4, lambda = 0.1)
#' 
#' ## II: Fit GPCCA-Pois model to the complete example multi-modal dataset
#' DAT <- example_MultiModalData_sim(dataType = "count", missVal = FALSE)
#' GPCCA.fit <- GPCCAmodel2(Y.list = DAT, d = 4, lambda = 0.1, missVal = FALSE)
#' 
#' ## III: Fit GPCCA-Bern model to the incomplete example multi-modal dataset
#' DAT <- example_MultiModalData_sim(dataType = "binary", missVal = TRUE)
#' GPCCA.fit <- GPCCAmodel2(Y.list = DAT, d = 4, lambda = 0.1)
#' 
#' ## IV: Fit GPCCA-Bern model to the complete example multi-modal dataset
#' DAT <- example_MultiModalData_sim(dataType = "binary", missVal = FALSE)
#' GPCCA.fit <- GPCCAmodel2(Y.list = DAT, d = 4, lambda = 0.1, missVal = FALSE)
#' 
#' ## Extract the fitted latent factors
#' LFs <- t(GPCCA.fit$Z)
#' 
#' ## Data visualization
#' lbls <- example_MultiModalData_labels
#' 
#' par(pty = "s", xpd = TRUE, mar = c(5.1, 4.1, 4.1, 4.1))
#' plot(x = LFs[, 1], y = LFs[, 2], col = factor(lbls),
#'      type = "p", pch = 16, cex = 0.8, asp = 1,
#'      xlab = "Latent Factor 1", ylab = "Latent Factor 2",
#'      main = "Colored by ground truth labels")
#' legend("right", inset = c(-0.3, 0), title = "Group",
#'        legend = levels(factor(lbls)), col = 1:6,
#'        pch = 16, cex = 0.8)
#' }

GPCCAmodel2 <- function(Y.list, dataType = NULL,
                        d = 2, lambda = 0.1, tol = 0.0001, maxiter = 50,
                        EarlyStop = TRUE, niter.ES = 5,
                        W.init = "PCA", verbose = 1, plot_RMSE = TRUE,
                        missVal = TRUE) {
  
  ##### Check the validity of arguments ----------------------------------------
  message("Checking validity of arguments...")
  
  ### Argument: Y.list
  stopifnot(!is.null(Y.list), is.list(Y.list))
  if (!all(sapply(X = Y.list, FUN = function(mat) is.matrix(mat) && is.numeric(mat)))) {
    stop("Every element of the list 'Y.list' must be a numeric matrix.")
  }
  if (!(length(Y.list) >= 2)) {
    stop("In multi-modal dataset 'Y.list', the number of modalities must be at least 2.")
  }
  
  ### Argument: dataType
  stopifnot(!is.na(dataType), is.character(dataType), length(dataType) == 1)
  if (!(dataType == "count" | dataType == "binary")) {
    stop("The 'dataType' of matrices in 'Y.list' must be one of 'count' and 'binary'.")
  }
  
  ### Argument: d
  stopifnot(!is.na(d), is.numeric(d), length(d) == 1)
  if (!(d >= 2)) {
    stop("The size of target dimension 'd' must be at least 2.")
  }
  if (!(d %% 1 == 0)) {
    stop("The size of target dimension 'd' must be an integer.")
  }
  
  ### Argument: lambda
  stopifnot(!is.na(lambda), is.numeric(lambda), length(lambda) == 1)
  if (!(lambda >= 0)) {
    stop("The ridge regularization parameter 'lambda' must be nonnegative.")
  }
  
  ### Argument: tol
  stopifnot(!is.na(tol), is.numeric(tol), length(tol) == 1)
  if (!(tol > 0)) {
    stop("The tolerance threshold 'tol' used in the stopping rule must be positive.")
  }
  
  ### Argument: maxiter
  stopifnot(!is.na(maxiter), is.numeric(maxiter), length(maxiter) == 1)
  if (!(maxiter >= 1)) {
    stop("The max number of iterations 'maxiter' allowed in the EM algorithm must be at least 1.")
  }
  if (!(maxiter %% 1 == 0)) {
    stop("The max number of iterations 'maxiter' allowed in the EM algorithm must be an integer.")
  }
  
  ### Argument: EarlyStop
  stopifnot(!is.na(EarlyStop), is.logical(EarlyStop), length(EarlyStop) == 1)
  
  ### Argument: niter.ES
  stopifnot(!is.na(niter.ES), is.numeric(niter.ES), length(niter.ES) == 1)
  if (!(niter.ES >= 2)) {
    stop("To decide for early stop, 'niter.ES' must be at least 2.")
  }
  if (!(niter.ES %% 1 == 0)) {
    stop("To decide for early stop, 'niter.ES' must be an integer.")
  }
  
  ### Argument: W.init
  stopifnot(!is.na(W.init), is.character(W.init), length(W.init) == 1)
  if (!(W.init == "PCA" | W.init == "rand.std.norm")) {
    stop("The initialization method 'W.init' must be one of 'PCA' and 'rand.std.norm'.")
  }
  
  ### Argument: verbose
  stopifnot(!is.na(verbose), is.numeric(verbose), length(verbose) == 1)
  if (!(verbose == 0 | verbose == 1)) {
    stop("Whether or not to print in every iteration 'verbose' must be one of 0 and 1.")
  }
  
  ### Argument: plot_RMSE
  stopifnot(!is.na(plot_RMSE), is.logical(plot_RMSE), length(plot_RMSE) == 1)
  
  ### Argument: missVal
  stopifnot(!is.na(missVal), is.logical(missVal), length(missVal) == 1)
  
  ##### Check the advanced requirements for argument: Y.list -------------------
  ##### (only suggestions are provided but no auto-fix is applied) -------------
  message("Checking requirements for the input multi-modal data...")
  
  ### Requirement of equal sample sizes for Y.list
  if (length(unique(sapply(X = Y.list, FUN = ncol))) != 1) {
    warning("All modalities in 'Y.list' must have equal sample size, with matched ordering of samples. ",
            "Artificial modality-wise missingness should be introduced by the user if there are any non-matching samples.")
    stop("Sample sizes of the given ", length(Y.list), " modalities are not equal.")
  }
  
  ### Requirement of no presence of problematic sample
  na.flags <- do.call(rbind, lapply(X = Y.list, FUN = is.na))
  invalid.flags <- colSums(na.flags) == nrow(na.flags)
  if (sum(invalid.flags) > 0) {
    warning("If a sample has only NA values across all modalities (all features are missing), it must be discarded.")
    stop("Consider removing these samples with column indices: ",
         paste(unname(which(invalid.flags)), collapse = ", "))
  }
  
  ##### Check the consistency between arguments --------------------------------
  message("Checking consistency between arguments...")
  
  ### Inconsistency due to missVal = FALSE and presence of missing value
  if (missVal == FALSE) {
    if (sum(na.flags) > 0) {
      warning("If the input multi-modal data 'Y.list' contains any missing value as NA, ",
              "the default setting 'missVal = TRUE' should be used.")
      stop("'missVal = FALSE' is specified explicitly but the input of 'Y.list' contains missing values.")
    }
  }
  
  ### Requirement of no non-finite values
  Y.cc <- do.call(rbind, Y.list)
  if (any(is.infinite(Y.cc) | is.nan(Y.cc))) {
    stop("The input multi-modal data 'Y.list' contains non-finite values (Inf/-Inf/NaN).")
  }
  
  ### Inconsistency due to dataType = "count" but invalid input data type
  if (dataType == "count") {
    if (!all(abs(Y.cc - round(Y.cc)) < 1e-08, na.rm = TRUE)) {
      stop("dataType = 'count' is declared but 'Y.list' contains non-integers.")
    }
    if (!all(Y.cc >= 0, na.rm = TRUE)) {
      stop("dataType = 'count' is declared but 'Y.list' contains negative values.")
    }
  }
  
  ### Inconsistency due to dataType = "binary" but invalid input data type
  if (dataType == "binary") {
    if (!all(abs(Y.cc - round(Y.cc)) < 1e-08, na.rm = TRUE)) {
      stop("dataType = 'binary' is declared but 'Y.list' contains non-integers.")
    }
    if (!all(Y.cc %in% c(0, 1, NA))) {
      stop("dataType = 'binary' is declared but 'Y.list' contains non-binary(0/1) values.")
    }
  }
  
  ##### ------------------------------------------------------------------------
  message("All checks are passed!")
  
  ##### Obtain the fitted GPCCA model ------------------------------------------
  if (dataType == "count") {
    message("Start fitting the GPCCA-Pois model...")
    
    if (missVal) {
      GPCCAfit <- EM_missingDAT_Pois(Y.list = Y.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                                     EarlyStop = EarlyStop, niter.ES = niter.ES,
                                     W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
    }
    
    if (!missVal) {
      GPCCAfit <- EM_completeDAT_Pois(Y.list = Y.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                                     EarlyStop = EarlyStop, niter.ES = niter.ES,
                                     W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
    }
  }
  
  if (dataType == "binary") {
    message("Start fitting the GPCCA-Bern model...")
    
    if (missVal) {
      GPCCAfit <- EM_missingDAT_Bern(Y.list = Y.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                                     EarlyStop = EarlyStop, niter.ES = niter.ES,
                                     W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
    }
    
    if (!missVal) {
      GPCCAfit <- EM_completeDAT_Bern(Y.list = Y.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                                     EarlyStop = EarlyStop, niter.ES = niter.ES,
                                     W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
    }
  }
  
  return(GPCCAfit)
}


