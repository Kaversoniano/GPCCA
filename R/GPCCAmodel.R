#' Fit the GPCCA Model
#'
#' Implementation of Generalized Probabilistic Canonical Correlation Analysis (GPCCA). 
#'   GPCCA performs both data integration and joint dimensionality reduction on 
#'   a multi-modal dataset. The key output is the resulting projections, i.e. 
#'   joint low-dimensional embeddings. The reconstructed data representations 
#'   learnt by GPCCA can be directly used for further downstream analysis of 
#'   users' preference. See \code{\link{GPCCAselect}} for a brief discussion and 
#'   a proposed technique of model selection on target dimension.
#'   
#' For high-dimensional data, feature selection is strongly recommended so that 
#'   only the highly informative features are included in the model fit. This helps 
#'   to reduce the computational resources required and boost the computational speed.
#'
#' @param X.list a list of numeric matrices. A multi-modal dataset of \eqn{R} 
#'   modalities, where each matrix \eqn{\mathbf{X}^{(r)}} is a data modality, 
#'   with \eqn{m_r} rows denoting features and \eqn{n} columns denoting samples 
#'   (\eqn{1 \le r \le R}). All modalities must have the same sample size \eqn{n}, 
#'   with the ordering of samples matched to each other. Artificial modality-wise 
#'   missing values should be introduced by the user if non-matching samples are present.
#' @param d an integer. The size of target dimension \eqn{d}, i.e. the number of 
#'   latent factors in low-dimensional subspace. (Default: \code{2})
#' @param lambda a numeric scalar. The ridge regularization parameter \eqn{\lambda}, 
#'   in the range of \eqn{(0, 1]}. A smaller \eqn{\lambda} corresponds to heavier 
#'   penalty, which leads to a more sparse (diagonal-like) structure of the 
#'   error covariance matrix \eqn{\mathbf{\Psi}}. In the extreme case, 
#'   \eqn{\lambda = 1} means no ridge regularization is applied. 
#'   It affects nothing if \code{diagCov = TRUE}. (Default: \code{0.5})
#' @param tol a numeric scalar. Tolerance of the RMSE measuring the difference 
#'   in the matrix of latent factors \eqn{\mathbf{Z}} between two consecutive 
#'   iterations. This tolerance threshold \eqn{\tau} determines the stopping rule 
#'   of the EM algorithm, and the RMSE is used to monitor the convergence.
#'   (Default: \code{0.0001})
#' @param maxiter an integer. The maximum number of iterations allowed in the 
#'   EM algorithm. (Default: \code{100})
#' @param EarlyStop a logical. Should early stop be used? If set to \code{TRUE}, 
#'   it stops the EM algorithm early. With early stop turned on, only sub-optimal 
#'   solution may be attained, but it helps avoid overfitting. (Default: \code{TRUE})
#' @param niter.ES an integer. The number of consecutive increase of RMSE allowed 
#'   to decide for early stop. It affects nothing if \code{EarlyStop = FALSE}. 
#'   (Default: \code{5})
#' @param force_blkInv a logical. Should blockwise inversion of the error 
#'   covariance matrix \eqn{\mathbf{\Psi}} be activated? A technique to boost 
#'   computation. It is automatically turned on if either the dimension of any 
#'   data modality \eqn{m_r} is larger than \eqn{500} or the total dimension 
#'   \eqn{m = \sum_{r=1}^{R}m_r} is larger than \eqn{1000}. 
#'   It affects nothing if \code{diagCov = TRUE}. (Default: \code{FALSE})
#' @param W.init One of \code{"PCA"} and \code{"rand.std.norm"}, 
#'   indicating which method is used for initializing the loading matrix 
#'   \eqn{\mathbf{W}}. \code{"rand.std.norm"} initializes all elements of 
#'   \eqn{\mathbf{W}} with i.i.d. standard normal random variates. \code{"PCA"} 
#'   uses Randomized Principal Component Analysis (RPCA) if all data modalities 
#'   are complete without missingness or Probabilistic Principal Component Analysis 
#'   (PPCA) if any missing value is present in the multi-modal data \code{X.list}. 
#'   (Default: \code{"PCA"})
#' @param verbose a binary. If set to \code{0}, it prevents the printing of 
#'   runtime and RMSE in every iteration. (Default: \code{1})
#' @param plot_RMSE a logical. After the EM algorithm is completed, should the 
#'   RMSEs in iterations be plotted? (Default: \code{TRUE})
#' @param missVal a logical. Does the input multi-modal data \code{X.list} 
#'   contain any missing value \code{NA}? If \code{X.list} is known to contain 
#'   no missing value, setting \code{missVal = FALSE} can reduce runtime 
#'   significantly. (Default: \code{TRUE})
#' @param diagCov a logical. Should the strict diagonal structure be used for 
#'   the error covariance matrix \eqn{\mathbf{\Psi}}? It is only recommended 
#'   when the total dimension of data \eqn{m} (total number of features) is 
#'   extremely large. (Default: \code{FALSE})
#'
#' @details
#' \bold{GPCCA model:} 
#'   Suppose there are \eqn{n} subjects on which \eqn{R} 
#'   modalities of data are collected. The model is specified as:
#'   \deqn{\mathbf{X} = \mathbf{W}\mathbf{Z} + \boldsymbol{\mu} + \mathbf{E}}
#' \itemize{
#'   \item Data matrix, consisting of \eqn{R} modalities: 
#'     \eqn{\mathbf{X} = [{\mathbf{X}^{(1)}}^\text{T}, \dots, 
#'     {\mathbf{X}^{(R)}}^\text{T}]^\text{T} \in \text{R}^{m \times n}} 
#'     where \eqn{\mathbf{X}^{(r)} \in \text{R}^{m_r \times n}}
#'   \item Loading matrix: 
#'     \eqn{\mathbf{W} \in \text{R}^{m \times d}}, constant parameter
#'   \item Mean vector: 
#'     \eqn{\boldsymbol{\mu} \in \text{R}^{m \times 1}}, constant parameter
#'   \item Error covariance matrix: 
#'     \eqn{\mathbf{\Psi} = \text{diag}(\mathbf{\Psi}^{(1)}, \dots, 
#'     \mathbf{\Psi}^{(R)}) \in \text{R}^{m \times m}} 
#'     with block-diagonal structure, constant parameter
#'   \item Matrix of latent factors: 
#'     \eqn{\mathbf{Z} \in \text{R}^{d \times n}} where 
#'     \eqn{\{\mathbf{Z}_{\cdot k}\}_{1 \le k \le n} \overset{\text{i.i.d.}}{\sim} 
#'     \mathcal{N}_d(\mathbf{0}, \mathbf{I})}, random
#'   \item Matrix of random error terms: 
#'     \eqn{\mathbf{E} \in \text{R}^{m \times n}} where 
#'     \eqn{\{\mathbf{E}_{\cdot k}\}_{1 \le k \le n} \overset{\text{i.i.d.}}{\sim} 
#'     \mathcal{N}_m(\mathbf{0}, \mathbf{\Psi})}, random
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
#'   The error covariance matrix \eqn{\mathbf{\Psi}} can be factorized as 
#'   \eqn{\mathbf{\Psi} = \mathbf{\Psi}_d^{\frac{1}{2}}\mathbf{R}\mathbf{\Psi}_d^{\frac{1}{2}}}, 
#'   where \eqn{\mathbf{\Psi}_d} is a diagonal matrix of marginal variances and 
#'   \eqn{\mathbf{R}} is the error correlation matrix. 
#'   With ridge regularization, the MPLE of \eqn{\mathbf{R}} is: 
#'   \deqn{\hat{\mathbf{R}}_{\text{ridge}} = 
#'   \lambda\hat{\mathbf{R}} + (1-\lambda)\mathbf{I}} where \eqn{\lambda \in (0, 1]} 
#'   and \eqn{\hat{\mathbf{R}}} denotes the original MLE without regularization. 
#'   As a result, the MPLE of \eqn{\mathbf{\Psi}} is: 
#'   \deqn{\hat{\mathbf{\Psi}}_{\text{ridge}} = 
#'   \hat{\mathbf{\Psi}} + (\dfrac{1}{\lambda} - 1)\hat{\mathbf{\Psi}}_d} 
#'   where \eqn{\lambda \in (0, 1]} and \eqn{\hat{\mathbf{\Psi}}} denotes the 
#'   original MLE without regularization. The ridge regularization parameter 
#'   \eqn{\lambda} manages the bias-variance trade-off, whose optimal value can 
#'   vary across datasets. Hence, multiple \eqn{\lambda} values should be tested 
#'   in exploratory analysis on a small subset to find the most suitable value.
#'
#' @return A list of 4 is returned, including:
#' \describe{
#'   \item{\code{Z}}{a numeric matrix; a matrix of size \eqn{d \times n} that 
#'     stores the fitted latent factors, i.e. the joint low-dimensional 
#'     embeddings of the original multi-modal data.}
#'   \item{\code{mu}}{a numeric vector; the estimated mean vector 
#'     \eqn{\boldsymbol{\mu}} of length \eqn{m}.}
#'   \item{\code{W}}{a numeric matrix; the estimated loading matrix 
#'     \eqn{\mathbf{W}} of size \eqn{m \times d}.}
#'   \item{\code{Psi}}{a sparse \code{Matrix}; the estimated error covariance matrix 
#'     \eqn{\mathbf{\Psi}} of size \eqn{m \times m}.}
#' }
#'
#' @seealso \code{\link{GPCCAselect}}, \code{\link{EM_completeDAT}}, 
#'   \code{\link{EM_missingDAT}}, \code{\link{EM_diagonalCOV}}
#'
#' @examples
#' \dontrun{
#' 
#' ## I: Fit GPCCA model to the incomplete example multi-modal dataset
#' ## (`GPCCAmodel` can handle both complete and incomplete data)
#' DAT <- example_MultiModalData_sim(missVal = TRUE)
#' GPCCA.fit <- GPCCAmodel(X.list = DAT, d = 4, lambda = 0.5)
#' 
#' ## II: Fit GPCCA model to the complete example multi-modal dataset
#' ## (`GPCCAmodel` can handle both complete and incomplete data)
#' DAT <- example_MultiModalData_sim(missVal = FALSE)
#' GPCCA.fit <- GPCCAmodel(X.list = DAT, d = 4, lambda = 0.5)
#' 
#' ## III: Fit GPCCA model to the complete example multi-modal dataset
#' ## (setting `missVal = FALSE` explicitly to reduce runtime)
#' DAT <- example_MultiModalData_sim(missVal = FALSE)
#' GPCCA.fit <- GPCCAmodel(X.list = DAT, d = 4, lambda = 0.5, missVal = FALSE)
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

GPCCAmodel <- function(X.list, d = 2, lambda = 0.5, tol = 0.0001, maxiter = 100,
                       EarlyStop = TRUE, niter.ES = 5, force_blkInv = FALSE,
                       W.init = "PCA", verbose = 1, plot_RMSE = TRUE,
                       missVal = TRUE, diagCov = FALSE) {
  
  ##### Check the validity of arguments ----------------------------------------
  message("Checking validity of arguments...")
  
  ### Argument: X.list
  stopifnot(!is.null(X.list), is.list(X.list))
  if (!all(sapply(X = X.list, FUN = function(mat) is.matrix(mat) && is.numeric(mat)))) {
    stop("Every element of the list 'X.list' must be a numeric matrix.")
  }
  if (!(length(X.list) >= 2)) {
    stop("In multi-modal dataset 'X.list', the number of modalities must be at least 2.")
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
  if (!(lambda > 0 & lambda <= 1)) {
    stop("The ridge regularization parameter 'lambda' must be in (0, 1].")
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
  
  ### Argument: force_blkInv
  stopifnot(!is.na(force_blkInv), is.logical(force_blkInv), length(force_blkInv) == 1)
  
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
  
  ### Argument: diagCov
  stopifnot(!is.na(diagCov), is.logical(diagCov), length(diagCov) == 1)
  
  ##### Check the advanced requirements for argument: X.list -------------------
  ##### (only suggestions are provided but no auto-fix is applied) -------------
  message("Checking requirements for the input multi-modal data...")
  
  ### Requirement of equal sample sizes for X.list
  if (length(unique(sapply(X = X.list, FUN = ncol))) != 1) {
    warning("All modalities in 'X.list' must have equal sample size, with matched ordering of samples. ",
            "Artificial modality-wise missingness should be introduced by the user if there are any non-matching samples.")
    stop("Sample sizes of the given ", length(X.list), " modalities are not equal.")
  }
  
  ### Requirement of no presence of problematic sample
  na.flags <- do.call(rbind, lapply(X = X.list, FUN = is.na))
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
      warning("If the input multi-modal data 'X.list' contains any missing value as NA, ",
              "the default setting 'missVal = TRUE' should be used.")
      stop("'missVal = FALSE' is specified explicitly but the input of 'X.list' contains missing values.")
    }
  }
  
  ### Requirement of no non-finite values
  X.cc <- do.call(rbind, X.list)
  if (any(is.infinite(X.cc) | is.nan(X.cc))) {
    stop("The input multi-modal data 'X.list' contains non-finite values (Inf/-Inf/NaN).")
  }
  
  ### Inconsistency due to redundant arguments specified when diagCov = TRUE
  if (diagCov == TRUE) {
    if (lambda != 0.5) {
      message("Setting 'diagCov = TRUE' disables ridge regularization. ",
              "Thus changing the ridge regularization parameter 'lambda' affects nothing.")
    }
    if (force_blkInv != FALSE) {
      message("Setting 'diagCov = TRUE' disables ridge regularization. ",
              "Thus specifying 'force_blkInv = TRUE' affects nothing, since diagonal is also block-diagonal.")
    }
  }
  
  ##### ------------------------------------------------------------------------
  message("All checks are passed!")
  
  ##### Obtain the fitted GPCCA model ------------------------------------------
  message("Start fitting the GPCCA model...")
  
  if (diagCov) {
    GPCCAfit <- EM_diagonalCOV(X.list = X.list, d = d, tol = tol, maxiter = maxiter,
                               EarlyStop = EarlyStop, niter.ES = niter.ES,
                               W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
  }
  
  if (missVal & !diagCov) {
    GPCCAfit <- EM_missingDAT(X.list = X.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                              EarlyStop = EarlyStop, niter.ES = niter.ES, force_blkInv = force_blkInv,
                              W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
  }
  
  if (!missVal & !diagCov) {
    GPCCAfit <- EM_completeDAT(X.list = X.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                               EarlyStop = EarlyStop, niter.ES = niter.ES, force_blkInv = force_blkInv,
                               W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
  }
  
  return(GPCCAfit)
}


