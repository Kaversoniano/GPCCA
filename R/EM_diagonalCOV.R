#' Fit the GPCCA Model with Simplified Covariance Structure
#'
#' Implementation of Generalized Probabilistic Canonical Correlation Analysis (GPCCA) 
#'   on a general multi-modal dataset that may contain missing values, if the 
#'   error covariance matrix \eqn{\mathbf{\Psi}} is assumed to be strict diagonal. 
#'   It is recommended for use when the total dimension of data (with all 
#'   modalities combined) is extremely high. In this case, ridge regularization 
#'   is not applied, without needing to specify the value of \eqn{\lambda}. 
#'   A sub-function invoked by the primary function \code{GPCCAmodel}. 
#'   See \code{\link{GPCCAmodel}} for more details of the GPCCA model.
#'
#' @param X.list a list of numeric matrices. A multi-modal dataset of \eqn{R} 
#'   modalities, where each matrix \eqn{\mathbf{X}^{(r)}} is a data modality, 
#'   with \eqn{m_r} rows denoting features and \eqn{n} columns denoting samples 
#'   (\eqn{1 \le r \le R}). All modalities must have the same sample size \eqn{n}, 
#'   with the ordering of samples matched to each other. Artificial modality-wise 
#'   missing values should be introduced by the user if non-matching samples are present.
#' @param d an integer. The size of target dimension \eqn{d}, i.e. the number of 
#'   latent factors in low-dimensional subspace. (Default: \code{2})
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
#' @seealso \code{\link{GPCCAmodel}}, \code{\link{EM_completeDAT}}, \code{\link{EM_missingDAT}}
#'
#' @examples
#' \dontrun{
#' 
#' ## Generate a 3-modality continuous dataset with 20% MCARs
#' DAT <- example_MultiModalData_sim(dataType = "continuous", missVal = TRUE, propMCAR = 0.2)
#' 
#' ## Fit GPCCA model to the incomplete example multi-modal dataset
#' GPCCA.fit <- EM_diagonalCOV(X.list = DAT, d = 4)
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

EM_diagonalCOV <- function(X.list, d = 2, tol = 0.0001, maxiter = 100,
                           EarlyStop = TRUE, niter.ES = 5,
                           W.init = "PCA", verbose = 1, plot_RMSE = TRUE) {
  # X.list <- list(...)
  
  X <- do.call(rbind, X.list)
  colnames(X) <- NULL
  rownames(X) <- NULL
  
  O <- (!is.na(X)) * 1 # indicator matrix (1 = observed, 0 = missing)
  
  m <- nrow(X) # number of total dimensions
  n <- ncol(X) # sample size
  
  ### Parameter Initialization (recommended)
  mu0 <- apply(X, 1, mean, na.rm = TRUE) # fixed
  
  if (W.init == "PCA") {
    W0 <- pcaMethods::pca(t(X), nPcs = d, method = "ppca")@loadings # PPCA
  }
  if (W.init == "rand.std.norm") {
    W0 <- matrix(rnorm(n = nrow(X) * d, 0, 1), nrow = nrow(X), ncol = d) # random
  }
  
  Psi0 <- diag(apply(X, 1, var, na.rm = TRUE))
  
  ### Starting Point in Parameter Space
  mu <- mu0 # mean vector (iter = 0)
  W <- W0 # loading matrix (iter = 0)
  Psi <- Psi0 # error covariance matrix (iter = 0)
  
  ### Additional Preparations
  Z.new <- matrix(1, nrow = d, ncol = n)
  Z.old <- matrix(0, nrow = d, ncol = n)
  RMSE <- sqrt(mean((as.vector(Z.new) - as.vector(Z.old))^2))
  iter <- 0 # index of iteration
  
  iters <- NULL
  t.dfs <- NULL
  RMSEs <- NULL
  
  ### Iterations of EM Algorithm
  while(RMSE > tol & iter < maxiter) {
    t.start <- Sys.time() # start time of the current iteration
    
    ########## E-step ##########
    ## Indices of observed and missing data
    ind.O <- lapply(1:n, function(k) which(O[, k] == 1))
    ind.M <- lapply(1:n, function(k) which(O[, k] == 0))
    
    fCore.rs <- fCore_diagcov(X, ind.O, ind.M,
                              as.matrix(t(W)), matrix(mu, nrow = m, ncol = 1), as.matrix(Psi),
                              n, m, d)
    
    sum_E.z <- fCore.rs$sum_E_z
    sum_E.zz <- fCore.rs$sum_E_zz
    sum_E.x <- fCore.rs$sum_E_x
    sum_E.xx <- fCore.rs$sum_E_xx
    sum_E.xz <- fCore.rs$sum_E_xz
    
    ## Sum of G_k terms
    sum_G <- sum_E.xx + n * Matrix::tcrossprod(mu) + W %*% sum_E.zz %*% t(W) - 2 * sum_E.x %*% t(mu) + 2 * W %*% sum_E.z %*% t(mu) - 2 * W %*% t(sum_E.xz)
    sum_G <- Matrix::Matrix(sum_G)
    sum_G <- diag(diag(sum_G)) # strict diagonalization
    sum_G <- Matrix::Matrix(sum_G, sparse = TRUE)
    
    ########## M-step ##########
    mu <- as.vector((sum_E.x - W %*% sum_E.z) / n) # mean vector (updated)
    W <- (sum_E.xz - Matrix::tcrossprod(mu, sum_E.z)) %*% Matrix::solve(sum_E.zz, tol = 0) # loading matrix (updated)
    Psi <- sum_G / n # error covariance matrix (updated)
    
    ## Estimated Expected Complete Log-likelihood E[log(Lc)]
    Z.old <- Z.new
    Z.new <- fCore.rs$Z
    RMSE <- sqrt(mean((as.vector(Z.new) - as.vector(Z.old))^2))
    RMSEs <- c(RMSEs, RMSE)
    
    ########## Logs ##########
    iter <- iter + 1 # increment of iteration
    iters <- c(iters, iter)
    
    t.end <- Sys.time() # end time of the current iteration
    t.df <- t.end - t.start # elapsed time of the current iteration
    t.dfs <- c(t.dfs, t.df)
    
    if (verbose) {
      cat(paste0("Iteration ", iter, " takes ", round(as.numeric(t.df, units = "secs"), 4), "s",
                 " with RMSE ", round(RMSE, 8)), "\n")
    }
    
    gc()
    
    ########## Early Stop ##########
    if (EarlyStop) {
      if ((iter >= 10 + niter.ES) & all(diff(tail(RMSEs, niter.ES + 1)) > 0)) break
    }
  }
  
  ### Graphs
  if (plot_RMSE) {
    plot(x = iters, y = log10(RMSEs), type = "l", lwd = 2, col = "red",
         xlab = "Iteration", ylab = "log10(RMSE)",
         main = "Logs: Convergence Check")
  }
  
  ### Values
  return(list(Z = Z.new, mu = mu, W = W, Psi = Psi))
}


