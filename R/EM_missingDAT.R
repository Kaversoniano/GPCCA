#' Fit the GPCCA Model on a General Multi-modal Dataset
#'
#' Implementation of Generalized Probabilistic Canonical Correlation Analysis (GPCCA) 
#'   on a general multi-modal dataset that may contain missing values. 
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
#' @seealso \code{\link{GPCCAmodel}}, \code{\link{EM_completeDAT}}, \code{\link{EM_diagonalCOV}}
#'
#' @examples
#' \dontrun{
#' 
#' ## Generate a 3-modality continuous dataset with 20% MCARs
#' DAT <- example_MultiModalData_sim(dataType = "continuous", missVal = TRUE, propMCAR = 0.2)
#' 
#' ## Fit GPCCA model to the incomplete example multi-modal dataset
#' GPCCA.fit <- EM_missingDAT(X.list = DAT, d = 4, lambda = 0.5)
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

EM_missingDAT <- function(X.list, d = 2, lambda = 0.5, tol = 0.0001, maxiter = 100,
                          EarlyStop = TRUE, niter.ES = 5, force_blkInv = FALSE,
                          W.init = "PCA", verbose = 1, plot_RMSE = TRUE) {
  # X.list <- list(...)
  
  X <- do.call(rbind, X.list)
  colnames(X) <- NULL
  rownames(X) <- NULL
  
  O <- (!is.na(X)) * 1 # indicator matrix (1 = observed, 0 = missing)
  
  m <- nrow(X) # number of total dimensions
  n <- ncol(X) # sample size
  
  ### Determining the Necessity of Block-wise Inversion
  ms <- sapply(X.list, nrow) # number of dimensions in each modality
  nms <- length(ms) # number of modalities
  
  blockwise_inv <- 0 # initial state of block-wise inversion prompt
  if (force_blkInv) blockwise_inv <- 1
  if (m > 1000) blockwise_inv <- 1
  if (sum(ms > 500) >= 1) blockwise_inv <- 1
  
  if (blockwise_inv == 1) {
    U.index <- cumsum(ms)
    L.index <- cumsum(ms) - ms + 1
    
    ms.index <- list()
    for (j in 1:nms) {
      ms.index[[j]] <- L.index[j]:U.index[j]
    }
    
    message("Blockwise inversion of the error covariance matrix is enabled.")
  } else {
    message("Blockwise inversion of the error covariance matrix is disabled.")
  }
  
  ### Parameter Initialization (recommended)
  mu0 <- apply(X, 1, mean, na.rm = TRUE) # fixed
  
  if (W.init == "PCA") {
    W0 <- pcaMethods::pca(t(X), nPcs = d, method = "ppca")@loadings # PPCA
  }
  if (W.init == "rand.std.norm") {
    W0 <- matrix(rnorm(n = nrow(X) * d, 0, 1), nrow = nrow(X), ncol = d) # random
  }
  
  Psi0 <- Matrix::bdiag(lapply(lapply(X.list, nrow), randcorr::randcorr)) # random
  D0 <- diag(apply(X, 1, sd, na.rm = TRUE))
  Psi0 <- D0 %*% Psi0 %*% D0
  
  block.diag.ind <- Psi0 != 0 # block diagonal structure
  
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
    
    if (blockwise_inv == 0) {
      fCore.rs <- fCore_default0(X, ind.O, ind.M,
                                 as.matrix(t(W)), matrix(mu, nrow = m, ncol = 1), as.matrix(Psi),
                                 n, m, d)
    }

    if (blockwise_inv == 1) {
      fCore.rs <- fCore_default1(X, ind.O, ind.M,
                                 as.matrix(t(W)), matrix(mu, nrow = m, ncol = 1), as.matrix(Psi),
                                 n, m, d, nms, ms.index)
    }
    
    sum_E.z <- fCore.rs$sum_E_z
    sum_E.zz <- fCore.rs$sum_E_zz
    sum_E.x <- fCore.rs$sum_E_x
    sum_E.xx <- fCore.rs$sum_E_xx
    sum_E.xz <- fCore.rs$sum_E_xz
    
    ## Sum of G_k terms
    sum_G <- sum_E.xx + n * Matrix::tcrossprod(mu) + W %*% sum_E.zz %*% t(W) - 2 * sum_E.x %*% t(mu) + 2 * W %*% sum_E.z %*% t(mu) - 2 * W %*% t(sum_E.xz)
    sum_G <- Matrix::Matrix(sum_G)
    sum_G[!block.diag.ind] <- 0 # block diagonalization
    sum_G <- Matrix::Matrix(sum_G, sparse = TRUE)
    
    ########## M-step ##########
    mu <- as.vector((sum_E.x - W %*% sum_E.z) / n) # mean vector (updated)
    W <- (sum_E.xz - Matrix::tcrossprod(mu, sum_E.z)) %*% Matrix::solve(sum_E.zz, tol = 0) # loading matrix (updated)
    Psi <- t(sum_G) / n # error covariance matrix (updated)
    Psi <- Psi + (1 / lambda - 1) * diag(diag(Psi)) # numerical enhancement
    
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


