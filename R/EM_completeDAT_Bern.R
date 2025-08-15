#' Fit the GPCCA-Bern Model on a Complete Multi-modal Dataset of Binary (0/1) Observations
#'
#' Implementation of GPCCA-Bern on a complete multi-modal dataset of binary (0/1) observations 
#'   without any missing value. 
#'   A sub-function invoked by the primary function \code{GPCCAmodel2}. 
#'   See \code{\link{GPCCAmodel2}} for more details of the extension models from GPCCA.
#'
#' @param Y.list a list of numeric matrices. A multi-modal dataset of \eqn{R} 
#'   modalities, where each matrix \eqn{\mathbf{Y}^{(r)}} is a data modality, 
#'   with \eqn{m_r} rows denoting features and \eqn{n} columns denoting samples 
#'   (\eqn{1 \le r \le R}). All modalities must have the same sample size \eqn{n}, 
#'   with the ordering of samples matched to each other. Artificial modality-wise 
#'   missing values should be introduced by the user if non-matching samples are present.
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
#' @seealso \code{\link{GPCCAmodel2}}, \code{\link{EM_missingDAT_Bern}}
#'
#' @examples
#' \dontrun{
#' 
#' ## Generate a 3-modality binary dataset with no missing value
#' DAT <- example_MultiModalData_sim(dataType = "binary", missVal = FALSE)
#' 
#' ## Fit GPCCA-Bern model to the complete example multi-modal dataset
#' GPCCA.fit <- EM_completeDAT_Bern(Y.list = DAT, d = 4, lambda = 0.1)
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

EM_completeDAT_Bern <- function(Y.list, d = 2, lambda = 0.1, tol = 0.0001, maxiter = 50,
                                EarlyStop = TRUE, niter.ES = 5,
                                W.init = "PCA", verbose = 1, plot_RMSE = TRUE) {
  
  Y <- do.call(rbind, Y.list)
  colnames(Y) <- NULL
  rownames(Y) <- NULL
  
  m <- nrow(Y) # number of total dimensions
  n <- ncol(Y) # sample size
  
  ### Parameter Initialization (recommended)
  mu0 <- logit(apply(Y, 1, mean)) %>% as.vector() # fixed
  
  if (W.init == "PCA") {
    tempY <- scale(x = t(Y), center = TRUE, scale = FALSE)
    W0 <- rsvd::rpca(A = tempY, k = d, center = FALSE, scale = FALSE)$rotation # RPCA
  }
  if (W.init == "rand.std.norm") {
    W0 <- matrix(rnorm(n = nrow(Y) * d, 0, 1), nrow = nrow(Y), ncol = d) # random
  }
  
  ### Starting Point in Parameter Space
  mu <- mu0 # mean vector (iter = 0)
  W <- W0 # loading matrix (iter = 0)
  
  ### Additional Preparations
  Z.new <- matrix(0, nrow = d, ncol = n)
  Z.old <- matrix(1, nrow = d, ncol = n)
  RMSE <- sqrt(mean((as.vector(Z.new) - as.vector(Z.old))^2))
  iter <- 0 # index of iteration
  
  iters <- NULL
  t.dfs <- NULL
  RMSEs <- NULL
  
  ### RStan Configurations
  stan_model_code <- "
  data {
    int<lower=1> m;
    int<lower=1> d;
    vector[m] mu;
    matrix[m, d] W;
    int<lower=0, upper=1> Y_k[m];
  }
  
  parameters {
    vector[d] Z_k;
  }
  
  model {
    vector[m] X_k = mu + W * Z_k;
  
    Z_k ~ normal(0, 1);
    Y_k ~ bernoulli_logit(X_k);
  }
  "
  
  sm <- stan_model(model_code = stan_model_code)
  
  ### Iterations of EM Algorithm
  while(RMSE > tol & iter < maxiter) {
    t.start <- Sys.time() # start time of the current iteration
    
    ########## E-step ##########
    Z.samples <- vector("list", length = n)
    
    E.z <- matrix(0, nrow = d, ncol = n)
    
    base_data <- list(m = m, d = d, mu = mu, W = W)
    
    for(k in 1:n) {
      data_k <- base_data
      data_k$Y_k <- Y[, k]
      
      init_fun <- function() {
        list(Z_k = Z.new[, k])
      }
      
      fit_k <- rstan::sampling(sm, data = data_k, chains = 1, iter = 100, warmup = 50,
                               init = init_fun, refresh = 0, show_messages = FALSE)
      
      Z_k.samples <- t(extract(fit_k, "Z_k")$Z_k)
      Z.samples[[k]] <- Z_k.samples
      S <- ncol(Z_k.samples)
      
      E.z[, k] <- rowMeans(Z_k.samples)
    }
    
    ########## M-step ##########
    fit.param <- optim(par = as.vector(cbind(mu, W)), fn = Q_rcpp0, gr = G_rcpp0,
                       Y = Y, Z_samples = Z.samples,
                       lambda = lambda, n = n, m = m, d = d, S = S,
                       method = "L-BFGS-B", control = list(trace = 0, maxit = 100))
    param.mat <- matrix(fit.param$par, nrow = m, ncol = d + 1)
    mu <- param.mat[, 1]
    W <- param.mat[, -1]
    
    ## Estimated Expected Complete Log-likelihood E[log(Lc)]
    Z.old <- Z.new
    Z.new <- E.z
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
  return(list(Z = Z.new, mu = mu, W = W))
}


