#' Fit the GPCCA Model
#'
#' Implementation of Generalized Probabilistic Canonical Correlation Analysis (GPCCA). 
#'   GPCCA performs both data integration and joint dimensionality reduction on 
#'   a multi-modal dataset. The key output is the resulting projections, i.e. 
#'   joint low-dimensional embeddings. The reconstructed data representations 
#'   learnt by GPCCA can be directly used for further downstream analysis of 
#'   user's preference.
#'
#' @param X.list a list of numeric matrices. A multi-modal dataset where each 
#'    matrix \eqn{\mathbf{X}^{(r)}} is a data modality, with \eqn{m_r} rows 
#'    denoting features and \eqn{n} columns denoting samples. 
#'    All modalities must have the same sample size \eqn{n}, with the ordering 
#'    of samples matched to each other. Artificial modality-wise missing values 
#'    should be introduced by the user if non-matching samples are present.
#' @param d an integer. The size of target dimension \eqn{d}, i.e. the number of 
#'   latent factors in low-dimensional subspace. (Default: \code{2})
#' @param lambda a numeric scalar. The ridge regularization parameter \eqn{\lambda}, 
#'   in the range of \eqn{(0, 1]}. A smaller \eqn{\lambda} corresponds to heavier 
#'   penalty, which leads to a more sparse (diagonal-like) structure of the 
#'   error covariance matrix \eqn{\mathbf{\Psi}}. In the extreme case, \eqn{\lambda = 1} 
#'   means no ridge regularization is applied. (Default: \code{0.5})
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
#'   covariance matrix \eqn{\mathbf{\Psi}} be activated? It is automatically 
#'   turned on if either the dimension of any data modality \eqn{m_r} is larger 
#'   than \eqn{500} or the total dimension \eqn{m = \sum_{r=1}^{R}m_r} is larger 
#'   than \eqn{1000}. (Default: \code{FALSE})
#' @param W.init One of \code{"PCA"} and \code{"rand.std.norm"}, 
#'   indicating which method is used for initializing the loading matrix 
#'   \eqn{\mathbf{W}}. \code{"rand.std.norm"} initializes all elements of 
#'   \eqn{\mathbf{W}} with i.i.d. standard normal random variates. \code{"PCA"} 
#'   uses Randomized Principal Component Analysis (RPCA) if all data modalities 
#'   are complete without missingness or Probabilistic Principal Component Analysis 
#'   (PPCA) if any missing value is present in the multi-modal data \code{X.list}. 
#'   (Default: \code{"PCA"})
#' @param verbose a binary. If set to \code{0}, it prevents the printing of 
#'   running time and RMSE in every iteration. (Default: \code{1})
#' @param plot_RMSE a logical. After the EM algorithm is completed, should the 
#'   RMSEs in iterations be plotted? (Default: \code{TRUE})
#' @param missVal a logical. Does the input multi-modal data \code{X.list} 
#'   contain any missing value \code{NA}? (Default: \code{TRUE})
#' @param diagCov a logical. Should the strict diagonal structure be used for 
#'   the error covariance matrix \eqn{\mathbf{\Psi}}? It is only recommended 
#'   when the total dimension of data \eqn{m} (total number of features) is 
#'   extremely large. (Default: \code{FALSE})
#'
#' @details
#' \bold{GPCCA model:} 
#'   Suppose there are \eqn{n} subjects on which \eqn{R} 
#'   modalities of data are collected. The model is specified as:
#' \deqn{\mathbf{X} = \mathbf{W}\mathbf{Z} + \boldsymbol{\mu} + \mathbf{E}}
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
#'   The EM algorithm for GPCCA stops when the difference in \eqn{\mathbf{Z}} 
#'   between two consecutive iterations (\eqn{t} and \eqn{t+1}) is smaller than 
#'   a threshold \eqn{\tau}: 
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
#' @seealso \code{\link{EM_completeDAT}}, \code{\link{EM_missingDAT}}, \code{\link{EM_diagonalCOV}}
#'
#' @examples
#' \dontrun{
#' 
#' ## Fit GPCCA model to the example multi-modal data
#' GPCCA.fit <- GPCCAmodel(X.list = example_MultiModalData_list, d = 5, lambda = 0.5)
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
  
  stopifnot(length(X.list) >= 2,
            d %% 1 == 0, d >= 2,
            lambda >= 0, lambda <= 1,
            tol > 0,
            maxiter %% 1 == 0, maxiter >= 1,
            EarlyStop %in% c(TRUE, FALSE),
            niter.ES %% 1 == 0, niter.ES >= 2,
            force_blkInv %in% c(TRUE, FALSE),
            W.init %in% c("PCA", "rand.std.norm"),
            verbose %in% c(1, 0),
            plot_RMSE %in% c(TRUE, FALSE),
            missVal %in% c(TRUE, FALSE),
            diagCov %in% c(TRUE, FALSE))
  
  if (diagCov) {
    GPCCAfit <- EM_diagonalCOV(X.list = X.list, d = d, tol = tol, maxiter = maxiter,
                               EarlyStop = EarlyStop, niter.ES = niter.ES,
                               W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
  }
  
  if (missVal) {
    GPCCAfit <- EM_missingDAT(X.list = X.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                              EarlyStop = EarlyStop, niter.ES = niter.ES, force_blkInv = force_blkInv,
                              W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
  }
  
  if (!missVal) {
    GPCCAfit <- EM_completeDAT(X.list = X.list, d = d, lambda = lambda, tol = tol, maxiter = maxiter,
                               EarlyStop = EarlyStop, niter.ES = niter.ES, force_blkInv = force_blkInv,
                               W.init = W.init, verbose = verbose, plot_RMSE = plot_RMSE)
  }
  
  return(GPCCAfit)
}


