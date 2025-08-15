#' Select the Target Dimension for GPCCA Model
#'
#' Generalized Probabilistic Canonical Correlation Analysis (GPCCA) involves two 
#'   hyper-parameters: the size of target dimension \eqn{d} and the ridge 
#'   regularization parameter \eqn{\lambda}. \code{GPCCAselect} provides a 
#'   built-in technique for selecting the best \eqn{d} given a set of candidates. 
#'   This model selection framework is computationally intensive when the 
#'   dimensionality of data is considerably high. Thus it is recommended to 
#'   perform such model selection on a subset of data samples using highly 
#'   informative features only.
#'   
#'   The default value \eqn{\lambda = 0.5} leads to good performance in general, 
#'   but there is no built-in method for selecting \eqn{\lambda} due to the 
#'   unsupervised nature of GPCCA. Users may need to use a customized measure of 
#'   performance and find the optimal \eqn{\lambda} adaptively. 
#'   See \code{\link{GPCCAmodel}} for more details of the GPCCA model.
#'
#' @param X.list a list of numeric matrices. A multi-modal dataset of \eqn{R} 
#'   modalities, where each matrix \eqn{\mathbf{X}^{(r)}} is a data modality, 
#'   with \eqn{m_r} rows denoting features and \eqn{n} columns denoting samples 
#'   (\eqn{1 \le r \le R}). All modalities must have the same sample size \eqn{n}, 
#'   with the ordering of samples matched to each other. Artificial modality-wise 
#'   missing values should be introduced by the user if non-matching samples are present.
#' @param d.set an integer vector. A set of candidate values for the size of 
#'   target dimension \eqn{d}, i.e. the number of latent factors 
#'   in low-dimensional subspace. (Default: \code{2:5})
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
#'   It affects nothing if \code{diagCov = TRUE}. (Default: \code{TRUE})
#' @param W.init One of \code{"PCA"} and \code{"rand.std.norm"}, 
#'   indicating which method is used for initializing the loading matrix 
#'   \eqn{\mathbf{W}}. \code{"rand.std.norm"} initializes all elements of 
#'   \eqn{\mathbf{W}} with i.i.d. standard normal random variates. \code{"PCA"} 
#'   uses Randomized Principal Component Analysis (RPCA) if all data modalities 
#'   are complete without missingness or Probabilistic Principal Component Analysis 
#'   (PPCA) if any missing value is present in the multi-modal data \code{X.list}. 
#'   (Default: \code{"rand.std.norm"})
#' @param missVal a logical. Does the input multi-modal data \code{X.list} 
#'   contain any missing value \code{NA}? If \code{X.list} is known to contain 
#'   no missing value, setting \code{missVal = FALSE} can reduce runtime 
#'   significantly. (Default: \code{TRUE})
#' @param diagCov a logical. Should the strict diagonal structure be used for 
#'   the error covariance matrix \eqn{\mathbf{\Psi}}? It is only recommended 
#'   when the total dimension of data \eqn{m} (total number of features) is 
#'   extremely large. (Default: \code{FALSE})
#' @param B an integer. The number of different initializations \eqn{B} 
#'   to be used for model selection. (Default: \code{5})
#' @param seed an integer. The seed for random number generation. 
#'   Set the seed for exact reproduction of your results. (Default: \code{1234})
#' @param N.cores an integer. The number of cores to use for parallel computing. 
#'   It is suggested to set it equal to the number of initializations \eqn{B}. 
#'   Must be exactly \eqn{1} on Windows, which uses the master process. 
#'   (Default: \code{1})
#'
#' @details
#' \bold{Model selection on target dimension:} 
#'   The size of target dimension \eqn{d} (the number of latent factors) is a 
#'   hyper-parameter of GPCCA model. Given a set of candidate values 
#'   \eqn{d_1 < d_2 < \dots < d_G}, for each \eqn{d_g}, \eqn{B} different 
#'   initializations are used to fit the model. With Louvain clustering on the 
#'   fitted latent factors, the \eqn{B} clustering results are aggregated into 
#'   a matrix \eqn{\mathbf{L}_g \in \text{R}^{n \times B}}. Then, the consensus 
#'   matrix \eqn{\mathbf{C}_g \in \text{R}^{n \times n}} is computed based on 
#'   \eqn{\mathbf{L}_g}. To assess the consistency of the \eqn{B} clustering 
#'   results, the following consensus score is defined for measuring 
#'   the agreement between multiple results under \eqn{d_g}: 
#'   \deqn{\mathcal{H}_g = \sum\limits_{i<j}\mathbf{C}_{g,ij}\log_2(\mathbf{C}_{g,ij})} 
#'   And the optimal choice of \eqn{d} is selected as the candidate value 
#'   \eqn{d_{g^*}} which corresponds to the highest consensus score: 
#'   \deqn{g^* = \text{argmax}_{1 \le g \le G}\mathcal{H}_g}
#'
#' @return A list of 2 is returned, including:
#' \describe{
#'   \item{\code{consensus.score}}{a named numeric vector; stores the consensus 
#'     scores under all candidate values for the target dimension.}
#'   \item{\code{optim.d}}{an integer (vector); gives the optimal choice(s) 
#'     of the target dimension.}
#' }
#'
#' @seealso \code{\link{GPCCAmodel}}, \code{\link{EM_completeDAT}}, 
#'   \code{\link{EM_missingDAT}}, \code{\link{EM_diagonalCOV}}
#'
#' @examples
#' \dontrun{
#' 
#' ## I: Select the target dimension for GPCCA model on incomplete data
#' ## (`GPCCAmodel` can handle both complete and incomplete data)
#' DAT <- example_MultiModalData_sim(missVal = TRUE)
#' GPCCA.ds <- GPCCAselect(X.list = DAT,
#'                         d.set = c(2, 4, 6, 8, 10),
#'                         B = 5, seed = 1234, N.cores = 5)
#' 
#' ## II: Select the target dimension for GPCCA model on complete data
#' ## (`GPCCAmodel` can handle both complete and incomplete data)
#' DAT <- example_MultiModalData_sim(missVal = FALSE)
#' GPCCA.ds <- GPCCAselect(X.list = DAT,
#'                         d.set = c(2, 4, 6, 8, 10),
#'                         B = 5, seed = 1234, N.cores = 5)
#' 
#' ## III: Select the target dimension for GPCCA model on complete data
#' ## (setting `missVal = FALSE` explicitly to reduce runtime)
#' DAT <- example_MultiModalData_sim(missVal = FALSE)
#' GPCCA.ds <- GPCCAselect(X.list = DAT,
#'                         d.set = c(2, 4, 6, 8, 10), missVal = FALSE,
#'                         B = 5, seed = 1234, N.cores = 5)
#' 
#' ## Output consensus scores under varying target dimensions
#' GPCCA.ds$consensus.score
#' 
#' ## Output the optimal choice(s) of target dimension
#' GPCCA.ds$optim.d
#' }

GPCCAselect <- function(X.list, d.set = 2:5, lambda = 0.5, tol = 0.0001, maxiter = 100,
                        EarlyStop = TRUE, niter.ES = 5, force_blkInv = TRUE,
                        W.init = "rand.std.norm", missVal = TRUE, diagCov = FALSE,
                        B = 5, seed = 1234, N.cores = 1) {
  
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
  
  ### Argument: d.set
  stopifnot(!is.na(d.set), is.numeric(d.set), length(d.set) >= 2)
  if (!all(d.set >= 2)) {
    stop("All candidate values of the target dimension 'd.set' must be at least 2.")
  }
  if (!all(d.set %% 1 == 0)) {
    stop("All candidate values of the target dimension 'd.set' must be integers.")
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
  
  ### Argument: missVal
  stopifnot(!is.na(missVal), is.logical(missVal), length(missVal) == 1)
  
  ### Argument: diagCov
  stopifnot(!is.na(diagCov), is.logical(diagCov), length(diagCov) == 1)
  
  ### Argument: B
  stopifnot(!is.na(B), is.numeric(B), length(B) == 1)
  if (!(B >= 2)) {
    stop("For model selection, the number of different initializations 'B' must be at least 2.")
  }
  if (!(B %% 1 == 0)) {
    stop("For model selection, the number of different initializations 'B' must be an integer.")
  }
  
  ### Argument: seed
  stopifnot(!is.na(seed), is.numeric(seed), length(seed) == 1)
  if (!(seed %% 1 == 0)) {
    stop("The seed for random number generation 'seed' must be an integer.")
  }
  
  ### Argument: N.cores
  stopifnot(!is.na(N.cores), is.numeric(N.cores), length(N.cores) == 1)
  if (!(N.cores >= 1)) {
    stop("The number of cores to use for parallel computing 'N.cores' must be at least 1.")
  }
  if (!(N.cores %% 1 == 0)) {
    stop("The number of cores to use for parallel computing 'N.cores' must be an integer.")
  }
  
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
  
  ##### Multiple GPCCA models with varying target dimensions -------------------
  message("Fitting multiple GPCCA models with varying target dimensions...")
  
  ds <- as.character(sort(d.set))
  GPCCA.fits.ds <- vector(mode = "list", length = length(ds))
  names(GPCCA.fits.ds) <- ds
  
  loc.d <- 0
  for (di in ds) {
    loc.d <- loc.d + 1
    
    seeds <- seed * (1:B)
    
    f_run_GPCCA <- function(sid) {
      set.seed(sid)
      
      if (diagCov) {
        GPCCA.fit <- EM_diagonalCOV(X.list = X.list, d = as.numeric(di),
                                    tol = tol, maxiter = maxiter,
                                    EarlyStop = EarlyStop, niter.ES = niter.ES,
                                    W.init = W.init, verbose = 0, plot_RMSE = FALSE)
      }
      
      if (missVal & !diagCov) {
        GPCCA.fit <- EM_missingDAT(X.list = X.list, d = as.numeric(di), lambda = lambda,
                                   tol = tol, maxiter = maxiter,
                                   EarlyStop = EarlyStop, niter.ES = niter.ES, force_blkInv = force_blkInv,
                                   W.init = W.init, verbose = 0, plot_RMSE = FALSE)
      }
      
      if (!missVal & !diagCov) {
        GPCCA.fit <- EM_completeDAT(X.list = X.list, d = as.numeric(di), lambda = lambda,
                                    tol = tol, maxiter = maxiter,
                                    EarlyStop = EarlyStop, niter.ES = niter.ES, force_blkInv = force_blkInv,
                                    W.init = W.init, verbose = 0, plot_RMSE = FALSE)
      }
      
      return(GPCCA.fit)
    }
    
    GPCCA.fits <- parallel::mclapply(X = seeds, FUN = f_run_GPCCA, mc.cores = N.cores)
    
    GPCCA.fits.ds[[loc.d]] <- GPCCA.fits
    
    message(B, " fits of GPCCA model are obtained for target dimension " , as.numeric(di), ".")
  }
  
  ##### Summary of models to obtain consensus score ----------------------------
  message("Combining multiple clustering results under different initializations...")
  
  rs.CS <- rep(NA, length(ds))
  names(rs.CS) <- ds
  
  loc.d <- 0
  for (di in ds) {
    loc.d <- loc.d + 1
    
    GPCCA.fits <- GPCCA.fits.ds[[loc.d]]
    
    ### Louvain clustering on the resulting projections from GPCCA
    set.seed(seed)
    rs.cl <- lapply(X = GPCCA.fits, FUN = LouvainClustering)
    
    ### Combination of multiple clustering results
    rs.cl.rep <- Reduce(cbind, rs.cl)
    
    ### Calculation of consensus matrix
    consensus.mat <- diceR::consensus_matrix(data = rs.cl.rep)
    
    ### Computation of empirical consensus score
    consensus.mat.info <- consensus.mat[lower.tri(consensus.mat)]
    
    if (all(apply(rs.cl.rep, 2, function(v) length(unique(v))) == 1)) {
      rs.CS[di] <- Inf
    } else {
      rs.CS[di] <- sapply(consensus.mat.info, ConsensusScore) %>% sum()
    }
  }
  
  ##### Selection of target dimension with consensus score ---------------------
  message("Finding the optimal choice of target dimension based on consensus scores...")
  
  d.selector <- function(CSs) {
    d.vals <- as.numeric(names(CSs))
    names(CSs) <- paste0("d", names(CSs))
    d.best <- which(CSs == max(CSs))
    return(d.vals[d.best])
  }
  
  optim.d <- d.selector(rs.CS)
  
  ##### ------------------------------------------------------------------------
  message("All done!")
  
  return(list(consensus.score = rs.CS, optim.d = optim.d))
}


