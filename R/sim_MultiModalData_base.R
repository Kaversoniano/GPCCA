
sim_MultiModalData_base <- function(missVal = TRUE, propMCAR = 0.2) {
  ## Sample size of each group
  n.C1 <- 50
  n.C2 <- 50
  n.C3 <- 50
  n.C4 <- 50
  n.C5 <- 50
  n.C6 <- 50
  
  ## Dimension of each modality
  m1 <- 30
  m2 <- 60
  m3 <- 90
  
  ## Function of AR(1) correlation structure
  ar1_cor <- function(n, rho) {
    exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - (1:n - 1))
    return(rho^exponent)
  }
  
  ## Value of rho in AR(1)
  rho.val <- 0.5
  
  ## S2N ratio and proportion of signal
  s2n.r <- 1/2
  sgl.p <- s2n.r / (1 + s2n.r)
  
  ### Modality 1 - data generation
  mu1.u <- runif(round(m1 * sgl.p), 1, 2)
  mu1.v <- runif(round(m1 * sgl.p), -2, -1)
  
  SDs <- diag(4 * rbeta(round(m1 * sgl.p), 1, 1))
  sigma1.u <- SDs %*% ar1_cor(round(m1 * sgl.p), rho.val) %*% SDs
  sigma1.v <- SDs %*% ar1_cor(round(m1 * sgl.p), rho.val) %*% SDs
  
  X1 <- rbind(cbind(MASS::mvrnorm(n.C1, mu1.u, sigma1.u),
                    MASS::mvrnorm(n.C1, rep(0, round(m1 * (1 - sgl.p))), diag(round(m1 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C2, mu1.v, sigma1.v),
                    MASS::mvrnorm(n.C2, rep(0, round(m1 * (1 - sgl.p))), diag(round(m1 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C3, mu1.u, sigma1.u),
                    MASS::mvrnorm(n.C3, rep(0, round(m1 * (1 - sgl.p))), diag(round(m1 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C4, mu1.v, sigma1.v),
                    MASS::mvrnorm(n.C4, rep(0, round(m1 * (1 - sgl.p))), diag(round(m1 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C5, mu1.v, sigma1.v),
                    MASS::mvrnorm(n.C5, rep(0, round(m1 * (1 - sgl.p))), diag(round(m1 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C6, mu1.u, sigma1.u),
                    MASS::mvrnorm(n.C6, rep(0, round(m1 * (1 - sgl.p))), diag(round(m1 * (1 - sgl.p))))))
  
  ### Modality 2 - data generation
  mu2.u <- runif(round(m2 * sgl.p), 1, 2)
  mu2.v <- runif(round(m2 * sgl.p), -2, -1)
  
  SDs <- diag(4 * rbeta(round(m2 * sgl.p), 1, 1))
  sigma2.u <- SDs %*% ar1_cor(round(m2 * sgl.p), rho.val) %*% SDs
  sigma2.v <- SDs %*% ar1_cor(round(m2 * sgl.p), rho.val) %*% SDs
  
  X2 <- rbind(cbind(MASS::mvrnorm(n.C1, mu2.u, sigma2.u),
                    MASS::mvrnorm(n.C1, rep(0, round(m2 * (1 - sgl.p))), diag(round(m2 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C2, mu2.v, sigma2.v),
                    MASS::mvrnorm(n.C2, rep(0, round(m2 * (1 - sgl.p))), diag(round(m2 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C3, mu2.v, sigma2.v),
                    MASS::mvrnorm(n.C3, rep(0, round(m2 * (1 - sgl.p))), diag(round(m2 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C4, mu2.u, sigma2.u),
                    MASS::mvrnorm(n.C4, rep(0, round(m2 * (1 - sgl.p))), diag(round(m2 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C5, mu2.u, sigma2.u),
                    MASS::mvrnorm(n.C5, rep(0, round(m2 * (1 - sgl.p))), diag(round(m2 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C6, mu2.v, sigma2.v),
                    MASS::mvrnorm(n.C6, rep(0, round(m2 * (1 - sgl.p))), diag(round(m2 * (1 - sgl.p))))))
  
  ### Modality 3 - data generation
  mu3.u <- runif(round(m3 * sgl.p), 1, 2)
  mu3.v <- runif(round(m3 * sgl.p), -2, -1)
  
  SDs <- diag(4 * rbeta(round(m3 * sgl.p), 1, 1))
  sigma3.u <- SDs %*% ar1_cor(round(m3 * sgl.p), rho.val) %*% SDs
  sigma3.v <- SDs %*% ar1_cor(round(m3 * sgl.p), rho.val) %*% SDs
  
  X3 <- rbind(cbind(MASS::mvrnorm(n.C1, mu3.v, sigma3.v),
                    MASS::mvrnorm(n.C1, rep(0, round(m3 * (1 - sgl.p))), diag(round(m3 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C2, mu3.u, sigma3.u),
                    MASS::mvrnorm(n.C2, rep(0, round(m3 * (1 - sgl.p))), diag(round(m3 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C3, mu3.u, sigma3.u),
                    MASS::mvrnorm(n.C3, rep(0, round(m3 * (1 - sgl.p))), diag(round(m3 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C4, mu3.v, sigma3.v),
                    MASS::mvrnorm(n.C4, rep(0, round(m3 * (1 - sgl.p))), diag(round(m3 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C5, mu3.u, sigma3.u),
                    MASS::mvrnorm(n.C5, rep(0, round(m3 * (1 - sgl.p))), diag(round(m3 * (1 - sgl.p))))),
              cbind(MASS::mvrnorm(n.C6, mu3.v, sigma3.v),
                    MASS::mvrnorm(n.C6, rep(0, round(m3 * (1 - sgl.p))), diag(round(m3 * (1 - sgl.p))))))
  
  ### Output the simulated data
  if (!missVal) {
    return(list(M1 = t(X1), M2 = t(X2), M3 = t(X3)))
  }
  
  if (missVal) {
    ## Generation of artificial missing values (MCAR)
    MAR.generator <- function(X.dat, p) {
      for (k in 1:nrow(X.dat)) {
        MARs.row <- sample(x = c(TRUE, FALSE), size = ncol(X.dat), replace = TRUE, prob = c(p, 1 - p))
        X.dat[k, MARs.row] <- NA
      }
      return(X.dat)
    }
    
    X1.ic <- MAR.generator(X1, propMCAR)
    X2.ic <- MAR.generator(X2, propMCAR)
    X3.ic <- MAR.generator(X3, propMCAR)
    
    return(list(M1 = t(X1.ic), M2 = t(X2.ic), M3 = t(X3.ic)))
  }
}


