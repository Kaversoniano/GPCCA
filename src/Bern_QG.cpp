// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::vec logit(const arma::vec & p) {
  return arma::log(p / (1.0 - p));
}

// [[Rcpp::export]]
arma::mat sigmoid(const arma::mat & X) {
  return 1.0 / (1.0 + arma::exp(-X));
}

// [[Rcpp::export]]
arma::mat softplus(const arma::mat & X) {
  return arma::log1p(arma::exp(X));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
double Q_rcpp0(const arma::vec & param_vec,
               const arma::mat & Y, const Rcpp::List & Z_samples,
               double lambda, int n, int m, int d, int S) {
  
  arma::vec mu = param_vec.subvec(0, m - 1);
  arma::mat W = arma::reshape(param_vec.subvec(m, m + m * d - 1), m, d);
  
  double Q = 0.0;
  
  for (int k = 0; k < n; k++) {
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    Q += arma::accu(Y.col(k) % arma::sum(X_k, 1)) - arma::accu(softplus(X_k));
  }
  
  double obj = - Q / (S * n) + lambda * arma::accu(W % W);
  
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::vec G_rcpp0(const arma::vec & param_vec,
                  const arma::mat & Y, const Rcpp::List & Z_samples,
                  double lambda, int n, int m, int d, int S) {
  
  arma::vec mu = param_vec.subvec(0, m - 1);
  arma::mat W = arma::reshape(param_vec.subvec(m, m + m * d - 1), m, d);
  
  arma::vec G_mu = arma::zeros(m);
  arma::mat G_W = arma::zeros(m, d);
  
  for (int k = 0; k < n; k++) {
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    arma::mat resid = Y.col(k) * arma::ones<arma::rowvec>(S) - sigmoid(X_k);
    G_mu += arma::sum(resid, 1);
    G_W  += resid * Z_k.t();
  }
  
  arma::vec grad_mu = - G_mu / (S * n);
  arma::mat grad_W = - G_W / (S * n) + 2 * lambda * W;
  
  arma::vec grad_vec = arma::join_vert(grad_mu, arma::vectorise(grad_W));
  
  return grad_vec;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
double Q_rcpp1(const arma::vec & param_vec,
               const arma::mat & Y, const Rcpp::List & Z_samples,
               double lambda, int n, int m, int d, int S,
               Rcpp::List ind_M, const arma::vec & mu_t, const arma::mat & W_t) {
  
  arma::vec mu = param_vec.subvec(0, m - 1);
  arma::mat W = arma::reshape(param_vec.subvec(m, m + m * d - 1), m, d);
  
  double Q = 0.0;
  
  for (int k = 0; k < n; k++) {
    arma::uvec iM = ind_M[k];
    iM -= 1;
    
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    arma::mat X_k_t = mu_t.elem(iM) * arma::ones<arma::rowvec>(S) + W_t.rows(iM) * Z_k;
    
    arma::mat Y_k = Y.col(k) * arma::ones<arma::rowvec>(S);
    arma::mat est_k_t = softplus(X_k_t);
    Y_k.rows(iM) = est_k_t;
    
    arma::mat est_k = softplus(X_k);
    Q += arma::accu(Y_k % X_k - est_k);
  }
  
  double obj = - Q / (S * n) + lambda * arma::accu(W % W);
  
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::vec G_rcpp1(const arma::vec & param_vec,
                  const arma::mat & Y, const Rcpp::List & Z_samples,
                  double lambda, int n, int m, int d, int S,
                  Rcpp::List ind_M, const arma::vec & mu_t, const arma::mat & W_t) {
  
  arma::vec mu = param_vec.subvec(0, m - 1);
  arma::mat W = arma::reshape(param_vec.subvec(m, m + m * d - 1), m, d);
  
  arma::vec G_mu = arma::zeros(m);
  arma::mat G_W = arma::zeros(m, d);
  
  for (int k = 0; k < n; k++) {
    arma::uvec iM = ind_M[k];
    iM -= 1;
    
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    arma::mat X_k_t = mu_t.elem(iM) * arma::ones<arma::rowvec>(S) + W_t.rows(iM) * Z_k;
    
    arma::mat Y_k = Y.col(k) * arma::ones<arma::rowvec>(S);
    arma::mat est_k_t = sigmoid(X_k_t);
    Y_k.rows(iM) = est_k_t;
    
    arma::mat est_k = sigmoid(X_k);
    arma::mat resid = Y_k - est_k;
    G_mu += arma::sum(resid, 1);
    G_W  += resid * Z_k.t();
  }
  
  arma::vec grad_mu = - G_mu / (S * n);
  arma::mat grad_W = - G_W / (S * n) + 2 * lambda * W;
  
  arma::vec grad_vec = arma::join_vert(grad_mu, arma::vectorise(grad_W));
  
  return grad_vec;
}


