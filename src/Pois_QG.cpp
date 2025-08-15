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
double Q_W_rcpp0(const arma::vec & W_vec, const arma::vec & mu,
                 const arma::mat & Y, const Rcpp::List & Z_samples,
                 double lambda, int n, int m, int d, int S) {
  
  arma::mat W = arma::reshape(W_vec, m, d);
  double Q = 0.0;
  
  for (int k = 0; k < n; k++) {
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    Q += arma::accu(Y.col(k) % arma::sum(X_k, 1)) - arma::accu(arma::exp(X_k));
  }
  
  double obj = - Q / (S * n) + lambda * arma::accu(W % W);
  
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::vec G_W_rcpp0(const arma::vec & W_vec, const arma::vec & mu,
                    const arma::mat & Y, const Rcpp::List & Z_samples,
                    double lambda, int n, int m, int d, int S) {
  
  arma::mat W = arma::reshape(W_vec, m, d);
  arma::mat G = arma::zeros(m, d);
  
  for (int k = 0; k < n; k++) {
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    arma::mat resid = Y.col(k) * arma::ones<arma::rowvec>(S) - arma::exp(X_k);
    G += resid * Z_k.t();
  }
  
  arma::mat grad = - G / (S * n) + 2 * lambda * W;
  
  return arma::vectorise(grad);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
double Q_W_rcpp1(const arma::vec & W_vec, const arma::vec & mu,
                 const arma::mat & Y, const Rcpp::List & Z_samples,
                 double lambda, int n, int m, int d, int S,
                 Rcpp::List ind_M, const arma::mat & W_t) {
  
  arma::mat W = arma::reshape(W_vec, m, d);
  double Q = 0.0;
  
  for (int k = 0; k < n; k++) {
    arma::uvec iM = ind_M[k];
    iM -= 1;
    
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    arma::mat X_k_t = mu.elem(iM) * arma::ones<arma::rowvec>(S) + W_t.rows(iM) * Z_k;
    
    arma::mat Y_k = Y.col(k) * arma::ones<arma::rowvec>(S);
    arma::mat est_k_t = arma::exp(X_k_t);
    Y_k.rows(iM) = est_k_t;
    
    arma::mat est_k = arma::exp(X_k);
    Q += arma::accu(Y_k % X_k - est_k);
  }
  
  double obj = - Q / (S * n) + lambda * arma::accu(W % W);
  
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::vec G_W_rcpp1(const arma::vec & W_vec, const arma::vec & mu,
                    const arma::mat & Y, const Rcpp::List & Z_samples,
                    double lambda, int n, int m, int d, int S,
                    Rcpp::List ind_M, const arma::mat & W_t) {
  
  arma::mat W = arma::reshape(W_vec, m, d);
  arma::mat G = arma::zeros(m, d);
  
  for (int k = 0; k < n; k++) {
    arma::uvec iM = ind_M[k];
    iM -= 1;
    
    arma::mat Z_k = Z_samples[k];
    arma::mat X_k = mu * arma::ones<arma::rowvec>(S) + W * Z_k;
    arma::mat X_k_t = mu.elem(iM) * arma::ones<arma::rowvec>(S) + W_t.rows(iM) * Z_k;
    
    arma::mat Y_k = Y.col(k) * arma::ones<arma::rowvec>(S);
    arma::mat est_k_t = arma::exp(X_k_t);
    Y_k.rows(iM) = est_k_t;
    
    arma::mat est_k = arma::exp(X_k);
    arma::mat resid = Y_k - est_k;
    G += resid * Z_k.t();
  }
  
  arma::mat grad = - G / (S * n) + 2 * lambda * W;
  
  return arma::vectorise(grad);
}


