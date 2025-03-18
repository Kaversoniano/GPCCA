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
arma::mat armaInv(const arma::mat & X) { return arma::inv(X); }

// [[Rcpp::export]]
arma::mat tcprod(const arma::vec & x) { return x * x.t(); }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
Rcpp::List fCore_missingF(const arma::mat & E_z, const arma::mat & E_x, const arma::mat & M, int n, int m, int d) {
  
  arma::mat sum_E_zz = arma::zeros(d, d);
  double E_tr = 0;
  arma::mat sum_E_xx = arma::zeros(m, m);
  
  for (int k = 0; k < n; k++) {
    arma::vec E_z_k = E_z.col(k);
    arma::mat E_zz_k = M + tcprod(E_z_k);
    sum_E_zz += E_zz_k;
    
    E_tr += arma::trace(E_zz_k);
    
    arma::vec E_x_k = E_x.col(k);
    sum_E_xx += tcprod(E_x_k);
  }
  
  // return Rcpp::List::create(
  //   _["sum_E_zz"] = sum_E_zz,
  //   _["E_tr"] = E_tr,
  //   _["sum_E_xx"] = sum_E_xx
  // );
  return Rcpp::List::create(
    Rcpp::Named("sum_E_zz") = sum_E_zz,
    Rcpp::Named("E_tr") = E_tr,
    Rcpp::Named("sum_E_xx") = sum_E_xx
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
Rcpp::List fCore_default0(const arma::mat & X, Rcpp::List ind_O, Rcpp::List ind_M,
                          const arma::mat & tW, const arma::mat & mu, const arma::mat & Psi,
                          int n, int m, int d) {
  
  arma::mat Z = arma::zeros(d, n);
  arma::mat sum_E_z = arma::zeros(d, 1);
  arma::mat sum_E_zz = arma::zeros(d, d);
  double E_tr = 0;
  arma::mat sum_E_x = arma::zeros(m, 1);
  arma::mat sum_E_xx = arma::zeros(m, m);
  arma::mat sum_E_xz = arma::zeros(m, d);
  
  arma::mat inv_Psi = armaInv(Psi);
  
  arma::mat M_o = arma::zeros(d, d);
  arma::mat E_z = arma::zeros(d, 1);
  arma::mat E_zz = arma::zeros(d, d);
  arma::mat E_x = arma::zeros(m, 1);
  arma::mat E_xx = arma::zeros(m, m);
  arma::mat E_xz = arma::zeros(m, d);
  
  for (int k = 0; k < n; k++) {
    arma::uvec iO = ind_O[k];
    arma::uvec iM = ind_M[k];
    iO -= 1;
    iM -= 1;
    
    arma::mat x_os = X.rows(iO);
    arma::mat x_o = x_os.col(k);
    arma::mat tW_o = tW.cols(iO);
    arma::mat mu_o = mu.rows(iO);
    arma::mat inv_Psi_o = inv_Psi(iO, iO) - inv_Psi(iO, iM) * armaInv(inv_Psi(iM, iM)) * inv_Psi(iM, iO);
    
    arma::mat P_o = tW_o * inv_Psi_o;
    
    M_o = armaInv(arma::eye(d, d) + P_o * tW_o.t());
    
    E_z = M_o * P_o * (x_o - mu_o);
    sum_E_z += E_z;
    
    Z.col(k) = E_z;
    
    E_zz = M_o + tcprod(E_z);
    sum_E_zz += E_zz;
    
    E_tr += arma::trace(E_zz);
    
    arma::mat tW_ms = tW.cols(iM);
    arma::mat W_ms = tW_ms.t();
    
    arma::mat est_M = W_ms * E_z + mu.rows(iM);
    
    E_x = X.col(k);
    E_x.rows(iM) = est_M;
    sum_E_x += E_x;
    
    E_xx = tcprod(X.col(k));
    E_xx(iM, iO) = est_M * x_o.t();
    E_xx(iO, iM) = x_o * est_M.t();
    E_xx(iM, iM) = W_ms * M_o * tW_ms + Psi(iM, iM) + tcprod(est_M);
    sum_E_xx += E_xx;
    
    E_xz = X.col(k) * E_z.t();
    E_xz.rows(iM) = W_ms * M_o + est_M * E_z.t();
    sum_E_xz += E_xz;
  }
  
  // return Rcpp::List::create(
  //   _["Z"] = Z,
  //   _["sum_E_z"] = sum_E_z,
  //   _["sum_E_zz"] = sum_E_zz,
  //   _["E_tr"] = E_tr,
  //   _["sum_E_x"] = sum_E_x,
  //   _["sum_E_xx"] = sum_E_xx,
  //   _["sum_E_xz"] = sum_E_xz
  // );
  return Rcpp::List::create(
    Rcpp::Named("Z") = Z,
    Rcpp::Named("sum_E_z") = sum_E_z,
    Rcpp::Named("sum_E_zz") = sum_E_zz,
    Rcpp::Named("E_tr") = E_tr,
    Rcpp::Named("sum_E_x") = sum_E_x,
    Rcpp::Named("sum_E_xx") = sum_E_xx,
    Rcpp::Named("sum_E_xz") = sum_E_xz
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
Rcpp::List fCore_default1(const arma::mat & X, Rcpp::List ind_O, Rcpp::List ind_M,
                          const arma::mat & tW, const arma::mat & mu, const arma::mat & Psi,
                          int n, int m, int d, int N_ms, Rcpp::List ms_index) {
  
  arma::mat Z = arma::zeros(d, n);
  arma::mat sum_E_z = arma::zeros(d, 1);
  arma::mat sum_E_zz = arma::zeros(d, d);
  double E_tr = 0;
  arma::mat sum_E_x = arma::zeros(m, 1);
  arma::mat sum_E_xx = arma::zeros(m, m);
  arma::mat sum_E_xz = arma::zeros(m, d);
  
  arma::mat inv_Psi = arma::zeros(m, m);
  for (int j = 0; j < N_ms; j++) {
    arma::uvec mj_index = ms_index[j];
    mj_index -= 1;
    inv_Psi(mj_index, mj_index) = armaInv(Psi(mj_index, mj_index));
  }
  
  arma::mat inv_Psi_o_full = arma::zeros(m, m);
  arma::mat M_o = arma::zeros(d, d);
  arma::mat E_z = arma::zeros(d, 1);
  arma::mat E_zz = arma::zeros(d, d);
  arma::mat E_x = arma::zeros(m, 1);
  arma::mat E_xx = arma::zeros(m, m);
  arma::mat E_xz = arma::zeros(m, d);
  
  for (int k = 0; k < n; k++) {
    arma::uvec iO = ind_O[k];
    arma::uvec iM = ind_M[k];
    iO -= 1;
    iM -= 1;
    
    arma::mat x_os = X.rows(iO);
    arma::mat x_o = x_os.col(k);
    arma::mat tW_o = tW.cols(iO);
    arma::mat mu_o = mu.rows(iO);
    
    inv_Psi_o_full.zeros();
    for (int j = 0; j < N_ms; j++) {
      arma::uvec mj_index = ms_index[j];
      mj_index -= 1;
      arma::uvec iO_j = arma::intersect(mj_index,iO);
      arma::uvec iM_j = arma::intersect(mj_index,iM);
      if (iO_j.n_elem == mj_index.n_elem) {
        inv_Psi_o_full(iO_j, iO_j) = inv_Psi(iO_j, iO_j);
      } else {
        inv_Psi_o_full(iO_j, iO_j) = inv_Psi(iO_j, iO_j) - inv_Psi(iO_j, iM_j) * armaInv(inv_Psi(iM_j, iM_j)) * inv_Psi(iM_j, iO_j);
      }
    }
    arma::mat inv_Psi_o = inv_Psi_o_full(iO, iO);

    
    arma::mat P_o = tW_o * inv_Psi_o;
    
    M_o = armaInv(arma::eye(d, d) + P_o * tW_o.t());
    
    E_z = M_o * P_o * (x_o - mu_o);
    sum_E_z += E_z;
    
    Z.col(k) = E_z;
    
    E_zz = M_o + tcprod(E_z);
    sum_E_zz += E_zz;
    
    E_tr += arma::trace(E_zz);
    
    arma::mat tW_ms = tW.cols(iM);
    arma::mat W_ms = tW_ms.t();
    
    arma::mat est_M = W_ms * E_z + mu.rows(iM);
    
    E_x = X.col(k);
    E_x.rows(iM) = est_M;
    sum_E_x += E_x;
    
    E_xx = tcprod(X.col(k));
    E_xx(iM, iO) = est_M * x_o.t();
    E_xx(iO, iM) = x_o * est_M.t();
    E_xx(iM, iM) = W_ms * M_o * tW_ms + Psi(iM, iM) + tcprod(est_M);
    sum_E_xx += E_xx;
    
    E_xz = X.col(k) * E_z.t();
    E_xz.rows(iM) = W_ms * M_o + est_M * E_z.t();
    sum_E_xz += E_xz;
  }
  
  // return Rcpp::List::create(
  //   _["Z"] = Z,
  //   _["sum_E_z"] = sum_E_z,
  //   _["sum_E_zz"] = sum_E_zz,
  //   _["E_tr"] = E_tr,
  //   _["sum_E_x"] = sum_E_x,
  //   _["sum_E_xx"] = sum_E_xx,
  //   _["sum_E_xz"] = sum_E_xz
  // );
  return Rcpp::List::create(
    Rcpp::Named("Z") = Z,
    Rcpp::Named("sum_E_z") = sum_E_z,
    Rcpp::Named("sum_E_zz") = sum_E_zz,
    Rcpp::Named("E_tr") = E_tr,
    Rcpp::Named("sum_E_x") = sum_E_x,
    Rcpp::Named("sum_E_xx") = sum_E_xx,
    Rcpp::Named("sum_E_xz") = sum_E_xz
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::mat diagInv(const arma::mat & X) { return arma::diagmat(1 / X.diag()); }

// [[Rcpp::export]]
Rcpp::List fCore_diagcov(const arma::mat & X, Rcpp::List ind_O, Rcpp::List ind_M,
                         const arma::mat & tW, const arma::mat & mu, const arma::mat & Psi,
                         int n, int m, int d) {
  
  arma::mat Z = arma::zeros(d, n);
  arma::mat sum_E_z = arma::zeros(d, 1);
  arma::mat sum_E_zz = arma::zeros(d, d);
  double E_tr = 0;
  arma::mat sum_E_x = arma::zeros(m, 1);
  arma::mat sum_E_xx = arma::zeros(m, m);
  arma::mat sum_E_xz = arma::zeros(m, d);
  
  arma::mat inv_Psi = diagInv(Psi);
  
  arma::mat M_o = arma::zeros(d, d);
  arma::mat E_z = arma::zeros(d, 1);
  arma::mat E_zz = arma::zeros(d, d);
  arma::mat E_x = arma::zeros(m, 1);
  arma::mat E_xx = arma::zeros(m, m);
  arma::mat E_xz = arma::zeros(m, d);
  
  for (int k = 0; k < n; k++) {
    arma::uvec iO = ind_O[k];
    arma::uvec iM = ind_M[k];
    iO -= 1;
    iM -= 1;
    
    arma::mat x_os = X.rows(iO);
    arma::mat x_o = x_os.col(k);
    arma::mat tW_o = tW.cols(iO);
    arma::mat mu_o = mu.rows(iO);
    arma::mat inv_Psi_o = inv_Psi(iO, iO);
    
    arma::mat P_o = tW_o * inv_Psi_o;
    
    M_o = armaInv(arma::eye(d, d) + P_o * tW_o.t());
    
    E_z = M_o * P_o * (x_o - mu_o);
    sum_E_z += E_z;
    
    Z.col(k) = E_z;
    
    E_zz = M_o + tcprod(E_z);
    sum_E_zz += E_zz;
    
    E_tr += arma::trace(E_zz);
    
    arma::mat tW_ms = tW.cols(iM);
    arma::mat W_ms = tW_ms.t();
    
    arma::mat est_M = W_ms * E_z + mu.rows(iM);
    
    E_x = X.col(k);
    E_x.rows(iM) = est_M;
    sum_E_x += E_x;
    
    E_xx = tcprod(X.col(k));
    E_xx(iM, iO) = est_M * x_o.t();
    E_xx(iO, iM) = x_o * est_M.t();
    E_xx(iM, iM) = W_ms * M_o * tW_ms + Psi(iM, iM) + tcprod(est_M);
    sum_E_xx += E_xx;
    
    E_xz = X.col(k) * E_z.t();
    E_xz.rows(iM) = W_ms * M_o + est_M * E_z.t();
    sum_E_xz += E_xz;
  }
  
  // return Rcpp::List::create(
  //   _["Z"] = Z,
  //   _["sum_E_z"] = sum_E_z,
  //   _["sum_E_zz"] = sum_E_zz,
  //   _["E_tr"] = E_tr,
  //   _["sum_E_x"] = sum_E_x,
  //   _["sum_E_xx"] = sum_E_xx,
  //   _["sum_E_xz"] = sum_E_xz
  // );
  return Rcpp::List::create(
    Rcpp::Named("Z") = Z,
    Rcpp::Named("sum_E_z") = sum_E_z,
    Rcpp::Named("sum_E_zz") = sum_E_zz,
    Rcpp::Named("E_tr") = E_tr,
    Rcpp::Named("sum_E_x") = sum_E_x,
    Rcpp::Named("sum_E_xx") = sum_E_xx,
    Rcpp::Named("sum_E_xz") = sum_E_xz
  );
}


