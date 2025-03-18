#' Example Multi-modal Dataset
#'
#' A synthetic multi-modal dataset with three modalities. The number of features 
#'   in each modality is \eqn{m_1 = 60}, \eqn{m_2 = 120} and \eqn{m_3 = 180}, 
#'   respectively. There are six groups of samples where each group has a equal 
#'   sample size of \eqn{100}. In total, there are \eqn{n = 600} samples, 
#'   with ground truth labels provided by \code{example_MultiModalData_labels}. 
#'   Artificial missing values are introduced with MCAR mechanism, leading to 
#'   an overall missing rate of 20\%.
#'
#' @format A list that consists of 3 numeric matrices:
#' \describe{
#'   \item{M1}{Modality 1: a matrix of size \eqn{m_1 \times n}, 
#'     i.e. 60 features (rows) by 600 samples (columns)}
#'   \item{M2}{Modality 2: a matrix of size \eqn{m_2 \times n}, 
#'     i.e. 120 features (rows) by 600 samples (columns)}
#'   \item{M3}{Modality 3: a matrix of size \eqn{m_3 \times n}, 
#'     i.e. 180 features (rows) by 600 samples (columns)}
#' }
#' @seealso \code{\link{example_MultiModalData_labels}}
"example_MultiModalData_list"

#' Ground Truth Labels of Example Multi-modal Dataset
#'
#' Ground truth sample labels of the synthetic multi-modal dataset given by 
#' \code{example_MultiModalData_list}.
#'
#' @format A character vector that consists of 100 \code{group 1}, 
#' 100 \code{group 2}, ... , 100 \code{group 6} in order.
#'
#' @seealso \code{\link{example_MultiModalData_list}}
"example_MultiModalData_labels"


