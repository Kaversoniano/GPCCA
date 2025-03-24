#' Example Multi-modal Dataset (complete)
#'
#' A synthetic multi-modal dataset with three modalities. The number of features 
#'   in each modality is \eqn{m_1 = 30}, \eqn{m_2 = 60} and \eqn{m_3 = 90}, 
#'   respectively. There are six groups of samples where each group has a equal 
#'   sample size of \eqn{50}. In total, there are \eqn{n = 300} samples, 
#'   with ground truth labels provided by \code{example_MultiModalData_labels}. 
#'   The dataset is complete with no missing value.
#'
#' @format A list that consists of 3 numeric matrices:
#' \describe{
#'   \item{M1}{Modality 1: a matrix of size \eqn{m_1 \times n}, 
#'     i.e. 30 features (rows) by 300 samples (columns)}
#'   \item{M2}{Modality 2: a matrix of size \eqn{m_2 \times n}, 
#'     i.e. 60 features (rows) by 300 samples (columns)}
#'   \item{M3}{Modality 3: a matrix of size \eqn{m_3 \times n}, 
#'     i.e. 90 features (rows) by 300 samples (columns)}
#' }
#' @seealso \code{\link{example_MultiModalData_list1}}, 
#'          \code{\link{example_MultiModalData_labels}}
"example_MultiModalData_list0"

#' Example Multi-modal Dataset (incomplete)
#'
#' A synthetic multi-modal dataset with three modalities. The number of features 
#'   in each modality is \eqn{m_1 = 30}, \eqn{m_2 = 60} and \eqn{m_3 = 90}, 
#'   respectively. There are six groups of samples where each group has a equal 
#'   sample size of \eqn{50}. In total, there are \eqn{n = 300} samples, 
#'   with ground truth labels provided by \code{example_MultiModalData_labels}. 
#'   Artificial missing values are introduced with MCAR mechanism, leading to 
#'   an overall missing rate of 20\%.
#'
#' @format A list that consists of 3 numeric matrices:
#' \describe{
#'   \item{M1}{Modality 1: a matrix of size \eqn{m_1 \times n}, 
#'     i.e. 30 features (rows) by 300 samples (columns)}
#'   \item{M2}{Modality 2: a matrix of size \eqn{m_2 \times n}, 
#'     i.e. 60 features (rows) by 300 samples (columns)}
#'   \item{M3}{Modality 3: a matrix of size \eqn{m_3 \times n}, 
#'     i.e. 90 features (rows) by 300 samples (columns)}
#' }
#' @seealso \code{\link{example_MultiModalData_list0}}, 
#'          \code{\link{example_MultiModalData_labels}}
"example_MultiModalData_list1"

#' Ground Truth Labels of Example Multi-modal Dataset
#'
#' Ground truth sample labels of the synthetic multi-modal dataset given by 
#' \code{example_MultiModalData_list0} (complete) or 
#' \code{example_MultiModalData_list1} (incomplete).
#'
#' @format A character vector that consists of 50 \code{"group 1"}, 
#' 50 \code{"group 2"}, ... , 50 \code{"group 6"} in order.
#'
#' @seealso \code{\link{example_MultiModalData_list0}}, 
#'          \code{\link{example_MultiModalData_list1}}
"example_MultiModalData_labels"


