
ConsensusScore <- function(elem) {
  if (elem == 0) return(0)
  if (elem > 0) return(elem * log2(elem))
}


