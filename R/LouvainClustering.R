
LouvainClustering <- function(model_fit) {
  LFs <- t(model_fit$Z)
  colnames(LFs) <- paste0("PC", 1:ncol(LFs))
  rownames(LFs) <- paste0("sample", 1:nrow(LFs))
  
  lc <- Seurat::FindNeighbors(object = LFs, compute.SNN = TRUE, verbose = FALSE)
  lc <- Seurat::FindClusters(object = lc$snn, verbose = FALSE)
  
  grps <- lc$res.0.8
  
  return(grps)
}


