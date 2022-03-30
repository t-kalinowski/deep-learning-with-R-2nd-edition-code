#!/usr/bin/env Rscript

if(!requireNamespace("remotes")) install.packages("remotes")

remotes::update_packages()
remotes::install_cran(c("readr", "tibble", "zip", "fs", "listarrays"))
remotes::install_github("rstudio/reticulate", force = TRUE)
remotes::install_github("rstudio/tensorflow", force = TRUE)
remotes::install_github("rstudio/keras", force = TRUE)
remotes::install_github("rstudio/tfdatasets", force = TRUE)
remotes::install_github("t-kalinowski/tfautograph", force = TRUE)

if("--fresh" %in% commandArgs(TRUE)) {
  reticulate::miniconda_uninstall()
  unlink("~/.pyenv", recursive = TRUE)
  unlink("~/.virtualenvs/r-reticulate", recursive = TRUE)
}


if (tensorflow:::is_mac_arm64()) {
  reticulate::install_miniconda()
  keras::install_keras(extra_packages = c("ipython"))
  reticulate::py_install(c("keras-tuner", "kaggle"),
                         envname = "r-reticulate",
                         pip = TRUE)

} else {
  python <- reticulate::install_python("3.9:latest")
  reticulate::virtualenv_create("r-reticulate", python = python)

  keras::install_keras(extra_packages = c("keras-tuner", "ipython", "kaggle"))
  reticulate::py_install(
    "numpy",
    envname = "r-reticulate",
    pip = TRUE,
    pip_options = c("--force-reinstall", "--no-binary numpy")
  )
}
