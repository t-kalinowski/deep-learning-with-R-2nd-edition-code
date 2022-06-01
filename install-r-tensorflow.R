#!/usr/bin/env Rscript

if(!requireNamespace("remotes")) install.packages("remotes")

remotes::update_packages()
remotes::install_cran(c("readr", "tibble", "zip", "fs", "listarrays"))

if("--fresh" %in% commandArgs(TRUE)) {
  reticulate::miniconda_uninstall()
  unlink("~/.pyenv", recursive = TRUE)
  unlink("~/.virtualenvs/r-reticulate", recursive = TRUE)
}


python <- reticulate::install_python("3.9:latest")
reticulate::virtualenv_create("r-reticulate", python = python)

keras::install_keras(
  envname = "r-reticulate",
  extra_packages = c("keras-tuner", "ipython", "kaggle"))
if(Sys.info()["sysname"] == "Linux")
  reticulate::py_install(
    "numpy",
    envname = "r-reticulate",
    pip = TRUE,
    pip_options = c("--force-reinstall", "--no-binary numpy")
  )

