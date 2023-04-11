FROM condaforge/mambaforge:latest

WORKDIR /app

COPY conda-lock.yml .

RUN mamba install --channel=conda-forge --name=base conda-lock && \
    conda-lock install --name nisqa conda-lock.yml && \
    mamba clean -afy

COPY . .
