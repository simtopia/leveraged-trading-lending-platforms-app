# Leveraged trading via Lending platforms
App to analyse loan positions offered by decentralised lending platforms through the lens of perpetual futures.
This app provides the simulations for the paper [Leveraged trading via lending platforms]().

### Deploying the app
Requirements can be installed using [poetry ](https://python-poetry.org) by running from the repository home directory
```
poetry install
```
activate the python environment by running
```
source .venv/bin/activate
```

In order to deploy the app, run
```
streamlit run app_lending_vs_perp_with_correlation.py
```
to view the app in your browser in  http://localhost:8501.
