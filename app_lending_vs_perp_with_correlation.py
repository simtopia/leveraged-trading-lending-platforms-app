import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

pio.templates.default = "plotly"

from typing import Callable, List, Tuple

# if not os.path.exists("results"):
#     os.makedirs("results")


@st.cache_data  # -- Magic command to cache data
def get_gbm(
    mu: float, sigma: float, dt: float, n_steps: int, p0: float, seed: int, n_mc
) -> np.array:
    """Get gbm paths"""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_mc, n_steps))

    paths = np.ones((n_mc, n_steps + 1)) * p0
    paths[:, 1:] = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    paths = paths.cumprod(axis=1)
    return paths


def sigmoid(x: np.array):
    return 1 / (1 + np.exp(-x))


def reciprocal_sigmoid(x: np.array):
    return np.log(x / (1 - x))


def get_utilisation(
    price_paths: np.array,
    u0: float,
    a: float,
    b: float = 0,
) -> np.array:
    latent_u = np.zeros_like(price_paths)
    norm_price_paths = (price_paths - price_paths.mean()) / price_paths.std()
    latent_u[:, 0] = reciprocal_sigmoid(u0)

    for i in range(1, latent_u.shape[1]):
        delta_P = norm_price_paths[:, i] - norm_price_paths[:, i - 1]
        latent_u[:, i] = latent_u[:, i - 1] + (a + b * latent_u[:, i - 1]) * delta_P

    u = sigmoid(latent_u)
    return u


def irm(
    u_optimal: float,
    r_0: float,
    r_1: float,
    r_2: float,
    utilisation: float,
    collateral: bool,
) -> float:
    if utilisation < u_optimal:
        r = r_0 + r_1 * utilisation / u_optimal
    else:
        r = r_0 + r_1 + r_2 * (utilisation - u_optimal) / (1 - u_optimal)
    if collateral:
        return utilisation * r
    else:
        return r


vect_irm = np.vectorize(irm, excluded=("u_optimal", "r_0", "r_1", "r_2", "collateral"))


def get_liquidation_call_mask(
    price_paths: np.array,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
) -> np.array:
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))
    mask = (
        price_paths * np.exp((r_collateral_eth - r_debt_dai) * time_array)
        <= 1 / lt * ltv0 * p0
    )
    return mask


def get_liquidation_times(
    price_paths: np.array,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
):
    mask = get_liquidation_call_mask(
        price_paths, dt, lt, ltv0, r_collateral_eth, r_debt_dai
    )
    mask = mask.astype(int)
    idx_liquidation_time = np.argmax(mask, axis=1, keepdims=True)
    liquidation_time = idx_liquidation_time * dt
    T = dt * price_paths.shape[1]
    liquidation_time[liquidation_time == 0] = (
        T + 0.1
    )  # big number out of considered time range
    return liquidation_time


def get_pnl_lending_position(
    price_paths,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
):
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))
    liquidation_times = get_liquidation_times(
        price_paths, dt, lt, ltv0, r_collateral_eth, r_debt_dai
    )

    payoff = (
        price_paths * np.exp(r_collateral_eth * time_array)
        - ltv0 * p0 * np.exp(r_debt_dai * time_array)
    ) * (time_array <= liquidation_times)
    for i, row in enumerate(payoff):
        payoff[i, time_array[i] > liquidation_times[i]] = payoff[
            i, time_array[i] == liquidation_times[i]
        ]

    pnl = payoff - p0 * (1 - ltv0)
    return pnl


def decompose_pnl_lending_position(
    price_paths,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
) -> Tuple[np.array, np.array]:
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))
    return price_paths - p0, price_paths * (
        1 - np.exp(r_collateral_eth * time_array)
    ) - ltv0 * p0 * (1 - np.exp(r_debt_dai * time_array))


@st.cache_data  # -- Magic command to cache data
def get_perps_price_mean_rev(
    price_paths: np.array,
    lambda_: float,
    sigma: float,
    dt: float,
    r: float,
    kappa: float = 1,
    seed: int = 1,
) -> np.array:
    n_mc = price_paths.shape[0]
    p0 = price_paths[0, 0]
    f0 = p0 * (1 + r / kappa)
    f = np.ones_like(price_paths) * f0
    rng = np.random.default_rng(seed)
    coef = 1 + r / kappa
    for step in range(1, f.shape[1]):
        z = rng.normal(size=n_mc)
        f[:, step] = (
            f[:, step - 1]
            + lambda_ * (price_paths[:, step - 1] - f[:, step - 1]) * dt
            + np.sqrt(dt) * sigma * z
        )
    return f


@st.cache_data  # -- Magic command to cache data
def get_perps_price_mean_rev_to_non_arb(
    price_paths: np.array,
    lambda_: float,
    sigma: float,
    dt: float,
    r: float,
    kappa: float = 1,
    seed: int = 1,
) -> np.array:
    n_mc = price_paths.shape[0]
    p0 = price_paths[0, 0]
    f0 = p0 * (1 + r / kappa)
    f = np.ones_like(price_paths) * f0
    rng = np.random.default_rng(seed)
    coef = 1 + r / kappa
    for step in range(1, f.shape[1]):
        z = rng.normal(size=n_mc)
        f[:, step] = (
            f[:, step - 1]
            + lambda_ * (coef * price_paths[:, step - 1] - f[:, step - 1]) * dt
            + np.sqrt(dt) * sigma * z
        )
    return f


def get_perps_price_non_arb(price_paths: np.array, r: float, kappa: float = 1):
    f = price_paths * (1 + r / kappa)
    return f


def get_funding_fee_perps(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    dt: float,
    kappa: float = 1,
):
    p0 = price_paths[0, 0]
    f0 = perps_price_paths[0, 0]

    n_mc, n_steps = price_paths.shape
    funding_fee = np.zeros(shape=(n_mc, n_steps))
    time_array = np.arange(n_steps) * dt
    for i, step in enumerate(range(1, n_steps)):
        t = time_array[step]
        funding_fee[:, step] = kappa * np.sum(
            dt
            * np.exp((t - time_array[:step]) * r_debt_dai[:, :step])
            * (perps_price_paths[:, :step] - price_paths[:, :step]),
            axis=1,
        )

    return funding_fee


def get_pnl_perps(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    dt: float,
    r: float,
    kappa: float = 1,
):
    n_mc, n_steps = perps_price_paths.shape
    funding_rate = get_funding_fee_perps(
        price_paths, perps_price_paths, r_debt_dai, dt, kappa
    )
    pnl = np.zeros_like(price_paths)
    pnl = perps_price_paths - perps_price_paths[:, 0].reshape(n_mc, 1) - funding_rate
    coef = 1 + r / kappa
    return 1 / coef * pnl


def get_liquidation_times_perp(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    lt_f: float,
    ltv0: float,
    dt: float,
    r: float,
    kappa: float = 1,
):
    pnl = get_pnl_perps(price_paths, perps_price_paths, r_debt_dai, dt, r, kappa)
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))

    maintenance_margin = price_paths * lt_f  # price_paths * (1 - max_ltv0) * lt_f
    initial_margin = p0 * (1 - ltv0)

    mask = maintenance_margin >= initial_margin + pnl

    T = dt * price_paths.shape[1]
    idx_liquidation_time = np.argmax(mask.astype(int), axis=1, keepdims=True)
    liquidation_time = idx_liquidation_time * dt
    liquidation_time[liquidation_time == 0] = T + 0.1
    return liquidation_time


def get_pnl_perps_after_liquidation(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    lt_f: float,
    ltv0: float,
    dt: float,
    r: float,
    kappa: float = 1,
):
    pnl = get_pnl_perps(price_paths, perps_price_paths, r_debt_dai, dt, r, kappa)
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))

    liquidation_times = get_liquidation_times_perp(
        dt=dt,
        kappa=kappa,
        lt_f=lt_f,
        ltv0=ltv0,
        perps_price_paths=perps_price_paths,
        price_paths=price_paths,
        r=r,
        r_debt_dai=r_debt_dai,
    )
    pnl = pnl * (time_array <= liquidation_times)
    for i, _ in enumerate(pnl):
        pnl[i, time_array[i] >= liquidation_times[i]] = pnl[
            i, time_array[i] == liquidation_times[i]
        ]
    return pnl


# ------------
# GLOBAL vars
# ------------
def get_idx_from_t(t):
    return int(t * 100)


dt = 0.01
n_steps = 100
p0 = 2000
seed = 1
n_mc = 10000

r = 0.05
kappa = 1
# --------------


# -- Set page config
apptitle = "Leveraged trading"
st.set_page_config(
    layout="wide",
    page_title=apptitle,
)


header_col1, header_col2 = st.columns([0.2, 0.8], gap="medium")
with header_col1:
    st.image("logo.jpeg", width=100)
with header_col2:
    st.title("Simtopia")

# --------
# INTRO
# --------
st.header("Introduction")
st.markdown(
    """
    In this dashboard we present the simulations underpinning the results in the paper [Leveraged trading via lending platforms]().

    Decentralised lending protocols enable users to enter a leveraged long
    or short trading position, which provides one of the economic rationales for overcollateralised
    loans. The smaller the haircut on the provided collateral (initial Loan-to-Value),
    the higher the leverage. This work compares loan positions on leading
    platforms with perpetual futures, a primary mechanism for trading leverage in a
    decentralised finance (DeFi) ecosystem. We introduce the notion of the implied
    funding fee/funding rate and contrast it with the funding fee/funding rate for perpetual
    futures and find that the former is significantly less volatile than the latter.
    Furthermore, we study PnL for both positions, the likelihood of liquidation for loan
    positions, and the margin calls for perpetual futures across multiple market conditions.
    """
)

# st.header("Notation and key objects")
# expander = st.expander("Loan Position")
# with expander:
#     st.markdown(
#         """
#         $(P_t)_{t\geq 0}$ denoted the price process of a risky asset (e.g. ETH), with a dollar stablecoin being the numeraire (e.g DAI).

#         Let $(r^{b,E}, r^{c,E})$, $(r^{b,D}, r^{c,D})$ be interest rates for borrowing and providing collateral for ETH and DAI respectively.
#         Furthermore, Let $\\theta^0\in[0,1)$ be an initial Loan-to-Value, meaning one can borrow up to $\\theta^0$  worth of ETH for every unit of ETH deposited as collateral.
#         To open an a long-ETH loan position an agent effectively requires $P_0(1 - \\theta^0)$ of DAI to establish the position. For capital efficiency the agent may use a (or a flashloan and a swap) along the following steps:

#         - Begin with $P_0(1-\theta^0)$ of DAI
#         - Obtain 1 ETH using flashswap (need to deposit $P_0$ DAI within one block for this to materialise)
#         - Deposit 1 ETH as collateral and start earning interests according to $e^{r^{c,E}\,t}$
#         - Borrow $\\theta^0 \,P_0$ of DAI against the collateral and start paying interests according to $\\theta^0 \,P_0 e^{r^{b,D}\, t}$
#         - Put together $\\theta^0 \,P_0$ and initial amount $P_0(1-\\theta^0)$  of DAI to complete the flashswap.

#         At any time the holder of the position may choose to pay back the loan $\\theta^0 \,P_0 e^{r^{b,D}\, t}$ in exchange for the collateral with
#         value $P_t\,e^{r^{c,E}\,t} $. Note that a rational agent will only do that if $\\theta^0 \,P_0 e^{r^{b,D}\, t} \leq P_t\,e^{r^{c,E}\,t}$, otherwise
#         it is better to walk away from the position.  Hence, up until the liquidation the agent in entitled to the  payoff
#         $$
#         (P_t\,e^{r^{c,E}\,t} -\\theta^0 \,P_0 e^{r^{b,D}\, t} )_{+} = e^{r^{b,D}\, t} (P_t\,e^{(r^{c,E}-r^{b,D})\,t} -\\theta^0 \,P_0  )_{+},
#         $$
#         where $x_+ = \max\{0,x\}$.

#         Let $\\theta \in (\\theta^0,1)$ be a liquidation threshold. If the value of the asset falls too low, the position will be liquidated. This occurs at time $\\tau^B$ defined as
#         $$
#         \\tau^B := \inf \left\{ t \geq 0: \\theta P_t e^{r^{c,E} \, t} \leq \\theta^0 \,P_0\ e^{r^{b,D} \,t}  \\right\} =  \inf \left\{ t \geq 0:  P_t e^{(r^{c,E} - r^{b,D})\, t} \leq \\theta^{-1 }\\theta^0 \,P_0  \\right\}
#         $$

#         The payoff accounting for liquidations is given by
#         $$
#         \psi(P_t) :=       (P_t\,e^{r^{c,E}\,t} - \\theta^0 \,P_0\,e^{r^{b,D}\, t} )\mathbb{1}_{\{t<\\tau^B\}}\,.
#         $$

#         The PnL is given by
#         $$
#         \\text{PnL}_t=\psi(P_t) - \psi(P_0)  = (P_t\,e^{(r^{c,E})\,t} - \\theta^0 \,P_0\,e^{r^{b,D}\, t} )\mathbb{1}_{\{t<\\tau^B\}} - P_0(1-\\theta^0)
#         $$
#         The PnL is driven by the change in the price of ETH/DAI and interest rates.

#         To see how lending position relates to trading spot with a leverage note that up to a liquidation event i.e for $t<\\tau^B$,
#         for and ignoring interest rates (one can think of small $t$), i.e  $r^{c,E}=r^{b,D}$ the PnL is given $P_t - P_0$ but the agent
#         only needs $P_0(1-\\theta^0)$ to enter the contract (and not $P_0$ which would be required to enter a spot trade).
#         Hence we see that the initial LTV translates into leverage. The closer $\\theta^0$ is to $1$ the higher the leverage and consequently the risk of a liquidation.
#         """
#     )

# expander = st.expander("Implied Funding fee of Loan Position")
# with expander:
#     st.markdown(
#         """
#         We see there is a fine balance between the change in the price of  ETH/DAI and the interest rate.
#         The changing interest rates on a lending protocol have a similar effect to the funding rate for perpetual
#         futures in the sense that if demand for a long position on a lending platform translates to significant change
#         in interest rates in comparison to the change of ETH then longs are losing money. We can also write for $t<\tau^B$
#         $$
#         \\text{PnL}_t = \psi(P_t) - \psi(P_0)  = P_t - P_0 - \left( P_t - P_0   - \psi(P_t ) + \psi(P_0) \\right)
#         $$
#         $$
#         \,\,\,\,\, =  P_t - P_0 - \left( P_t(1 -e^{r^{C,E} t })   - \\theta ^0 P_0 (1 - e^{r^{b,D} t}) \\right)
#         $$
#         $$
#         \,\,\,\,\, = P_te^{r^{C,E} t } - P_0  - \\theta^0P_0(e^{r^{b,D} t} - 1)\,.
#         $$
#         This shows one can decompose the PnL of the loan position into $ P_t - P_0 $ which corresponds to gains from leveraged trading and
#           $\left( P_t(1 -e^{r^{C,E} t })   - \\theta ^0 P_0 (1 - e^{r^{b,D} t}) \\right)$  which can be thought of as a counterpart of funding fee for perpetual futures.
#         """
#     )

# expander = st.expander("Perpetual futures")
# with expander:
#     st.markdown(
#         """
#         An agent seeking to enter a leveraged long or short trading position instead of using loan protocol may use a venue offering perpetual future contracts.

#         *Definition - Perpetuals Futures*

#         *A perpetual contract with price F and a funding rate design to track spot price $P$ is an agreement between two parties referred to as the long side and short side.
#         There is 0 cost to enter the agreement. The long side has the right to terminate the contract at any time $t \geq 0$, at which point they will receive a payment of $F_t - F_0$.
#         In return, for a fixed scaling parameter $\kappa>0 $, at every $t_i$, $i=1,2,\ldots$  the long-side must pay to the short side
#         the $\\frac{\kappa}{t_{i+1} - t_i}\int_{t_i}^{t_{i+1}}\left( (F_s-P_s\\right)ds$ referred to as the funding rate up until the contract is terminated.*


#         As frequency of funding rate payment increases funding rate converges to $\kappa \int_0^t \left( F_s -P_s \\right)ds$ and its value at time $t$ is
#         $\kappa \int_0^t e^{r(t-s)}\left( F_s -P_s \\right)ds$.
#         The PnL generated by the long perpetual futures  is given by
#         $$
#         \\text{PnL}_t^{F} :=  F_t - F_0  -  \kappa \int_0^t \,e^{r(t-s)} \left( F_s -P_s \\right)ds\,.
#         $$

#         The fair price of the perpetual futures can be deduced by requiring that there is no cash-and-carry arbitrage opportunities in the market. An agent who begins with zero capital, shorts spot and at the same going long futures contract (or being long spot and short futures contract) shouldn't at any time have positive cash flows with probability one.

#         Through the computation we discount all the cash flows with $r^{c,D}$. The economic rationale for this is that $r^{c,D}$ is a rate at which blockchain users are willing lend liquid stable coins and hence we can take it as a proxy for risk free rate on these markets.

#         *Lemma - Non arbitrage price of perp option*

#         *Assume that spot and futures prices are given by stochastic process $(P_t)_{t\geq 0}$ and $(S_t)_{t\geq 0}$ adapted to a filtration $(\mathcal F_t)_{t\geq 0}$.
#         Assume now there there is a measure $\mathbb P$ such that*
#         $$
#             \mathbb E^{\mathbb P} \left[  \int_0^\infty e^{-r^{c,D}s} \left( |P_s| + |F_s| \\right) ds  \\right] <\infty\,.
#         $$
#         *Then either there is cash-and-carry arbitrage or spot and perpetual future price satisfy for any $t>0$ and $\kappa>0$*

#         $$
#             F_t = P_t\,(1+\\frac{r^{c,D}}{\kappa})
#         $$

#         """
#     )

# expander = st.expander("Margin trading and PnL of long Perps considering liquidation")
# with expander:
#     st.markdown(
#         """
#         To compare perpetual futures to the corresponding loan position, assume that there is a venue that allows agents to take long
#         perpetual future position using the same leverage as a lending platform that is $\\frac{1}{1-\\theta^0}$.

#         We first assume that the non-arbitrage condition established in the Lemma holds, that is  $F_t = P_t(1+\\frac{r}{\kappa})$.

#         To enter a long perpetual futures position, an agent

#         1. Deposits $P_0(1-\\theta^0)$ on an initial margin account. This corresponds to the initial margin.
#         2. Gains exposure to $P_0 = F_0(1+\\frac{r}{\kappa})^{-1}$ perpetual futures with a cash-flow
#         $$
#         \\text{PnL}^F_t := \left( 1 + \\frac{r}{\kappa}\\right)^{-1}\left( F_t - F_0  -  \kappa \int_0^t \,e^{r(t-s)} \left( F_s -P_s \\right)ds \\right)\,.
#         $$

#         We assume that
#         $$
#         \\text{maintenance margin}_t = P_t(1-\\theta^0) \cdot \\theta^F, \,\,\,\,0 \leq \\theta^F \leq 1.
#         $$

#         The position gets liquidated when $\\text{initial margin} + \\text{PnL}_t^F \leq \\text{maintenance margin}_t$. The first time this happens is given by

#         $$
#         \\tau^F := \left\{t \geq 0 : P_t(1-\\theta^0) \cdot \\theta^F \leq P_0(1-\\theta^0) +  \\text{PnL}_t^F\\right\},
#         $$

#         thus the PnL including liquidation is $\\text{PnL}_{t\wedge \\tau^F}^F$.
#         """
#     )

st.subheader("Numerical experiment")
expander = st.expander("model")
with expander:
    st.markdown(
        """
        Assume that the price ETH/DAI follows geometric Brownian motion with drift $\mu\in\mathbb R$ and volatility $\sigma\in(0,\infty)$ so that
        $$
            P_t=P_0 \exp\left( (\mu - \\frac{1}{2}\sigma^2)t + \sigma W_t  \\right)\,.
        $$
        From the data one can see that not arbitrager condition $F_t = P_t(1+\\frac{r}{\kappa})$ is often violated. This is because we have not
        considered market frictions such as slippage, gas fees etc. To better reflect market data model the price process $(F_t)_{t\geq 0}$ by stochastic mean-reverting process.

        For mean-reversion parameter $\lambda>0$, and volatility $\sigma^F$ one can model price of future contract as
        $$
            dF_t = \lambda (P_t - F_t)dt + \sigma^F F_t dW_t^F\,,
        $$

        where $[W^F,W](t) = \\rho t$. One can then compare PnLs and liquidation times of perpetual futures and loan positions in different market regimes.
        """
    )

# ----------
# Price processes
# -----------

st.subheader("Price process of underlying and price process of perps option")
price_col1, price_col2, _ = st.columns([0.2, 0.2, 0.6], gap="medium")
with price_col1:
    st.write("Price process")
    mu = st.slider("$\mu$", -1.2, 0.3, 0.0, step=0.3)  # min, max, default
    sigma = st.slider("$\sigma$", 0.1, 0.5, 0.3, 0.2)  # min, max, default

with price_col2:
    st.write("Perps Price process")
    select_perp = st.selectbox(
        "How do we define the Perp price process?",
        [
            "Non-arbitrage",
            "Mean-reversion to P",
            "Mean-reversion to non-arbitrage price",
        ],
        index=1,
    )
    lambda_ = st.slider(
        "$\lambda$ mean-reversion parameter", 1, 200, 50
    )  # min, max, default
    sigma_f = st.slider("$\sigma_F$", 0.01, 500.0, 100.0)  # min, max, default


price_paths = get_gbm(mu, sigma, dt, n_steps, p0, seed, n_mc)
time = np.arange(0, n_steps + 1) * dt
df_price_paths = pd.DataFrame(
    price_paths.T, columns=["mc{}".format(i) for i in range(price_paths.shape[0])]
)
df_price_paths = df_price_paths.assign(time=time, underlying="spot")
if select_perp == "Non-arbitrage":
    perps_price_paths = get_perps_price_non_arb(price_paths, r=0.01, kappa=kappa)
elif select_perp == "Mean-reversion to P":
    perps_price_paths = get_perps_price_mean_rev(
        price_paths,
        lambda_=lambda_,
        sigma=sigma_f,
        dt=dt,
        r=0.01,
        kappa=kappa,
    )
else:
    perps_price_paths = get_perps_price_mean_rev_to_non_arb(
        price_paths,
        lambda_=lambda_,
        sigma=sigma_f,
        dt=dt,
        r=0.01,
        kappa=kappa,
    )
df_perps_price_paths = pd.DataFrame(
    perps_price_paths.T,
    columns=["mc{}".format(i) for i in range(perps_price_paths.shape[0])],
)
df_perps_price_paths = df_perps_price_paths.assign(time=time, underlying="perp")

df_price = pd.concat([df_price_paths, df_perps_price_paths])
df_price_melted = pd.melt(
    df_price,
    id_vars=["underlying", "time"],
    value_vars=["mc{}".format(i) for i in range(perps_price_paths.shape[0])],
    value_name="price",
)
_, price_plot_col, _ = st.columns([0.1, 0.8, 0.1])
with price_plot_col:
    samples = ["mc{}".format(i) for i in range(20)]
    fig_price = px.line(
        df_price_melted[df_price_melted["variable"].isin(samples)],
        x="time",
        y="price",
        line_group="variable",
        color="underlying",
    )
    # fig_price.write_image(f"results/price_mu{mu}_sigma{sigma}.png")
    st.plotly_chart(fig_price, use_container_width=True)


# ----------------------------------
# Utilisation of collateral Pool
# ----------------------------------
st.subheader("Utilisation of collateral pool")
with st.expander("Impact of lending on the spot"):
    st.markdown(
        """
        If market participants are bullish on the price of ETH/DAI and wish to enter a leveraged position via a lending protocol,
        they are depositing ETH as collateral and borrowing DAI.

        On the other hand, opening a long ETH loan position, ceteris paribus decreases utilisation and hence interest rate for lending ETH,
        and increases utilisation and therefore interest rate for borrowing DAI
        (how these rates change depends on the elasticity of market supply and demand curves and interest rate mechanism).
        This implies that the PnL of a loan position decreases, and the liquidation risk increases.

        We model model the utilisation of the DAI and ETH pool in terms of the ETH price as

        $$
        \mathcal U_t^{ABC} = g(\\nu_t^{ABC}), \quad \\text{where } \quad d\\nu^{ABC}_t = \\alpha^{ABC} dP_t, g(\\nu^{ABC}_0) = u_0^{ABC}
        $$
        where $g(x) = (1+e^{-x})^{-1}$ is the sigmoid function and $ABC$ denotes ETH or DAI.

        Note that the quadratic covariation of $\mathcal U_t, P_t$ satisfies

        $$
        d[P_t, \mathcal U_t^{ABC}] = g'(\\nu_t) \\alpha^{ABC} d[P_t, P_t]
        $$
        hence the parameter $\\alpha^{ABC}$ provides the strength of the association between $P_t, \mathcal U^{ABC}_t$.
        """
    )

col1, col2, col3 = st.columns([0.2, 0.2, 0.8], gap="medium")
with col1:
    st.write("Impact of ETH price on ETH utilisation")
    alpha_eth = st.slider("$\\alpha^{ETH}$", -1.0, 1.0, -0.15, step=0.01)
    u0_eth = st.slider("$u_0^{ETH}$", 0.0, 1.0, 0.4, step=0.01)
    st.write("Random seed")
    seed_ = st.slider("seed", 0, n_mc, 0, step=1)
with col2:
    st.write("Impact of ETH price on DAI utilisation")
    alpha_dai = st.slider("$\\alpha^{DAI}$", -1.0, 1.0, 0.05, step=0.01)
    u0_dai = st.slider("$u_0^{DAI}$", 0.0, 1.0, 0.4, step=0.01)

u_eth = get_utilisation(price_paths=price_paths, u0=u0_eth, a=alpha_eth)
u_dai = get_utilisation(price_paths=price_paths, u0=u0_dai, a=alpha_dai)

# Create figure with secondary y-axis
_, col, _ = st.columns([0.1, 0.8, 0.1])
with col:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=time,
            y=price_paths[seed_, :],
            name="ETH price",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=time, y=u_eth[seed_, :], name="ETH pool utilisation"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=time, y=u_dai[seed_, :], name="DAI pool utilisation"),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="time")

    # Set y-axes titles
    fig.update_yaxes(title_text="pool utilisation", secondary_y=True)
    fig.update_yaxes(title_text="ETH-USD price", secondary_y=False)
    # fig.write_image(f"results/price_utilisation_mu{mu}_sigma{sigma}.png")

    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# Interest rate model
# --------------------------
st.subheader("Interest rate model")
r_0 = 0
r_1 = 0.04
r_2 = 2.5
u_optimal = 0.45

utilisation = np.linspace(0, 1, 100)

rate = vect_irm(
    u_optimal=u_optimal,
    r_0=r_0,
    r_1=r_1,
    r_2=r_2,
    utilisation=utilisation,
    collateral=False,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        At this point we define the interest rate defined by the protocol. In Aave, this is given by

        $$r_{IRM} (\mathcal U) = r_0 + r_1 \cdot \\frac{\mathcal U}{\mathcal U^*} \cdot \, \mathbf 1_{\mathcal U \leq \mathcal U^*} + r_2 \cdot \\frac{\mathcal U - \mathcal U^*}{1-\mathcal U^*} \cdot \, \mathbf 1_{\mathcal U > \mathcal U^*}$$
        for $r_0,r_1,r_2 >0$ and $\mathcal U^* \in [0,1)$ the targeted pool utilisation by the protocol.

        We set $r_0 = %.2f, r_1 = %.2f, r_2 = %.2f, U^*=%.2f$

        """
        % (r_0, r_1, r_2, u_optimal)
    )
with col2:
    fig = px.line(x=utilisation, y=rate, labels={"x": "utilisation", "y": "rate"})
    st.plotly_chart(fig)

r_collateral_eth = vect_irm(
    u_optimal=u_optimal, r_0=r_0, r_1=r_1, r_2=r_2, utilisation=u_eth, collateral=True
)
r_debt_dai = vect_irm(
    u_optimal=u_optimal, r_0=r_0, r_1=r_1, r_2=r_2, utilisation=u_dai, collateral=False
)

st.markdown("""---""")
st.subheader("Price and Interest rates")
_, col, _ = st.columns([0.1, 0.8, 0.1])
with col:
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=time, y=price_paths[seed_, :], name="ETH price"), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=time, y=r_collateral_eth[seed_, :], name="ETH collateral rate"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=time, y=r_debt_dai[seed_, :], name="DAI debt rate"),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="time")

    # Set y-axes titles
    fig.update_yaxes(title_text="interest rates", secondary_y=True)
    fig.update_yaxes(title_text="ETH-USD price", secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# Loan position and perps params
# --------------------------------
st.markdown("""---""")
st.subheader("Loan position and long Perps position parameters")
col1, col2, _ = st.columns([0.2, 0.2, 0.8], gap="medium")
with col1:
    st.write("Initial Loan-To-Value")
    ltv0 = st.slider(r"$\theta^0$", 0.5, 1.0, 0.75)  # min, max, default
    st.write("Liquidation threshold in Lending protocol")
    lt = st.slider(r"$\theta$", 0.5, 1.0, 0.95)  # min, max, default
with col2:
    st.write("Maintenance margin account size in Perps option trading")
    lt_f = st.slider(r"$\theta^F$", 0.0, 1.0, 0.05)  # min, max, default


# ------------------------------------
# Liquidations - PnL and Stopping time
# ------------------------------------
st.markdown("""---""")
st.subheader("PnL - funding fee - liquidation times")

col1, _ = st.columns([0.2, 0.8])
with col1:
    st.write("Time")
    t = st.slider(r"$t$", 0.0, 1.0, 0.10)

liquidation_times_lending = get_liquidation_times(
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    price_paths=price_paths,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
liquidation_times_perps = get_liquidation_times_perp(
    dt=dt,
    kappa=kappa,
    lt_f=lt_f,
    ltv0=ltv0,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r=r_debt_dai,
    r_debt_dai=r_debt_dai,
)
pnl_lending_position = get_pnl_lending_position(
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    price_paths=price_paths,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
pnl_perps = get_pnl_perps_after_liquidation(
    dt=dt,
    kappa=kappa,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r=0,  # r_debt_dai,
    lt_f=lt_f,
    ltv0=ltv0,
    r_debt_dai=r_debt_dai,
)


df_pnl_cperp = pd.DataFrame(
    pnl_lending_position.T,
    columns=["mc{}".format(i) for i in range(pnl_lending_position.shape[0])],
)
df_pnl_cperp = df_pnl_cperp.assign(time=time, derivative="cPerp")
df_pnl_perp = pd.DataFrame(
    pnl_perps.T, columns=["mc{}".format(i) for i in range(pnl_perps.shape[0])]
)
df_pnl_perp = df_pnl_perp.assign(time=time, derivative="Perp")
df_pnl = pd.concat([df_pnl_perp, df_pnl_cperp])
df_pnl_melted = pd.melt(
    df_pnl,
    id_vars=["time", "derivative"],
    value_vars=["mc{}".format(i) for i in range(pnl_lending_position.shape[0])],
    value_name="PnL",
)

st.markdown("Comparison of PnL at time {}".format(t))
_, price_col, _ = st.columns([0.1, 0.8, 0.1])
with price_col:
    fig = px.histogram(
        df_pnl_melted[df_pnl_melted["time"] == t],
        x="PnL",
        color="derivative",
        opacity=0.5,
        nbins=200,
        barmode="overlay",
        # range_x=(-400, 200),
    )
    names = {"Perp": "Perp", "cPerp": "Loan position"}
    fig.for_each_trace(
        lambda x: x.update(name=names[x.name], legendgroup=names[x.name])
    )
    # fig.write_image(f"results/pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}.png")
    st.plotly_chart(fig, use_container_width=True)
    table = (
        df_pnl_melted[df_pnl_melted["time"] == t][["derivative", "PnL"]]
        .groupby("derivative")
        .agg({"PnL": [np.mean, np.std]})
        .reset_index()
    )
    table = table.replace(to_replace="cPerp", value="Loan position")
    table.columns = ["derivative", "mean PnL", "std dev PnL"]
    st.dataframe(table, hide_index=True, use_container_width=True)

    # df_pnl_melted[df_pnl_melted["time"] == t][["derivative", "PnL"]].groupby(
    #     "derivative"
    # ).agg({"PnL": [np.mean, np.std]}).reset_index().to_csv(
    #     f"results/pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}.csv"
    # )

    cols = ["mc{}".format(i) for i in range(pnl_perps.shape[0])]
    pnl_diff = df_pnl_cperp[cols] - df_pnl_perp[cols]
    pnl_diff = pnl_diff.assign(time=time)
    pnl_diff_melted = pd.melt(
        pnl_diff, id_vars=["time"], value_vars=cols, value_name="diff_pnl"
    )
    fig = px.histogram(
        pnl_diff_melted[pnl_diff_melted["time"] == t],
        x="diff_pnl",
        opacity=0.5,
        nbins=200,
        labels={"diff_pnl": "(PnL loan position) - (PnL long perp position)"},
    )
    # fig.write_image(f"results/diff_pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}.png")
    st.plotly_chart(fig, use_container_width=True)
    table = (
        pnl_diff_melted[pnl_diff_melted["time"] == t]["diff_pnl"]
        .describe()
        .loc[["mean", "std"]]
    )
    table = table.replace(to_replace="cPerp", value="Loan position")
    st.dataframe(table, use_container_width=True)


# -------------
# Funding fee
# -------------
st.markdown("""---""")
st.markdown("Comparison of Funding Fees at time {}".format(t))
funding_fee_perp = get_funding_fee_perps(
    dt=dt,
    kappa=kappa,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r_debt_dai=r_debt_dai,
)
_, funding_fee_lending = decompose_pnl_lending_position(
    price_paths=price_paths,
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
df_funding_fee_cperp = pd.DataFrame(
    funding_fee_lending.T,
    columns=["mc{}".format(i) for i in range(funding_fee_lending.shape[0])],
)
df_funding_fee_cperp = df_funding_fee_cperp.assign(time=time, derivative="cPerp")
df_funding_fee_perp = pd.DataFrame(
    funding_fee_perp.T,
    columns=["mc{}".format(i) for i in range(funding_fee_perp.shape[0])],
)
df_funding_fee_perp = df_funding_fee_perp.assign(time=time, derivative="Perp")
df_funding_fee = pd.concat([df_funding_fee_perp, df_funding_fee_cperp])
df_funding_fee_melted = pd.melt(
    df_funding_fee,
    id_vars=["time", "derivative"],
    value_vars=["mc{}".format(i) for i in range(funding_fee_lending.shape[0])],
    value_name="funding_fee",
)


_, price_col, _ = st.columns([0.1, 0.8, 0.1])
with price_col:
    fig = px.histogram(
        df_funding_fee_melted[df_funding_fee_melted["time"] == t],
        x="funding_fee",
        color="derivative",
        opacity=0.5,
        nbins=100,
        barmode="overlay",
        # range_x=(-15, 15),
        labels={"cPerp": "Loan position", "funding_fee": "funding fee"},
    )
    names = {"Perp": "Perp", "cPerp": "Loan position"}
    fig.for_each_trace(
        lambda x: x.update(name=names[x.name], legendgroup=names[x.name])
    )
    # fig.write_image(f"results/funding_fee_mu{mu}_sigma{sigma}_thetaF{lt_f}.png")
    st.plotly_chart(fig, use_container_width=True)

    table = (
        df_funding_fee_melted[df_pnl_melted["time"] == t][["derivative", "funding_fee"]]
        .groupby("derivative")
        .agg({"funding_fee": [np.mean, np.std]})
        .reset_index()
    )
    table.columns = ["derivative", "mean funding fee", "std dev funding fee"]
    table = table.replace(to_replace="cPerp", value="Loan position")
    st.dataframe(table, hide_index=True, use_container_width=True)

    # df_funding_fee_melted[df_pnl_melted["time"] == t][
    #     ["derivative", "funding_fee"]
    # ].groupby("derivative").agg(
    #     {"funding_fee": [np.mean, np.std]}
    # ).reset_index().to_csv(
    #     f"results/funding_fee_mu{mu}_sigma{sigma}_thetaF{lt_f}.csv"
    # )


# --------------
# Stopping time
# --------------
st.markdown("""---""")
st.markdown("CDF of Liquidation time")
liquidation_times_lending = get_liquidation_times(
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    price_paths=price_paths,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
liquidation_times_perps = get_liquidation_times_perp(
    dt=dt,
    r=r_debt_dai,
    kappa=kappa,
    lt_f=lt_f,
    ltv0=ltv0,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r_debt_dai=r_debt_dai,
)
df_liquidation_times_cperps = pd.DataFrame(
    {"liquidation_times": liquidation_times_lending.flatten(), "mc": np.arange(n_mc)}
)
df_liquidation_times_cperps = df_liquidation_times_cperps.assign(derivative="cPerp")
df_liquidation_times_perps = pd.DataFrame(
    {"liquidation_times": liquidation_times_perps.flatten(), "mc": np.arange(n_mc)}
)
df_liquidation_times_perps = df_liquidation_times_perps.assign(derivative="Perp")
df = pd.concat([df_liquidation_times_perps, df_liquidation_times_cperps])
_, col, _ = st.columns([0.1, 0.8, 0.1])
with col:
    fig = px.ecdf(
        df,
        x="liquidation_times",
        color="derivative",
        labels={"liquidation_times": "liquidation times"},
        range_x=(0, 1),
    )
    names = {"Perp": "Perp", "cPerp": "Loan position"}
    fig.for_each_trace(
        lambda x: x.update(name=names[x.name], legendgroup=names[x.name])
    )
    # fig.write_image(f"results/liquidation_times_mu{mu}_sigma{sigma}.png")
    st.plotly_chart(fig, use_container_width=True)
