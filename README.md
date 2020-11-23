<p align="center">
    <img width=60% src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/logo_v1.png">
</p>

<!-- buttons -->
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://pypi.org/project/PyPortfolioOpt/">
        <img src="https://img.shields.io/badge/pypi-v1.2.7-brightgreen.svg"
            alt="pypi"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
    <a href="https://github.com/robertmartin8/PyPortfolioOpt/graphs/commit-activity">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg"
            alt="issues"></a> &nbsp;
    <a href="https://pyportfolioopt.readthedocs.io/en/latest/">
        <img src="https://img.shields.io/badge/docs-passing-brightgreen.svg"
            alt="docs"></a> &nbsp;
    <a href="https://travis-ci.org/robertmartin8/PyPortfolioOpt">
        <img src="https://travis-ci.org/robertmartin8/PyPortfolioOpt.svg?branch=master"
            alt="travis"></a> &nbsp;
    <a href="https://mybinder.org/v2/gh/robertmartin8/pyportfolioopt/master">
        <img src="https://mybinder.org/badge_logo.svg"
            alt="binder"></a> &nbsp;
</p>

<!-- content -->

https://github.com/robertmartin8/PyPortfolioOpt

<!--
PyPortfolioOpt is a library that implements portfolio optimisation methods, including classical mean-variance optimisation techniques and Black-Litterman allocation, as well as more recent developments in the field like shrinkage and Hierarchical Risk Parity, along with some novel experimental features like exponentially-weighted covariance matrices.
-->

PyPortfolioOpt는 기존 평균 분산 최적화 기술 및 Black-Litterman 할당을 포함한 포트폴리오 최적화 방법뿐만 아니라 축소<sup>shrinkage</sup> 및 계층적 위험 패리티<sup>Hierarchical Risk Parity</sup>와 같은 분야의 최근 개발과 지수 가중치 공분산 행렬<sup>exponentially-weighted covariance matrices</sup>과 같은 일부 새로운 실험 기능을 구현하는 라이브러리입니다.

<!--
It is **extensive** yet easily **extensible**, and can be useful for both the casual investor and the serious practitioner. Whether you are a fundamentals-oriented investor who has identified a handful of undervalued picks, or an algorithmic trader who has a basket of interesting signals, PyPortfolioOpt can help you combine your alpha streams in a risk-efficient way.
-->

**광범위**<sup>extensive</sup>하지만 **쉽게 확장**<sup>extensible</sup> 할 수 있으며 일반 투자자와 진지한 실무자 모두에게 유용 할 수 있습니다. 소수의 저평가 된 픽을 식별 한 펀더멘털 지향 투자자이든 흥미로운 신호들을 보유한 알고리즘 트레이더이든 PyPortfolioOpt는 위험 효율적인 방식<sup>risk-efficient way</sup>으로 알파 스트림을 결합하는 데 도움을 줄 수 있습니다.

<!--
Head over to the [documentation on ReadTheDocs](https://pyportfolioopt.readthedocs.io/en/latest/) to get an in-depth look at the project, or check out the [cookbook](https://github.com/robertmartin8/PyPortfolioOpt/tree/master/cookbook) to see some examples showing the full process from downloading data to building a portfolio.
-->

[ReadTheDocs](https://pyportfolioopt.readthedocs.io/en/latest/)의 문서로 이동하여 프로젝트에 대해 자세히 살펴 보거나 [쿡북](https://github.com/robertmartin8/PyPortfolioOpt/tree/master/cookbook)을 확인하여 데이터 다운로드에서 포트폴리오 구축에 이르는 전체 프로세스를 보여주는 몇 가지 예제를 확인하십시오.

<center>
<img src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/conceptual_flowchart_v2.png" style="width:70%;"/>
</center>

## Table of contents

- [Table of contents](#table-of-contents)
- [Getting started](#getting-started)
  - [For development](#for-development)
- [A quick example](#a-quick-example)
- [What's new](#whats-new)
- [An overview of classical portfolio optimisation methods](#an-overview-of-classical-portfolio-optimisation-methods)
- [Features](#features)
  - [Expected returns](#expected-returns)
  - [Risk models (covariance)](#risk-models-covariance)
  - [Objective functions](#objective-functions)
  - [Adding constraints or different objectives](#adding-constraints-or-different-objectives)
  - [Black-Litterman allocation](#black-litterman-allocation)
  - [Other optimisers](#other-optimisers)
- [Advantages over existing implementations](#advantages-over-existing-implementations)
- [Project principles and design decisions](#project-principles-and-design-decisions)
- [Roadmap](#roadmap)
- [Testing](#testing)
- [Contributing](#contributing)
- [Getting in touch](#getting-in-touch)

## Getting started

<!--
If you would like to play with PyPortfolioOpt interactively in your browser, you may launch Binder [here](https://mybinder.org/v2/gh/robertmartin8/pyportfolioopt/master). It takes a while to set up, but it lets you try out the cookbook recipes without having to deal with all of the requirements.
-->
브라우저에서 대화식으로 PyPortfolioOpt를 플레이하려면 [여기](https://mybinder.org/v2/gh/robertmartin8/pyportfolioopt/master)에서 Binder를 실행할 수 있습니다. 설정하는 데 시간이 걸리지만 모든 요구 사항을 처리하지 않고도 쿡북 레시피를 시험해 볼 수 있습니다.

<!-- *Note: if you are on windows, you first need to installl C++. ([download](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16), [install instructions](https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view))* -->
참고 : Windows를 사용하는 경우 먼저 C ++를 설치해야합니다. ([다운로드](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16), [설치 지침](https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view))

<!-- This project is available on PyPI, meaning that you can just: -->
이 프로젝트는 PyPI를 사용할 수 있습니다. 즉, 다음을 수행 할 수 있습니다.

```bash
pip install PyPortfolioOpt
```

<!-- However, it is best practice to use a dependency manager within a virtual environment.My current recommendation is to get yourself set up with [poetry](https://github.com/sdispater/poetry) then just run -->
그러나 가상 환경 내에서 종속성 관리자를 사용하는 것이 가장 좋습니다. 현재 권장 사항은 [poetry](https://github.com/sdispater/poetry)를 설정한 다음 실행하는 것입니다.

```bash
poetry add PyPortfolioOpt
```

<!-- Otherwise, clone/download the project and in the project directory run: -->
그렇지 않으면 프로젝트를 복제/다운로드하고 프로젝트 디렉토리에서 다음을 실행합니다.

```bash
python setup.py install
```

<!-- Thanks to Thomas Schmelzer, PyPortfolioOpt now supports Docker (requires **make**, **docker**, **docker-compose**). Build your first container with `make build`; run tests with `make test`. For more information, please read [this guide](https://docker-curriculum.com/#introduction). -->

Thomas Schmelzer 덕분에 PyPortfolioOpt는 이제 Docker를 지원합니다 (**make**, **docker**, **docker-compose** 필요). `make build`로 첫 번째 컨테이너를 빌드하십시오. `make test`로 테스트를 실행하십시오. 자세한 내용은 이 [가이드](https://docker-curriculum.com/#introduction)를 참조하십시오.

### For development

<!-- If you would like to make major changes to integrate this with your proprietary system, it probably makes sense to clone this repository and to just use the source code. -->
이를 당신의 시스템과 통합하기 위해 주요 변경을 수행하려면 이 저장소를 복제하고 소스 코드를 사용하는 것이 좋습니다.

```bash
git clone https://github.com/robertmartin8/PyPortfolioOpt
```

<!-- Alternatively, you could try: -->
또는 다음을 시도 할 수 있습니다.

```bash
pip install -e git+https://github.com/robertmartin8/PyPortfolioOpt.git
```

## A quick example

<!-- Here is an example on real life stock data, demonstrating how easy it is to find the long-only portfolio that maximises the Sharpe ratio (a measure of risk-adjusted returns). -->
다음은 Sharpe 비율(위험 조정 수익의 척도)을 최대화하는 매수 전용 포트폴리오<sup>long-only portfolio</sup>를 찾는 것이 얼마나 쉬운 지 보여주는 실제 주식 데이터의 예입니다.

```python
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in price data
df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
```

<!-- This outputs the following weights: -->
그러면 다음 가중치가 출력됩니다.

```txt
{'GOOG': 0.01269,
 'AAPL': 0.09202,
 'FB': 0.19856,
 'BABA': 0.09642,
 'AMZN': 0.07158,
 'GE': 0.0,
 'AMD': 0.0,
 'WMT': 0.0,
 'BAC': 0.0,
 'GM': 0.0,
 'T': 0.0,
 'UAA': 0.0,
 'SHLD': 0.0,
 'XOM': 0.0,
 'RRC': 0.0,
 'BBY': 0.06129,
 'MA': 0.24562,
 'PFE': 0.18413,
 'JPM': 0.0,
 'SBUX': 0.03769}

Expected annual return: 33.0%
Annual volatility: 21.7%
Sharpe Ratio: 1.43
```

<!-- This is interesting but not useful in itself. However, PyPortfolioOpt provides a method which allows you to convert the above continuous weights to an actual allocation that you could buy. Just enter the most recent prices, and the desired portfolio size ($10,000 in this example): -->
이것은 흥미롭지만 그 자체로는 유용하지 않습니다. 그러나 PyPortfolioOpt는 위의 연속 가중치를 매수할 수 있는 실제 할당으로 변환 할 수 있는 방법을 제공합니다. 가장 최근 가격과 원하는 포트폴리오 크기 (이 예에서는 $10,000)를 입력하기만 하면 됩니다.

```python
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
```

```txt
11 out of 20 tickers were removed
Discrete allocation: {'GOOG': 0, 'AAPL': 5, 'FB': 11, 'BABA': 5, 'AMZN': 1,
                      'BBY': 7, 'MA': 14, 'PFE': 50, 'SBUX': 5}
Funds remaining: $8.42
```

<!-- *Disclaimer: nothing about this project constitues investment advice, and the author bears no responsibiltiy for your subsequent investment decisions. Please refer to the [license](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/LICENSE.txt) for more information.* -->
*면책 조항 :이 프로젝트에 대한 어떤 것도 투자 조언을 구성하지 않으며 저자는 후속 투자 결정에 대해 책임을지지 않습니다. 자세한 내용은 [라이센스]https://github.com/robertmartin8/PyPortfolioOpt/blob/master/LICENSE.txt)를 참조하십시오.*

## What's new

As of v1.2.0:

- Docker support
- Idzorek's method for specifying Black-Litterman views using percentage confidences.
- Industry constraints: limit your sector exposure.
- Multiple additions and improvements to `risk_models`:
  - Introduced a new API, in which the function `risk_models.risk_matrix(method="...")` allows
    all the different risk models to be called. This should make testing easier.
  - All methods now accept returns data instead of prices, if you set the flag `returns_data=True`.
- Automatically fix non-positive semidefinite covariance matrices!
- Additions and improvements to `expected_returns`:
  - Introduced a new API, in which the function `expected_returns.return_model(method="...")` allows
    all the different return models to be called. This should make testing easier.
  - Added option to 'properly' compound returns.
  - CAPM return model.
- `from pypfopt import plotting`: moved all plotting functionality into a new class and added
  new plots. All other plotting functions (scattered in different classes) have been retained,
  but are now deprecated.

## An overview of classical portfolio optimisation methods

<!-- Harry Markowitz's 1952 paper is the undeniable classic, which turned portfolio optimisation from an art into a science. The key insight is that by combining assets with different expected returns and volatilities, one can decide on a mathematically optimal allocation which minimises the risk for a target return – the set of all such optimal portfolios is referred to as the **efficient frontier**. -->
Harry Markowitz의 1952년 논문은 포트폴리오 최적화를 예술에서 과학으로 바꾼 부인할 수 없는 고전입니다. 주요 통찰은 자산을 서로 다른 기대 수익률 및 변동성과 결합함으로써 목표 수익률에 대한 위험을 최소화하는 수학적으로 최적의 할당을 결정할 수 있다는 것입니다. 이러한 모든 최적의 포트폴리오 세트를 **효율적인 프론티어**<sup>efficient frontier</sup>라고합니다.

<center>
<img src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/efficient_frontier_white.png" style="width:60%;"/>
</center>

<!-- Although much development has been made in the subject, more than half a century later, Markowitz's core ideas are still fundamentally important and see daily use in many portfolio management firms. The main drawback of mean-variance optimisation is that the theoretical treatment requires knowledge of the expected returns and the future risk-characteristics (covariance) of the assets. Obviously, if we knew the expected returns of a stock life would be much easier, but the whole game is that stock returns are notoriously hard to forecast. As a substitute, we can derive estimates of the expected return and covariance based on historical data – though we do lose the theoretical guarantees provided by Markowitz, the closer our estimates are to the real values, the better our portfolio will be. -->

이 주제에서 많은 발전이 이루어졌지만 반세기가 지난 후에도 Markowitz의 핵심 아이디어는 여전히 근본적으로 중요하며 많은 포트폴리오 관리 회사에서 매일 사용됩니다. 평균 분산 최적화<sup>mean-variance optimisation</sup>의 가장 큰 단점은 이론적 처리에 예상 수익률<sup>expected returns</sup>과 자산의 미래 위험 특성 (공분산)<sup>uture risk-characteristics (covariance)</sup>에 대한 지식이 필요하다는 것입니다. 분명히 우리가 주식 수명의 기대 수익률이 훨씬 쉬울 것이라는 것을 안다면, 전체 게임은 주식 수익률을 예측하기가 매우 어렵다는 것입니다. 대안으로 과거 데이터를 기반으로 예상 수익률과 공분산 추정치를 도출 할 수 있습니다. Markowitz가 제공한 이론적 보증을 잃더라도 추정치가 실제 값에 가까울수록 포트폴리오가 더 좋아집니다.

<!-- Thus this project provides four major sets of functionality (though of course they are intimately related) -->
따라서 이 프로젝트는 4 가지 주요 기능 세트를 제공합니다(물론 밀접하게 관련되어 있음).

<!-- 
- Estimates of expected returns
- Estimates of risk (i.e covariance of asset returns)
- Objective functions to be optimised
- Optimisers. 
-->

- 기대 수익률 추정
- 위험 추정 (즉, 자산 수익률의 공분산)
- 최적화 할 목적 함수
- 옵티 마이저.

<!-- A key design goal of PyPortfolioOpt is **modularity** – the user should be able to swap in their components while still making use of the framework that PyPortfolioOpt provides. -->
PyPortfolioOpt의 핵심 디자인 목표는 **모듈성**<sup>modularity</sup>입니다. 사용자는 PyPortfolioOpt가 제공하는 프레임워크를 사용하면서 구성 요소를 교체 할 수 있어야합니다.

## Features

<!-- In this section, we detail PyPortfolioOpt's current available functionality as per the above breakdown. More examples are offered in the Jupyter notebooks [here](https://github.com/robertmartin8/PyPortfolioOpt/tree/master/cookbook). Another good resource is the [tests](https://github.com/robertmartin8/PyPortfolioOpt/tree/master/tests). -->
이 섹션에서는 위의 분석에 따라 PyPortfolioOpt의 현재 사용 가능한 기능에 대해 자세히 설명합니다. 여기 Jupyter 노트북에서 더 많은 [예제]((ttps://github.com/robertmartin8/PyPortfolioOpt/tree/master/cookbook)가 제공됩니다. 또 다른 좋은 리소스는 [테스트](https://github.com/robertmartin8/PyPortfolioOpt/tree/master/tests)입니다.

<!-- A far more comprehensive version of this can be found on [ReadTheDocs](https://pyportfolioopt.readthedocs.io/en/latest/), as well as possible extensions for more advanced users. -->
이것의 훨씬 더 포괄적인 버전은 [ReadTheDocs](https://pyportfolioopt.readthedocs.io/en/latest/)에서 찾을 수 있으며 고급 사용자를 위한 가능한 확장 기능도 있습니다.

### Expected returns

<!--
- Mean historical returns:
    - the simplest and most common approach, which states that the expected return of each asset is equal to the mean of its historical returns.
    - easily interpretable and very intuitive
- Exponentially weighted mean historical returns:
    - similar to mean historical returns, except it gives exponentially more weight to recent prices
    - it is likely the case that an asset's most recent returns hold more weight than returns from 10 years ago when it comes to estimating future returns.
- Capital Asset Pricing Model (CAPM):
    - a simple model to predict returns based on the beta to the market
    - this is used all over finance! 
-->

- 평균 역사적 수익<sup>Mean historical returns</sup> :
    - 각 자산의 예상 수익률<sup>expected returns</sup>이 과거 수익률<sup>historical returns</sup>의 평균과 같다는 가장 간단하고 일반적인 접근 방식입니다.
    - 쉽게 해석 가능하고 매우 직관적
- 지수 가중치 평균 과거 수익률 :
    - 평균 역사적 수익률과 유사하지만 최근 가격에 기하 급수적으로 더 많은 가중치를 부여합니다.
    - 미래 수익률을 추정 할 때 자산의 가장 최근 수익률이 10년 전 수익률보다 더 많은 비중을 차지하는 경우 일 수 있습니다.
- 자본 자산 가격 모델<sup>Capital Asset Pricing Model</sup> (CAPM) :
    - 시장 베타를 기반으로 수익을 예측하는 간단한 모델
    - 이것은 금융 전반에 걸쳐 사용됩니다!    

### Risk models (covariance)

<!-- The covariance matrix encodes not just the volatility of an asset, but also how it correlated to other assets. This is important because in order to reap the benefits of diversification (and thus increase return per unit risk), the assets in the portfolio should be as uncorrelated as possible. -->
공분산 행렬은 자산의 변동성 뿐 아니라 다른 자산과의 상관관계도 인코딩합니다. 이는 다각화<sup>diversification</sup>의 이점을 누리고 (따라서 단위 위험 당 수익을 높이기 위해) 포트폴리오의 자산이 가능한 한 상관관계가 없어야하기 때문에 중요합니다.
<!-- 
- Sample covariance matrix:
    - an unbiased estimate of the covariance matrix
    - relatively easy to compute
    - the de facto standard for many years
    - however, it has a high estimation error, which is particularly dangerous in mean-variance optimisation because the optimiser is likely to give excess weight to these erroneous estimates.
- Semicovariance: a measure of risk that focuses on downside variation.
- Exponential covariance: an improvement over sample covariance that gives more weight to recent data
- Covariance shrinkage: techniques that involve combining the sample covariance matrix with a structured estimator, to reduce the effect of erroneous weights. PyPortfolioOpt provides wrappers around the efficient vectorised implementations provided by `sklearn.covariance`.
    - manual shrinkage
    - Ledoit Wolf shrinkage, which chooses an optimal shrinkage parameter. We offer three shrinkage targets: `constant_variance`, `single_factor`, and `constant_correlation`.
    - Oracle Approximating Shrinkage
- Minimum Covariance Determinant:
    - a robust estimate of the covariance
    - implemented in `sklearn.covariance` 
-->

- 샘플 공분산 행렬<sup>sample covariance matrix</sup> :
    - 공분산 행렬의 편향되지 않은 추정
    - 비교적 계산하기 쉬움
    - 수년간 사실상의 표준
    - 그러나 이는 추정 오류가 높으며, 이는 최적화 프로그램이 이러한 잘못된 추정에 초과 가중치를 부여 할 가능성이 높기 때문에 평균 분산 최적화에서 특히 위험합니다.
- 반공 분산<sup>semicovariance</sup> : 하방 변동<sup>downside variation</sup>에 초점을 맞춘 위험 측정.
- 지수 공분산<sup>exponential covariance</sup> : 최근 데이터에 더 많은 가중치를 부여하여 샘플 공분산 개선
- 공분산 축소<sup>covariance shrinkage</sup> : 잘못된 가중치의 영향을 줄이기 위해 샘플 공분산 행렬을 구조화된 추정기와 결합하는 기술입니다. PyPortfolioOpt는 `sklearn.covariance`에서 제공하는 효율적인 벡터화된 구현에 대한 래퍼를 제공합니다.
    - 수동 수축<sup>manual shrinkage</sup>
    - Ledoit Wolf 수축, 최적의 수축 매개 변수를 선택. 세 가지 수축 목표 : `constant_variance`, `single_factor` 및 `constant_correlation`을 제공합니다.
    - Oracle 근사 수축<sup>Approximating Shrinkage</sup>
- 최소 공분산 결정<sup>Minimum Covariance Determinant</sup> :
    - 공분산의 강건한 추정
    - `sklearn.covariance`에서 구현 됨

<p align="center">
    <img width=60% src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/corrplot_white.png">
</p>

<!-- (This plot was generated using `plotting.plot_covariance`) -->
(이 플롯은 `plotting.plot_covariance`를 사용하여 생성되었습니다)

### Objective functions

<!-- - Maximum Sharpe ratio: this results in a *tangency portfolio* because on a graph of returns vs risk, this portfolio corresponds to the tangent of the efficient frontier that has a y-intercept equal to the risk-free rate. This is the default option because it finds the optimal return per unit risk.
- Minimum volatility. This may be useful if you're trying to get an idea of how low the volatility *could* be, but in practice it makes a lot more sense to me to use the portfolio that maximises the Sharpe ratio.
- Efficient return, a.k.a. the Markowitz portfolio, which minimises risk for a given target return – this was the main focus of Markowitz 1952
- Efficient risk: the Sharpe-maximising portfolio for a given target risk.
- Maximum quadratic utility. You can provide your own risk-aversion level and compute the appropriate portfolio. -->

- 최대 샤프 비율<sup>Maximum Sharpe ratio</sup> : 수익 대 위험의 그래프에서 이 포트폴리오는 무위험 비율과 동일한 y 절편을 갖는 효율적인 프론티어의 탄젠트에 해당하기 때문에 *접선 포트폴리오*<sup>tangency portfolio</sup>가 됩니다. 단위 위험 당 최적의 수익률을 찾기 때문에 이것이 기본 옵션입니다.
- 최소 변동성<sup>Minimum volatility</sup>. 변동성이 얼마나 낮을 수 있는지에 대한 아이디어를 얻으려는 경우 유용 할 수 있지만 실제로는 Sharpe 비율을 최대화하는 포트폴리오를 사용하는 것이 훨씬 더 합리적입니다.
- 효율적인 수익<sup>Efficient return</sup> 즉, 주어진 목표 수익에 대한 위험을 최소화하는 효율적인 수익, 일명 Markowitz 포트폴리오 – 이것이 Markowitz 1952의 주요 논점이었습니다.
- 효율적인 위험<sup>Efficient risk</sup> : 주어진 목표 위험에 대한 Sharpe 최대화 포트폴리오.
- 최대 2차 효용<sup>Maximum quadratic utility</sup>. 자신의 위험 회피 수준<sup>risk-aversion level</sup>을 제공하고 적절한 포트폴리오를 계산할 수 있습니다.

### Adding constraints or different objectives

<!-- - Long/short: by default all of the mean-variance optimisation methods in PyPortfolioOpt are long-only, but they can be initialised to allow for short positions by changing the weight bounds: -->
- Long/short : 기본적으로 PyPortfolioOpt의 모든 평균 분산 최적화 방법은 long-only이지만 가중치 경계를 변경하여 공매도(short positions)를 허용하도록 초기화 할 수 있습니다.

```python
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
```

- Market neutrality: for the `efficient_risk` and `efficient_return` methods, PyPortfolioOpt provides an option to form a market-neutral portfolio (i.e weights sum to zero). This is not possible for the max Sharpe portfolio and the min volatility portfolio because in those cases because they are not invariant with respect to leverage. Market neutrality requires negative weights:
- 시장 중립성<sup>Market neutrality</sup> : `efficiency_risk` 및 `efficiency_return` 메서드의 경우 PyPortfolioOpt는 시장 중립 포트폴리오를 형성하는 옵션을 제공합니다(즉, 가중치 합계가 0이 됨). 이는 최대 Sharpe 포트폴리오와 최소 변동성 포트폴리오에서는 불가능합니다. 왜냐하면 이러한 경우 레버리지와 관련하여 변하지 않기 때문입니다. 시장 중립성은 음의 가중치를 필요로 합니다.

```python
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
ef.efficient_return(target_return=0.2, market_neutral=True)
```

<!-- - Minimum/maximum position size: it may be the case that you want no security to form more than 10% of your portfolio. This is easy to encode: -->
- 최소/최대 포지션 크기 : 포트폴리오의 10% 이상을 구성하는 증권<sup>security</sup>를 원하지 않는 경우 일 수 있습니다. 이것은 인코딩하기 쉽습니다.

```python
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))
```

<!-- One issue with mean-variance optimisation is that it leads to many zero-weights. While these are "optimal" in-sample, there is a large body of research showing that this characteristic leads mean-variance portfolios to underperform out-of-sample. To that end, I have introduced an objective function that can reduce the number of negligible weights for any of the objective functions. Essentially, it adds a penalty (parameterised by `gamma`) on small weights, with a term that looks just like L2 regularisation in machine learning. It may be necessary to try several `gamma` values to achieve the desired number of non-negligible weights. For the test portfolio of 20 securities, `gamma ~ 1` is sufficient -->

평균 분산 최적화의 한 가지 문제는 많은 0 가중치<sup>zero-weights</sup>로 이어진다는 것입니다. 이것이 "최적(optimal)"인 표본<sup>in-sample</sup>이지만, 이 특성이 평균 분산 포트폴리오가 표본 외<sup>out-of-sample</sup>의 성과를 저조하게 만든다는 것을 보여주는 많은 연구가 있습니다. 이를 위해 모든 목적 함수에 대해 무시할 수 있는 가중치 수를 줄일 수있는 목적 함수를 도입했습니다. 기본적으로 머신러닝의 L2 정규화<sup>L2 regularisation</sup>와 유사한 용어를 사용하여 작은 가중치에 대한 패널티 (`gamma`로 매개 변수화 됨)를 추가합니다. 무시할 수 없는 원하는 가중치 수를 얻기 위해 여러 `gamma` 값을 시도해야 할 수 있습니다. 20개 증권의 테스트 포트폴리오의 경우 `gamma ~ 1`이면 충분합니다.

```python
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=1)
ef.max_sharpe()
```

### Black-Litterman allocation

<!-- As of v0.5.0, we now support Black-Litterman asset allocation, which allows you to combine a prior estimate of returns (e.g the market-implied returns) with your own views to form a posterior estimate. This results in much better estimates of expected returns than just using the mean historical return. Check out the [docs](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html) for a discussion of the theory, as well as advice on formatting inputs. -->

v0.5.0부터 이제 Black-Litterman 자산 할당을 지원하므로 수익에 대한 이전 추정치 (예 : 시장 내재 수익률<sup>he market-implied returns</sup>)와 자신의 견해를 결합하여 사후 추정치를 형성 할 수 있습니다. 그 결과 평균 과거 수익률을 사용하는 것보다 기대 수익률을 훨씬 더 잘 추정 할 수 있습니다. 이론에 대한 토론과 입력 형식 지정에 대한 조언은 [문서](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html)를 확인하십시오.

```python
S = risk_models.sample_cov(df)
viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
bl = BlackLittermanModel(S, pi="equal", absolute_views=viewdict, omega="default")
rets = bl.bl_returns()

ef = EfficientFrontier(rets, S)
ef.max_sharpe()
```

### Other optimisers

<!-- The features above mostly pertain to solving efficient frontier optimisation problems via quadratic programming (though this is taken care of by `cvxpy`). However, we offer different optimisers as well: -->
위의 기능은 주로 2차<sup>quadratic</sup> 프로그래밍을 통해 효율적인 프론티어 최적화 문제를 해결하는 것과 관련이 있습니다(이는 `cvxpy`에서 처리됨). 그러나 우리는 다른 최적화 도구도 제공합니다.

<!-- 
- Hierarchical Risk Parity, using clustering algorithms to choose uncorrelated assets
- Markowitz's critical line algorithm (CLA) -->

- 계층적 위험 패리티<sup>Hierarchical Risk Parity</sup>, 상관 관계가 없는 자산을 선택하기 위해 클러스터링 알고리즘을 사용
- Markowitz의 임계선<sup>critical line</sup> 알고리즘 (CLA)

<!-- Please refer to the [documentation](https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimisers.html) for more. -->
자세한 내용은 [설명서](https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimisers.html)를 참조하십시오.

## Advantages over existing implementations
(기존 구현 대비 장점)
<!-- 
- Includes both classical methods (Markowitz 1952 and Black-Litterman), suggested best practices (e.g covariance shrinkage), along with many recent developments and novel features, like L2 regularisation, shrunk covariance, hierarchical risk parity.
- Native support for pandas dataframes: easily input your daily prices data.
- Extensive practical tests, which use real-life data.
- Easy to combine with your proprietary strategies and models.
- Robust to missing data, and price-series of different lengths (e.g FB data
  only goes back to 2012 whereas AAPL data goes back to 1980). -->

- 두 가지 고전적 방법(Markowitz 1952 및 Black-Litterman), 제안된 모범 사례 (예 : 공분산 축소), L2 정규화<sup>L2 regularisation</sup>, 축소 공분산<sup>shrunk covariance</sup>, 계층적 위험 패리티<sup>herarchical risk parity</sup>와 같은 많은 최근 개발 및 새로운 기능이 포함됩니다.
- Pandas 데이터 프레임에 대한 기본 지원 : 일일 가격 데이터를 쉽게 입력합니다.
- 실제 데이터를 사용하는 광범위한 실제 테스트.
- 독점<sup>proprietary</sup> 전략 및 모델과 쉽게 결합 할 수 있습니다.
- 누락된 데이터 및 다양한 길이의 가격 시리즈에 견고합니다 (예 : FB 데이터는 2012년으로 거슬러 올라가고 AAPL 데이터는 1980년으로 거슬러 올라갑니다).

## Project principles and design decisions
<!-- 
- It should be easy to swap out individual components of the optimisation process
  with the user's proprietary improvements.
- Usability is everything: it is better to be self-explanatory than consistent.
- There is no point in portfolio optimisation unless it can be practically
  applied to real asset prices.
- Everything that has been implemented should be tested.
- Inline documentation is good: dedicated (separate) documentation is better.
  The two are not mutually exclusive.
- Formatting should never get in the way of coding: because of this,
  I have deferred **all** formatting decisions to [Black](https://github.com/ambv/black). -->

- 최적화 프로세스의 개별 구성 요소를 사용자의 독점적인 개선 사항으로 쉽게 교체 할 수 있어야 합니다.
- 사용성은 모든 것입니다. 일관성보다는 자명한 것이 낫습니다.
- 실제 자산 가격에 실제로 적용 할 수 없는 경우 포트폴리오 최적화에는 의미가 없습니다.
- 구현된 모든 것을 테스트해야합니다.
- 인라인 문서가 좋습니다. 전용(별도의) 문서가 더 좋습니다. 둘은 상호 배타적이지 않습니다.
- 서식 지정이 코딩에 방해가되어서는 안됩니다.이 때문에 모든 서식 지정 결정을 [Black](https://github.com/ambv/black)으로 연기했습니다.

## Roadmap

<!-- Feel free to raise an issue requesting any new features – here are some of the things I want to implement: -->
새로운 기능을 요청하는 문제를 자유롭게 제기하십시오. 구현하고 싶은 몇 가지 사항은 다음과 같습니다.
<!-- 
- Optimising for higher moments (i.e skew and kurtosis)
- Factor modelling: doable but not sure if it fits within the API.
- Proper CVaR optimisation – remove NoisyOpt and use linear programming
- More objective functions, including the Calmar Ratio, Sortino Ratio, etc.
- Monte Carlo optimisation with custom distributions
- Open-source backtests using either `Backtrader <https://www.backtrader.com/>`_ or
  `Zipline <https://github.com/quantopian/zipline>`_.
- Further support for different risk/return models -->

- 더 높은 모멘텀<sup>higher moments</sup>에 최적화 (예 : 왜곡<sup>skew</sup> 및 첨도<sup>kurtosis</sup>)
- 팩터 모델링 : 가능하지만 API에 맞는지 확실하지 않습니다.
- 적절한 CVaR 최적화 – NoisyOpt 제거 및 선형 프로그래밍 사용
- Calmar Ratio, Sortino Ratio 등 더 많은 객관적인 함수
- 사용자 지정 분포를 사용한 Monte Carlo 최적화
- `Backtrader`(https://www.backtrader.com/)_ 또는 `Zipline`(https://github.com/quantopian/zipline)_을 사용하는 오픈 소스 백테스트.
- 다양한 위험/수익 모델에 대한 추가 지원

## Testing

<!-- Tests are written in pytest (much more intuitive than `unittest` and the variants in my opinion), and I have tried to ensure close to 100% coverage. Run the tests by navigating to the package directory and simply running `pytest` on the command line. -->
테스트는 pytest로 작성되었으며 (제 생각에는 `unittest`와 변형보다 훨씬 직관적입니다), 100%에 가까운 범위를 보장하려고 노력했습니다. 패키지 디렉토리로 이동하고 명령 줄에서 pytest를 실행하여 테스트를 실행합니다.

<!-- PyPortfolioOpt provides a test dataset of daily returns for 20 tickers: -->
PyPortfolioOpt는 20 개의 티커에 대한 일일 수익률 테스트 데이터 세트를 제공합니다.

```python
['GOOG', 'AAPL', 'FB', 'BABA', 'AMZN', 'GE', 'AMD', 'WMT', 'BAC', 'GM',
'T', 'UAA', 'SHLD', 'XOM', 'RRC', 'BBY', 'MA', 'PFE', 'JPM', 'SBUX']
```

 <!-- These tickers have been informally selected to meet several criteria:

- reasonably liquid
- different performances and volatilities
- different amounts of data to test robustness

Currently, the tests have not explored all of the edge cases and combinations
of objective functions and parameters. However, each method and parameter has
been tested to work as intended. -->

이러한 티커는 몇 가지 기준을 충족하도록 비공식적으로 선택되었습니다.

- 합리적인 유동성
- 다른 성능과 변동성
- 견고성을 테스트하기 위한 다양한 양의 데이터

현재 테스트에서는 모든 엣지 케이스와 목적 함수 및 매개 변수의 조합을 탐색하지 않았습니다. 그러나 각 방법 및 매개 변수는 의도 한대로 작동하도록 테스트되었습니다.

## Contributing

Contributions are *most welcome*. Have a look at the [Contribution Guide](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/CONTRIBUTING.md) for more.

## Getting in touch
연락하기

If you would like to reach out for any reason, be it consulting opportunities or just for a chat, please do so via the [form](https://reasonabledeviations.com/about/) on my website.
