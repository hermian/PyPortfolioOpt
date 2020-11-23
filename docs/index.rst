.. image:: ../media/logo_v1-grey.png
   :scale: 40 %
   :align: center
   :alt: PyPortfolioOpt

.. raw:: html

    <meta prefix="og: http://ogp.me/ns#" property="og:title" content="PyPortfolioOpt" />
    <meta prefix="og: http://ogp.me/ns#" property="og:description" content="Portfolio optimisation in python" />
    <meta prefix="og: http://ogp.me/ns#" property="og:image" content="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/logo_v1.png"/>

    <embed>
        <p align="center">
            <a href="https://www.python.org/">
                <img src="https://img.shields.io/badge/python-v3-brightgreen.svg?style=flat-square"
                    alt="python"></a> &nbsp;
            <a href="https://pypi.org/project/PyPortfolioOpt/">
                <img src="https://img.shields.io/badge/pypi-v1.2.7-brightgreen.svg?style=flat-square"
                    alt="python"></a> &nbsp;
            <a href="https://opensource.org/licenses/MIT">
                <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square"
                    alt="MIT license"></a> &nbsp;
            <a href="https://github.com/robertmartin8/PyPortfolioOpt/graphs/commit-activity">
                <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg?style=flat-square"
                    alt="MIT license"></a> &nbsp;
        </p>
    </embed>


.. PyPortfolioOpt is a library that implements portfolio optimisation methods, including classical efficient frontier techniques and Black-Litterman allocation, as well as more recent developments in the field like shrinkage and Hierarchical Risk Parity, along with some novel experimental features like exponentially-weighted covariance matrices.

PyPortfolioOpt는 고전적인 효율적인 프론티어 기술과 Black-Litterman 할당을 포함한 포트폴리오 최적화 방법을 구현하는 라이브러리일 뿐만 아니라 수축 :sup:`shrinkage` 및 계층 위험 패리티 :sup:`Hierarchical Risk Parity` 와 같은 분야의 최근 개발과 기하급수적으로 가중된 공분산 행렬과 같은 새로운 실험 기능을 포함합니다.

.. It is **extensive** yet easily **extensible**, and can be useful for both the casual investor and the serious practitioner. Whether you are a fundamentals-oriented investor who has identified a handful of undervalued picks, or an algorithmic trader who has a basket of interesting signals, PyPortfolioOpt can help you combine your alpha-generators in a risk-efficient way.

광범위하지만 쉽게 확장할 수 있으며, 일반 투자자와 전문 실무자 모두에게 유용 할 수 있습니다. 소수의 저평가된 추천을 식별하는 기본적 분석 투자자이든 흥미로운 신호 바구니가 있는 알고리즘 트레이더이든 PyPortfolioOpt는 알파 생성기를 위험 효율적인 방식으로 결합하는 데 도움이 될 수 있습니다.


Installation
============

Installation on macOS or linux is as simple as::

    pip install PyPortfolioOpt

Windows users need to go through the additional step of downloading C++ (for ``cvxpy``). You can download this `here <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_, with additional instructions `here <https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view>`_.

For the sake of best practice, it is good to do this with a dependency manager. I suggest you set yourself up with `poetry <https://github.com/sdispater/poetry>`_, then within a new poetry project run:

.. code-block:: text

    poetry add PyPortfolioOpt

The alternative is to clone/download the project, then in the project directory run

.. code-block:: text

    python setup.py install

Thanks to Thomas Schmelzer, PyPortfolioOpt now supports Docker (requires **make**, **docker**, **docker-compose**). Build your first container with ``make build``; run tests with ``make test``. For more information, please read `this guide <https://docker-curriculum.com/#introduction>`_.

.. note::
    If any of these methods don't work, please `raise an issue
    <https://github.com/robertmartin8/PyPortfolioOpt/issues>`_  on GitHub

For developers
--------------

.. If you are planning on using PyPortfolioOpt as a starting template for significant modifications, it probably makes sense to clone this repository and to just use the source code

PyPortfolioOpt를 중요한 수정을 위한 시작 템플릿으로 사용할 계획이라면 이 리포지토리를 복제하고 소스 코드를 사용하는 것이 좋습니다.

.. code-block:: text

    git clone https://github.com/robertmartin8/PyPortfolioOpt

.. Alternatively, if you still want the convenience of ``from pypfopt import x``, you should try

또는, 당신은 여전히 ``from pypfopt import x`` 편의를 원하는 경우, 다음을 시도합니다.

.. code-block:: text

    pip install -e git+https://github.com/robertmartin8/PyPortfolioOpt.git


A Quick Example
===============

.. This section contains a quick look at what PyPortfolioOpt can do. For a guided tour, please check out the :ref:`user-guide`. For even more examples, check out the Jupyter notebooks in the `cookbook <https://github.com/robertmartin8/PyPortfolioOpt/tree/master/cookbook>`_.

이 섹션에는 PyPortfolioOpt이 수행할 수 있는 작업을 빠르게 살펴볼 수 있습니다. 가이드 투어는 :ref:`user-guide` 를 확인하십시오. 더 많은 예제는 `cookbook <https://github.com/robertmartin8/PyPortfolioOpt/tree/master/cookbook>`_ 의 Jupyter 노트북을 확인하십시오.

.. If you already have expected returns ``mu`` and a risk model ``S`` for your set of assets, generating an optimal portfolio is as easy as::

이미 기대 수익률 ``mu`` 을 가진다면 자산 집합에 대한 위험 모델 ``S`` 이 있는 경우 최적의 포트폴리오를 생성하는 것은 다음과 같이 쉽습니다.

    from pypfopt.efficient_frontier import EfficientFrontier

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()

.. However, if you would like to use PyPortfolioOpt's built-in methods for calculating the expected returns and covariance matrix from historical data, that's fine too::

그러나 과거 데이터에서 기대 수익률 및 공분산 행렬을 계산하기 위해 PyPortfolioOpt의 기본 제공 방법을 사용하려는 경우 다음과 같습니다.

    import pandas as pd
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns

    # Read in price data
    df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    ef.portfolio_performance(verbose=True)

.. This outputs the following:

이렇게 하면 출력은 다음과 같습니다.

.. code-block:: text

   Expected annual return: 33.0%
   Annual volatility: 21.7%
   Sharpe Ratio: 1.43


Contents
========

.. toctree::
    :maxdepth: 2

    UserGuide
    ExpectedReturns
    RiskModels
    EfficientFrontier
    BlackLitterman
    OtherOptimisers
    Postprocessing
    Plotting

.. toctree::
    :caption: Other information
    
    Roadmap
    Contributing
    About

Advantages over existing implementations
========================================

.. - Includes both classical methods (Markowitz 1952 and Black-Litterman), suggested   best practices (e.g covariance shrinkage), along with many recent developments and novel features, like L2 regularisation, shrunk covariance, hierarchical risk parity.
.. - Native support for pandas dataframes: easily input your daily prices data.
.. - Extensive practical tests, which use real-life data.
.. - Easy to combine with your proprietary strategies and models.
.. - Robust to missing data, and price-series of different lengths (e.g FB data only goes back to 2012 whereas AAPL data goes back to 1980).

- L2 정규화 :sup:`L2 regularisation`, 축소된 공분산 :sup:`shrunk covariance`, 계층적 위험 패리티 :sup:`hierarchical risk parity` 와 같은 많은 최근의 개발 및 새로운 기능과 함께 권장 되는 모범 사례(예 : 공부산 수축)를 모두 포함하는 고전적 메서드(Markowitz 1952 및 Black-Litterman)이 모두 포함되어 있습니다.
- pandas 데이터 프레임에 대한 기본 지원 : 쉽게 일일 가격 데이터를 입력합니다.
- 실제 데이터를 사용하는 광범위한 실용적인 테스트.
- 독점 전략 및 모델과 쉽게 결합할 수 있습니다.
- 누락된 데이터 및 다양한 길이의 가격 계열(예: FB 데이터는 2012년으로 거슬러 올라가는 반면 AAPL 데이터는 1980년으로 거슬러 올라갑니다).  


Project principles and design decisions
=======================================

- It should be easy to swap out individual components of the optimisation process with the user's proprietary improvements.
- Usability is everything: it is better to be self-explanatory than consistent.
- There is no point in portfolio optimisation unless it can be practically applied to real asset prices.
- Everything that has been implemented should be tested.
- Inline documentation is good: dedicated (separate) documentation is better. The two are not mutually exclusive.
- Formatting should never get in the way of good code: because of this, I have deferred **all** formatting decisions to `Black <https://github.com/ambv/black>`_.

- 최적화 프로세스의 개별 구성 요소를 사용자의 독점 개선으로 쉽게 교체할 수 있어야 합니다.
- 사용적합성이 모든 것입니다. 일관성보다 자명하는 것이 좋습니다.
- 실제 자산 가격에 실질적으로 적용될 수 없다면 포트폴리오 최적화에는 아무런 의미도 없습니다.
- 구현된 모든 것을 테스트해야 합니다.
- 인라인 문서는 좋습니다: 전용(별도) 문서가 더 좋습니다. 둘은 상호 배타적이지 않습니다.
- 서식은 좋은 코드의 방해가 되어서는 안됩니다: 이 때문에 모든 서식 결정을 `Black <https://github.com/ambv/black>`_ 으로연기했습니다.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
