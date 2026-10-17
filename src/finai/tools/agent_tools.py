"""The functions in this package, wrapped as LangChain tools for the agent.

The model sees each tool's name, its description (the docstring's summary)
and an argument schema with each argument's description (the docstring's
Args section, via ``parse_docstring=True``). Those words are its only guide
to when and how to call a tool, so they state units, defaults and limits.

``user_id`` is never a tool argument. ``get_user_portfolio`` takes the run's
``RunnableConfig``, which LangChain injects and leaves out of the schema; an
argument named ``config`` sent by the model is ignored (verified against
langchain-core 1.6.9).

Expected failures (an unknown ticker, Finnhub unavailable, a company not
indexed) are returned as data the model can explain. Unexpected errors raise.

The numeric tools return a ``citation`` string built from their own result,
so the model copies the source rather than composing it.
"""

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from finai.tools import filings, financials, market_data, portfolio, prices, trends

# The five items ingested (parse.TARGET_ITEMS), as an enum in the schema so
# the model sees the allowed values rather than learning them from an error.
Section = Literal["Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8"]


class MissingUserError(RuntimeError):
    """The run's config carries no user_id; the caller must authenticate."""


@tool(parse_docstring=True)
def get_live_market_data(ticker: str) -> dict[str, Any]:
    """Get the latest quote and valuation for a US-listed stock.

    Returns price (USD), % change, market cap with its currency, 52-week
    range, P/E on trailing-twelve-month and last-fiscal-year EPS with the EPS
    and the period each runs to, and the quote time in UTC. Fields Finnhub
    lacks are null, not 0; quote_time says how current the price is. When
    pe_source is "price / EPS", P/E is this quote's price over EPS per this
    share; otherwise it is Finnhub's company-level figure and the EPS is in
    eps_currency per a different share, so it must not be set against the
    price. Use for current prices and valuation, not for history.

    Args:
        ticker: US ticker symbol, e.g. "NVDA" or "BRK.B".
    """
    try:
        data = market_data.get_live_market_data(ticker)
    except market_data.TickerNotFoundError as error:
        return {"status": "not_found", "ticker": ticker.upper(), "message": str(error)}
    except market_data.MarketDataError as error:
        return {
            "status": "unavailable",
            "ticker": ticker.upper(),
            "message": str(error),
        }
    return {"status": "ok", **data.model_dump(mode="json")}


@tool(parse_docstring=True)
def query_company_10k(
    ticker: str,
    question: str,
    section: Section | None = None,
    fiscal_year: int | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Search a company's annual report (10-K) for passages answering a question.

    Returns the passages with what a citation needs: ticker, fiscal year,
    filing date, section and accession number. Without fiscal_year it
    searches the latest year indexed and lists available_years; to compare
    years, call once per year. If the company's 10-K isn't stored yet, it is
    fetched in the background: status is then "queued", "ingesting" or
    "failed", and message says when to expect it. "no_10k" means the company
    files no 10-K (a foreign filer on Form 20-F, an ETF or a fund) and
    "unknown_ticker" that SEC doesn't know the ticker; message says which.
    "year_not_indexed" means that year isn't stored. Figures in passages are
    as printed in the filing; tables give the unit and year in their caption
    and headings.

    Args:
        ticker: US ticker symbol, e.g. "AAPL".
        question: What to look for, in plain words.
        section: Limit to one part of the 10-K: "Item 1" (business), "Item 1A"
            (risk factors), "Item 7" (management's discussion and results),
            "Item 7A" (market risk) or "Item 8" (financial statements).
        fiscal_year: The company's fiscal year, e.g. 2025. NVIDIA's fiscal
            2026 ended in January 2026.
        k: Number of passages, 1 to 20.
    """
    try:
        result = filings.query_company_10k(ticker, question, section, fiscal_year, k)
    except ValueError as error:
        return {"status": "invalid_arguments", "message": str(error)}
    return result.model_dump(mode="json")


# Plain @tool: the model passes no arguments, so there is no Args section
# for parse_docstring to read (it rejects the docstring without one).
@tool
def get_user_portfolio(config: RunnableConfig) -> dict[str, Any]:
    """Get the signed-in user's holdings: ticker, shares and average cost.

    Amounts are exact decimals as strings, in USD. An empty list means the
    user holds nothing. Takes no arguments: the user comes from the session.
    """
    user_id = (config.get("configurable") or {}).get("user_id")
    if not user_id:
        raise MissingUserError("no user_id in the run's config")
    return portfolio.get_user_portfolio(user_id=user_id).model_dump(mode="json")


@tool(parse_docstring=True)
def get_financial_history(ticker: str) -> dict[str, Any]:
    """Get a company's reported annual figures and their computed trends.

    From the latest 10-K's financial statements: revenue, gross and operating
    margin, net income, diluted EPS and cash for its three fiscal years, total
    debt for two (the balance sheet has two), and revenue by product,
    geography or segment where the company tags it. Amounts are in USD, EPS in
    USD per share, margins and growth are fractions (0.064 is 6.4%). trends
    holds each year's change over the previous and, in span, the change from
    the oldest year to the newest with the compound annual growth rate.
    Margin changes are in percentage points. Growth is null, with a note, when
    a year is a loss or zero; the change is still given. A breakdown with
    adds_up_to_total false has overlapping members: report them one by one,
    never summed. Use for exact reported figures and growth rather than
    reading numbers from 10-K passages, and cite it with the returned
    citation.

    Args:
        ticker: US ticker symbol, e.g. "AAPL".
    """
    symbol = ticker.strip().upper()
    try:
        history = financials.get_financial_history(symbol)
    except LookupError as error:
        # UnknownTickerError, FilingNotFoundError (no 10-K), or a filing
        # with no annual revenue facts.
        return {"status": "not_found", "ticker": symbol, "message": str(error)}
    return {
        "status": "ok",
        "citation": f"[{symbol} 10-K FY{max(history.fiscal_years)}, Item 8]",
        **history.model_dump(mode="json"),
        "trends": trends.compute_trends(history).model_dump(mode="json"),
    }


@tool(parse_docstring=True)
def get_price_history(ticker: str, period: prices.Period = "1y") -> dict[str, Any]:
    """Get a stock's past performance: return, volatility, drawdown and range.

    From daily closes adjusted for splits and dividends, so total_return
    includes dividends. All rates are fractions (0.064 is 6.4%):
    annualised_return (3y and 5y only), annualised_volatility, and
    max_drawdown (negative) with its peak and trough dates. Returns per fiscal
    year use the company's 10-K year ends, so they line up with
    get_financial_history. week_52_high and week_52_low are the prices traded
    over the year to end_date. full_period is false when the stock has traded
    for less than the period; start_date says where it begins. Data runs to
    the last completed trading day, end_date. Cite with the returned citation.

    Args:
        ticker: US ticker symbol, e.g. "NVDA" or "BRK.B".
        period: How far back: "1y", "3y" or "5y".
    """
    symbol = ticker.strip().upper()
    try:
        history = prices.get_price_history(symbol, period)
    except market_data.TickerNotFoundError as error:
        return {"status": "not_found", "ticker": symbol, "message": str(error)}
    except prices.PriceHistoryError as error:
        return {"status": "unavailable", "ticker": symbol, "message": str(error)}
    except ValueError as error:
        return {"status": "invalid_arguments", "message": str(error)}
    return {
        "status": "ok",
        "citation": f"[Alpaca prices, to {history.end_date.isoformat()}]",
        **history.model_dump(mode="json"),
    }


TOOLS = [
    get_live_market_data,
    query_company_10k,
    get_user_portfolio,
    get_financial_history,
    get_price_history,
]
