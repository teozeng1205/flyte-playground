from datetime import datetime
from functools import cache

from json2vec.core.processors.base import register

USD_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.91,
    "GBP": 0.78,
    "JPY": 143.5,
    "AUD": 1.52,
    "CAD": 1.36,
    "CHF": 0.84,
    "CNY": 7.12,
    "HKD": 7.82,
    "SGD": 1.34,
    "NZD": 1.65,
    "SEK": 10.5,
    "NOK": 10.7,
    "DKK": 6.79,
    "CZK": 22.8,
    "PLN": 3.75,
    "HUF": 358.0,
    "RON": 4.55,
    "BGN": 1.78,
    "HRK": 6.85,
    "TRY": 34.1,
    "RUB": 92.0,
    "UAH": 41.0,
    "KZT": 470.0,
    "GEL": 2.75,
    "AZN": 1.70,
    "AED": 3.6725,
    "SAR": 3.75,
    "QAR": 3.64,
    "BHD": 0.376,
    "OMR": 0.3845,
    "KWD": 0.306,
    "ILS": 3.65,
    "JOD": 0.709,
    "LBP": 89500.0,
    "EGP": 49.0,
    "MAD": 9.8,
    "TND": 3.1,
    "DZD": 134.0,
    "LYD": 4.9,
    "MRO": 39.5,
    "MUR": 46.0,
    "ZAR": 18.3,
    "NAD": 18.3,
    "BWP": 13.7,
    "ZMW": 24.5,
    "MZN": 63.0,
    "AOA": 870.0,
    "NGN": 1550.0,
    "GHS": 15.8,
    "KES": 129.0,
    "TZS": 2600.0,
    "UGX": 3750.0,
    "ETB": 125.0,
    "RWF": 1320.0,
    "BIF": 2850.0,
    "CDF": 2850.0,
    "XAF": 565.0,
    "XOF": 565.0,
    "XPF": 109.0,
    "ZWL": 13.0,  # ZWL is volatile (proxy)
    "ARS": 920.0,
    "CLP": 930.0,
    "COP": 4070.0,
    "PEN": 3.8,
    "BOB": 6.9,
    "PYG": 7600.0,
    "UYU": 39.0,
    "BRL": 5.12,
    "MXN": 17.1,
    "GTQ": 7.75,
    "HNL": 24.6,
    "NIO": 36.8,
    "CRC": 525.0,
    "PAB": 1.0,
    "BBD": 2.0,
    "BSD": 1.0,
    "DOP": 59.0,
    "JMD": 157.0,
    "TTD": 6.8,
    "HTG": 132.0,
    "CUP": 24.0,
    "CUC": 1.0,
    "XCD": 2.7,
    "ANG": 1.79,
    "AWG": 1.79,
    "KYD": 0.833,
    "BMD": 1.0,
    "VES": 36.0,  # very volatile; treated as ballpark
    "INR": 83.2,
    "PKR": 279.0,
    "BDT": 118.0,
    "LKR": 300.0,
    "NPR": 133.0,
    "BTN": 83.2,
    "MVR": 15.4,
    "SCR": 14.5,
    "AFN": 71.0,
    "IRR": 420000.0,
    "IQD": 1310.0,
    "KHR": 4100.0,
    "LAK": 21500.0,
    "MMK": 3500.0,
    "MNT": 3450.0,
    "KGS": 86.0,
    "UZS": 12600.0,
    "TJS": 10.9,
    "TMT": 3.5,
    "AMD": 390.0,
    "BAM": 1.78,
    "MKD": 56.0,
    "ISK": 138.0,
    "ALL": 92.0,
    "RSD": 107.0,
    "MDL": 17.7,
    "BYN": 3.3,
    "XAG": 0.03,
    "XAU": 0.0004,  # metals (per 1 USD)
    "KRW": 1360.0,
    "TWD": 32.5,
    "PHP": 58.3,
    "MYR": 4.5,
    "THB": 36.0,
    "IDR": 16000.0,
    "VND": 25400.0,
    "BND": 1.34,
    "KHM": 4100.0,
    "CVE": 101.0,
    "SHP": 0.78,
    "SLL": 20900.0,
    "GNF": 8650.0,
    "MWK": 1730.0,
    "SZL": 18.3,
    "LSL": 18.3,
    "SOS": 57000.0,
    "SDG": 595.0,
    "ERN": 15.0,
    "DJF": 177.0,
    "GMD": 68.0,
    "MGA": 4500.0,
    "MRU": 39.5,
    "STN": 23.3,
    "SYP": 13000.0,
    "KPW": 900.0,  # highly controlled; placeholder
    "YER": 250.0,
    "ARM": 390.0,  # ARM non-ISO; keep AZN/AMD used above
    "SRD": 36.0,
    "GYD": 209.0,
    "BZD": 2.0,
    "FJD": 2.25,
    "WST": 2.8,
    "TOP": 2.35,
    "PGK": 3.8,
    "SBD": 8.5,
    "VUV": 120.0,
    "XDR": 0.76,  # IMF SDR per USD (approx)
    "MOP": 8.06,
}


@cache
def get_rate_to_usd(code: str) -> float:
    """
    Return the approximate rate for 1 USD in the given currency code.
    Raises ValueError if the code isn't in the table.
    """
    code = code.upper()
    try:
        return 1 / USD_RATES[code]
    except KeyError:
        raise ValueError(f"Currency {code} not found in offline table.")

@register
def shim_shopping_data(observation: dict, **kwargs):
    for col in [
        "outboundStops",
        "inboundStops",
        "outboundOperatingCarrier",
        "inboundOperatingCarrier",
        "outboundDestinationAirportCode",
        "inboundDestinationAirportCode",
        "outboundOriginAirportCode",
        "inboundOriginAirportCode",
        "outboundMarketingCarrier",
        "inboundMarketingCarrier",
    ]:
        observation[col] = str(observation[col]).split("|")

    if "multiCarrier" in observation.keys():
        observation["multiCarrier"] = str(observation["multiCarrier"])

    if observation["returnDate"] in ["0", 0]:
        observation["returnDate"] = None
    else:
        observation["returnDate"] = str(observation["returnDate"])

    if observation["departDate"] in ["0", 0]:
        observation["departDate"] = None
    else:
        observation["departDate"] = str(observation["departDate"])

    if "totalAmount" in observation.keys() and observation["totalAmount"] is not None:
        observation["totalAmount"] *= get_rate_to_usd(observation["currencyCode"])

    if "tax_yr_amount" in observation.keys() and observation["tax_yr_amount"] is not None:
        observation["tax_yr_amount"] *= get_rate_to_usd(observation["taxCurrencyCode"])

    if "tax_yq_amount" in observation.keys() and observation["tax_yq_amount"] is not None:
        observation["tax_yq_amount"] *= get_rate_to_usd(observation["taxCurrencyCode"])

    # print(observation["timestamp"])

    observed_date = datetime.fromtimestamp(int(observation["timestamp"]) * 0.001)

    observation["age"] = (datetime.strptime(kwargs['current_date'], "%Y-%m-%d") - observed_date).total_seconds() * 0.01666666666

    observation["total_tax"] = (observation.get("tax_yq_amount", 0) or 0) + (observation.get("tax_yr_amount", 0) or 0)

    return [observation]
