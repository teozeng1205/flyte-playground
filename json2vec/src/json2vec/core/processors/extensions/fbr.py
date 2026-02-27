from json2vec.core.processors.base import register


@register
def shim_fbr(observation: dict, **kwargs) -> list[dict]|None:

    observation["fareclass"] = list(observation.get("fareclass", "") or "")

    observation["disc_date"] = (
        str(observation["disc_date"])
        if str(observation["disc_date"]) != "999999"
        else None
    )

    # ensure timestamp matches pattern YYMMDDHHMM else coerce to None
    observation["timestamp"] = (
        str(observation["timestamp"])
        if str(observation["timestamp"]).isdigit() and len(str(observation["timestamp"])) == 10
        else None
    )

    return [observation]