from json2vec.core.processors.base import register


@register
def shim_prodperf(observation: dict, **kwargs) -> list[dict]|None:

    if observation.get("status") != 'Match': 
        return

    if not observation.get("fare_calc_resolved"):
        return

    observation["fbas_cd"] = list(observation["fbas_cd"])

    return [observation]

# @register
# def shim_bfr(observation: dict, **kwargs):

#     for seq_key, sequence in observation.items():

#         for brand_key, brand in sequence.items():

#             deference(brand["fareIdTableNumber"])

#             deference(brand["transportSummary"])

#             clean(brand["SubCode TierValue"])

#             yield ...