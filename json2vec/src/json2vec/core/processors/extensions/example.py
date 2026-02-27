from json2vec.core.processors.base import register


@register
def default(item):
    return [item]
