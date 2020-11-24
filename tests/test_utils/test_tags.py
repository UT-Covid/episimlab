import pytest
import logging
import xsimlab as xs

import episimlab.utils.tags


@pytest.fixture(params=[
    'naked_class', 'process', 'func', 'model'
])
def example_obj(request) -> dict:
    # function/class defs
    def example_func():
        pass

    @xs.process
    class ExampleProcess:
        pass

    class ExampleClass:
        pass

    # construct dict of defs
    obj_dict = dict(
        naked_class=ExampleClass,
        process=ExampleProcess,
        func=example_func,
        model=xs.Model(dict())
    )
    # use request as key
    return obj_dict[request.param]


@pytest.mark.parametrize('tags, expected', [
    ('tag1', ['tag1']),
    (['tag1'], ['tag1']),
    (['tag1', 'tag2'], ['tag1', 'tag2']),
])
def test_can_add_tag(example_obj, tags, expected):
    tagged_obj = episimlab.utils.tags.tag(tag_name=tags)(example_obj)
    for tag in expected:
        assert tag in tagged_obj._tags
