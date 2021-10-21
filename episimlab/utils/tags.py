from collections.abc import Sequence

"""
Work in progress
"""

def _add_tags(obj, tags):
    raise NotImplementedError()
    assert isinstance(obj, (
        # class,
        # function
        bool
    ))


def tag(tag_name):
    """Decorator factory that returns a `tag_decorator`, which adds string
    `tag_name`s to the decorated object's `TAGS` attribute.
    """
    def tag_decorator(obj):
        if isinstance(tag_name, str):
            tag_name = [tag_name]
        elif (isinstance(tag_name, Sequence) and
              all((isinstance(t, str) for t in tag_name))):
            pass
        else:
            raise TypeError("kwarg 'tag_name' must be a string or sequence " +
                            "containing strings")
        return _add_tags(obj=obj, tags=tag_name)
    return tag_decorator
