from collections.abc import Iterable


def _add_tags(obj, tags):
    assert isinstance(obj, (
        # class,
        # function
        bool
    ))


def tag(tag_name):
    def tag_decorator(obj):
        if isinstance(tag_name, str):
            tag_name = [tag_name]
        elif (isinstance(tag_name, Iterable) and
              all((isinstance(t, str) for t in tag_name))):
            pass
        else:
            raise TypeError("kwarg 'tag_name' must be a string or iterable " +
                            "containing strings")
        return _add_tags(obj=obj, tags=tag_name)
    return tag_decorator
