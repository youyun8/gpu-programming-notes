"""Scaled-down Tensara test cases for problems whose official tests are too big to emulate.

The official test generators close over their sizes via lambda default
arguments, so a smaller case is built by re-binding those defaults.
"""


def with_dims(tc, dims, arg="m"):
    """Clone a test case whose create_inputs closes over `arg`=dims and whose extra params read tc['dims']."""
    create = tc["create_inputs"]
    new = dict(tc)
    new["name"] = f"{tc.get('name', 'case')} -> {dims}"
    new["dims"] = dims
    new["create_inputs"] = lambda: create(**{arg: dims})
    return new


SMALL = {}


def extra_cases(prob, slug):
    make = SMALL.get(slug)
    return make(prob) if make else []
