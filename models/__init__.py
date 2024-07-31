import inspect
from models.basemodel import BaseModel

def get_class(mod_name, base_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    mod_path = "{}.{}".format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    classes = [c for c in classes if c[1].__module__ == mod_path]
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]

def get_model(name):
    return get_class(name, __name__, BaseModel)