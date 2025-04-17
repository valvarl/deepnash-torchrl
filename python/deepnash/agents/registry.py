MODULE_REGISTRY = {}


def register_module(name=None):
    def decorator(cls):
        module_name = name or cls.__name__
        if module_name in MODULE_REGISTRY:
            raise ValueError(f"Module {module_name} is already registered.")
        MODULE_REGISTRY[module_name] = cls
        return cls

    return decorator


def build_module(name, *args, **kwargs):
    if name not in MODULE_REGISTRY:
        raise ValueError(f"Module {name} is not registered.")
    return MODULE_REGISTRY[name](*args, **kwargs)
