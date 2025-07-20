def create_cls(name: str, **defaults):
    def __init__(self, **kw):
        # 先用默认值，再被实例参数覆盖
        for k, v in defaults.items():
            setattr(self, k, None)
        # 额外的实例参数也允许
        for k, v in kw.items():
            setattr(self, k, v)

    cls_dict = dict(defaults)      # 类属性
    cls_dict['__init__'] = __init__
    return type(name, (object,), cls_dict)
