from flask import Flask


class ViewFunc:
    def __init__(self, func, rule, **options):
        self.rule = rule
        self.options = options
        self.func = func
        self.func_self = None
        self.__name__ = func.__name__

    def __repr__(self):
        return f"{self.rule}, {self.func}, {self.options}"

    def __get__(self, instance, owner):
        self.func_self = instance
        return self

    def __call__(self, *args, **kwargs):
        return self.func(self.func_self, *args, **kwargs)


def route(rule, **options):
    def decorator(func):
        return ViewFunc(func, rule, **options)

    return decorator


class FlaskApp:

    def __init__(self, name):
        self.app = Flask(name)
        self.register_routes()

    def register_routes(self):
        for cls in self.__class__.mro():
            for name in vars(cls):
                v = getattr(self, name)
                if isinstance(v, ViewFunc):
                    self.app.route(v.rule, **v.options)(v)
                    print("register:", v)

    def start(self,
              host: str = None,
              port: int = None,
              debug: bool = None,
              load_dotenv: bool = True,
              **options):
        self.app.run(host, port, debug, load_dotenv, **options)


if __name__ == '__main__':
    class MyFlaskApp(FlaskApp):
        @route('/config', methods=['GET'])
        def config(self):
            pass
            return ""

        @route('/get_state', methods=['POST'])
        def get_state(self):
            pass
            return ""


    app = MyFlaskApp("main")
