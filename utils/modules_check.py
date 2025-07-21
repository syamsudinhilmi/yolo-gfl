import importlib
import inspect
import pkgutil

from torch.nn import Module as TorchModule


def list_modules_in_package(package_name: str) -> list[str]:
    """
    Import a package, iterate its submodules, and collect
    all classes that subclass torch.nn.Module.
    """
    module = importlib.import_module(package_name)
    found = set()

    # 1) Collect classes in the root module itself
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, TorchModule):
            found.add(name)

    # 2) If it's a package, walk its submodules
    if hasattr(module, "__path__"):
        for finder, subname, ispkg in pkgutil.iter_modules(module.__path__):
            full_name = f"{package_name}.{subname}"
            try:
                submod = importlib.import_module(full_name)
            except ImportError:
                continue
            for name, obj in inspect.getmembers(submod, inspect.isclass):
                if issubclass(obj, TorchModule):
                    found.add(name)

    return sorted(found)

def main():
    categories = ["conv", "block", "head", "utils"]
    base_pkg = "ultralytics.nn.modules"
    all_modules = {}

    for cat in categories:
        pkg_name = f"{base_pkg}.{cat}"
        try:
            classes = list_modules_in_package(pkg_name)
            all_modules[cat] = classes
        except ModuleNotFoundError:
            all_modules[cat] = []

    # Print out in a YAML-friendly way
    for cat, classes in all_modules.items():
        print(f"# modules in {base_pkg}.{cat}")
        for cls in classes:
            print(f"- {cls}")
        print("")

if __name__ == "__main__":
    main()
