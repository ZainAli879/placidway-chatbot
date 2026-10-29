import importlib
import importlib.util
import pkgutil

import app


def test_application_uses_chroma_without_sentence_transformers_or_torch(monkeypatch):
    forbidden = {"sentence_transformers", "torch"}
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"production imported forbidden dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    module_names = [info.name for info in pkgutil.walk_packages(app.__path__, "app.")]
    for module_name in module_names:
        importlib.import_module(module_name)
    assert importlib.util.find_spec("chromadb") is not None
    assert importlib.util.find_spec("app.retrieval.hybrid_search") is not None
