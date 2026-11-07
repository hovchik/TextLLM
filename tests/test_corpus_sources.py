"""Offline checks of the license and part filters for the Common Pile source.

    python tests/test_corpus_sources.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import build_corpus as bc  # noqa: E402


def test_open_license_keeps_open_licenses():
    for lic in ["CC BY 4.0", "CC BY-SA", "https://creativecommons.org/licenses/by-sa/4.0/",
                "Creative Commons - Attribution Share-Alike", "Public Domain", "", None]:
        assert bc.open_license(lic), lic


def test_open_license_refuses_nc_nd():
    for lic in ["CC BY-NC 4.0", "cc-by-nd", "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "Creative Commons - Attribution NonCommercial", "Attribution-NoDerivatives 4.0",
                "Attribution Non-Commercial"]:
        assert not bc.open_license(lic), lic


def test_wiki_filter():
    keep = bc._wiki_keep(bc.CP_WIKIS)
    assert keep({"source": "wiki/dump/wikibooks.com"})
    assert keep({"source": "x", "metadata": {"wiki": "wikivoyage"}})
    assert not keep({"source": "wiki/dump/wikipedia.com"})
    assert not keep({"source": "wiki/dump/wiktionary.com"})


def test_unknown_common_pile_part():
    try:
        bc.common_pile("unused", 1, ["books3"])
    except SystemExit as e:
        assert "books3" in str(e)
    else:
        raise AssertionError("unknown part accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
