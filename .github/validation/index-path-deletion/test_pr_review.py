"""Additional local PR review checks, excluded from submission."""
import pytest
from tantivy import Index, Query

from paperqa import Docs
from paperqa.agents.search import SearchIndex
from paperqa.types import Doc, Text
from test_deletion import BASELINE


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_restored_real_docs_and_stored_fields_are_unchanged(tmp_path, legacy):
    fields = ["file_location", "body", "title", "year"]
    cls = BASELINE.SearchIndex if legacy else SearchIndex
    index = cls(fields=fields, index_name="test", index_directory=tmp_path)
    original_docs = {}
    for i, location in enumerate(("a-b.txt", "a/b.txt", "A.B.txt", "folder/a-b.txt")):
        doc = Doc(dockey=f"key-{i}", docname=f"Author{i}2026", citation=f"Author {i}. Example research. 2026.")
        text = Text(text=f"Research evidence for paper {i} about resilience.", name=f"Author{i}2026 pages 1-2", doc=doc, embedding=[1.0, 0.0])
        docs = Docs(docs={doc.dockey: doc}, texts=[text], docnames={doc.docname})
        original_docs[location] = docs.model_dump_json()
        await index.add_document({"file_location": location, "body": text.text,
                                  "title": f"Research title {i}", "year": str(2020+i)}, document=docs)
    await index.save_index()
    if legacy:
        index = SearchIndex(fields=fields, index_name="test", index_directory=tmp_path)
    before = {name: (await index.get_saved_object(name)).model_dump_json() for name in original_docs}
    assert before == original_docs

    await index.remove_from_index("a-b.txt")
    await index.save_index()
    expected_names = set(original_docs) - {"a-b.txt"}
    for name in expected_names:
        assert (await index.get_saved_object(name)).model_dump_json() == original_docs[name]
    matches = await index.query("resilience", keep_filenames=True, top_n=10)
    assert {name for _, name in matches} == expected_names
    assert all(isinstance(docs, Docs) for docs, _ in matches)
    native = Index.open(str(tmp_path / "test" / "index")); native.reload()
    reader = native.searcher()
    for _, address in reader.search(Query.all_query(), limit=10).hits:
        stored = reader.doc(address)
        name = stored.get_first("file_location")
        number = ("a-b.txt", "a/b.txt", "A.B.txt", "folder/a-b.txt").index(name)
        assert stored.get_first("title") == f"Research title {number}"
        assert stored.get_first("year") == str(2020 + number)
