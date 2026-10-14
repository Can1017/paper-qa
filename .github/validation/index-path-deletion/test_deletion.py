"""Local validation only; no tests added to the PR."""

import contextlib
import json
import os
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from tantivy import Index, Query, SchemaBuilder

from paperqa.agents.search import SearchIndex, get_directory_index
from paperqa import Settings

BASELINE = types.ModuleType("paperqa.agents._baseline_search")
exec(compile(Path(os.environ["PQA_DELETION_BASELINE"]).read_text() if os.getenv("PQA_DELETION_BASELINE") else subprocess.check_output(["git", "show", "main:src/paperqa/agents/search.py"], text=True),
             "upstream-search.py", "exec"), BASELINE.__dict__)


class Payload(BaseModel):
    data: dict


async def populate(root, locations, legacy=False, shared_body=False):
    cls = BASELINE.SearchIndex if legacy else SearchIndex
    index = cls(index_name="test", index_directory=root)
    for n, location in enumerate(locations):
        body = "resilience research shared body" if shared_body else f"resilience research document {n}"
        await index.add_document({"file_location": location, "body": body},
                                 document=Payload(data={"text": body}))
    await index.save_index()
    return SearchIndex(index_name="test", index_directory=root) if legacy else index


def raw_paths(root):
    index = Index.open(str(root / "test" / "index"))
    index.reload()
    reader = index.searcher()
    if not reader.num_docs:
        return []
    return sorted(reader.doc(addr).get_first("file_location")
                  for _, addr in reader.search(Query.all_query(), limit=reader.num_docs).hits)


PATHS = ["plain", "last.txt", "papers/last.txt", r"C:\papers\last.txt",
         "papers/a-b.txt", "PAPERS/A-B.TXT", 'papers/[a] ("quoted").pdf',
         "璁烘枃/娴嬭瘯鏂囦欢.pdf", "x" * 80 + ".txt", "a" * 80, "!!!"]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("keep_sibling", [False, True])
@pytest.mark.parametrize("location", PATHS)
async def test_exact_deletion(tmp_path, legacy, keep_sibling, location):
    locations = [location, "keeper"] if keep_sibling else [location]
    index = await populate(tmp_path, locations, legacy=legacy)
    assert await index.count == len(locations)
    await index.remove_from_index(location)
    await index.save_index()
    expected = ["keeper"] if keep_sibling else []
    assert await index.count == len(expected)
    assert raw_paths(tmp_path) == expected
    assert sorted(await index.index_files) == expected
    loaded = SearchIndex(index_name="test", index_directory=tmp_path)
    assert await loaded.count == len(expected)
    assert sorted(await loaded.index_files) == expected
    if keep_sibling:
        assert await loaded.get_saved_object("keeper")
        assert await loaded.query("resilience")
    await index.delete_document(location)
    await index.delete_document("missing.txt")
    assert raw_paths(tmp_path) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("target", ["a-b.txt", "a/b.txt", "A.B.txt", "a_b.txt"])
async def test_tokenizer_collisions_preserve_other_files(tmp_path, legacy, target):
    locations = ["a-b.txt", "a/b.txt", "A.B.txt", "a_b.txt", "other.md"]
    index = await populate(tmp_path, locations, legacy=legacy)
    await index.remove_from_index(target)
    await index.save_index()
    assert raw_paths(tmp_path) == sorted(set(locations) - {target})
    assert len(await index.query("resilience", top_n=10)) == 4


@pytest.mark.asyncio
async def test_shared_cache_is_retained_for_other_file(tmp_path):
    index = await populate(tmp_path, ["first.txt", "second.txt"], shared_body=True)
    await index.remove_from_index("first.txt")
    await index.save_index()
    assert raw_paths(tmp_path) == ["second.txt"]
    assert (await index.get_saved_object("second.txt")).data == {"text": "resilience research shared body"}
    assert len(await index.query("resilience")) == 1


@pytest.mark.asyncio
async def test_uncommitted_sibling_is_not_lost(tmp_path):
    index = SearchIndex(index_name="test", index_directory=tmp_path)
    for location in ("a-b.txt", "a/b.txt"):
        await index.add_document({"file_location": location, "body": location},
                                 document=Payload(data={"path": location}))
    await index.remove_from_index("a-b.txt")
    await index.save_index()
    assert raw_paths(tmp_path) == ["a/b.txt"]
    assert (await index.get_saved_object("a/b.txt")).data == {"path": "a/b.txt"}


@pytest.mark.asyncio
async def test_cached_reader_refreshes_after_addition(tmp_path):
    index = await populate(tmp_path, ["first.txt"])
    assert await index.count == 1
    await index.add_document({"file_location": "second.txt", "body": "a different document"},
                             document=Payload(data={"text": "second"}))
    await index.save_index()
    assert await index.count == 2


@pytest.mark.asyncio
async def test_restoration_failure_rolls_back_deletion(tmp_path):
    index = await populate(tmp_path, ["a-b.txt", "a/b.txt"])
    original_writer = index.writer

    class FaultyWriter:
        def __init__(self, writer): self.writer = writer
        def delete_documents_by_query(self, query): return self.writer.delete_documents_by_query(query)
        def add_document(self, document): raise RuntimeError("synthetic restoration failure")
        def rollback(self): return self.writer.rollback()
        def wait_merging_threads(self): return self.writer.wait_merging_threads()

    @contextlib.asynccontextmanager
    async def writer(reset=False):
        async with original_writer(reset=reset) as native:
            yield FaultyWriter(native)

    with patch.object(index, "writer", writer), pytest.raises(RuntimeError, match="restoration"):
        await index.delete_document("a-b.txt")
    assert raw_paths(tmp_path) == ["a-b.txt", "a/b.txt"]
    assert set(await index.index_files) == {"a-b.txt", "a/b.txt"}
    # The same instance remains usable after rollback.
    await index.remove_from_index("a-b.txt")
    await index.save_index()
    assert raw_paths(tmp_path) == ["a/b.txt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
async def test_directory_sync_removes_last_file(tmp_path, absolute):
    papers = tmp_path / "papers"
    papers.mkdir()
    paper = papers / "last.txt"
    paper.write_text("Research on resilient systems and document processing. " * 30)
    manifest = papers / "manifest.csv"
    manifest.write_text("file_location,title,year\nlast.txt,last,2026\n")
    settings = Settings(embedding="sparse", parsing={"use_doc_details": False, "multimodal": False},
                        agent={"index": {"name": "test", "paper_directory": papers,
                        "index_directory": tmp_path, "manifest_file": manifest,
                        "use_absolute_paper_directory": absolute}})
    first = await get_directory_index(settings)
    assert await first.count == 1
    paper.unlink()
    emptied = await get_directory_index(settings)
    assert await emptied.count == 0
    assert await emptied.index_files == {}
    assert raw_paths(tmp_path) == []
