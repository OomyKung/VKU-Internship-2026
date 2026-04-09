"""Unit tests for Node2Vec-style embedding generation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import networkx as nx

from fim_hybrid.node2vec_embeddings import (
    Node2VecConfig,
    build_node2vec_cache_path,
    generate_node2vec_embeddings,
)


class Node2VecEmbeddingTestCase(unittest.TestCase):
    """Check deterministic coverage and cache reuse for Node2Vec embeddings."""

    def test_generate_node2vec_embeddings_covers_all_nodes(self) -> None:
        graph = nx.path_graph(6)

        result = generate_node2vec_embeddings(
            graph,
            Node2VecConfig(
                dimensions=4,
                walk_length=6,
                num_walks=4,
                window=2,
                random_seed=7,
            ),
        )

        self.assertEqual(set(result.embedding_frame["node_id"]), set(graph.nodes()))
        self.assertEqual(len(result.embedding_frame.filter(like="node2vec_").columns), 4)
        self.assertFalse(result.loaded_from_cache)

    def test_generate_node2vec_embeddings_reuses_cache(self) -> None:
        graph = nx.path_graph(6)
        config = Node2VecConfig(
            dimensions=4,
            walk_length=6,
            num_walks=4,
            window=2,
            random_seed=7,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = build_node2vec_cache_path(Path(temp_dir), "toy_graph", config)
            self.assertIsNotNone(cache_path)

            first = generate_node2vec_embeddings(graph, config, cache_path=cache_path)
            second = generate_node2vec_embeddings(graph, config, cache_path=cache_path)

            self.assertTrue(cache_path.is_file())
            self.assertFalse(first.loaded_from_cache)
            self.assertTrue(second.loaded_from_cache)
            self.assertTrue(first.embedding_frame.equals(second.embedding_frame))

    def test_generate_node2vec_embeddings_can_scale_and_reduce_dimensions(self) -> None:
        graph = nx.path_graph(6)

        result = generate_node2vec_embeddings(
            graph,
            Node2VecConfig(
                dimensions=8,
                walk_length=6,
                num_walks=4,
                window=2,
                scale_embeddings=True,
                pca_components=3,
                random_seed=7,
            ),
        )

        self.assertEqual(len(result.embedding_frame.filter(like="node2vec_").columns), 3)
        self.assertEqual(set(result.embedding_frame["node_id"]), set(graph.nodes()))


if __name__ == "__main__":
    unittest.main()
