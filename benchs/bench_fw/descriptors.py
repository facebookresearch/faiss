# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class IndexDescriptor:
    factory: str
    bucket: Optional[str] = None
    path: Optional[str] = None
    parameters: Optional[dict[str, int]] = None
    # range metric definitions
    # key: name
    # value: one of the following:
    #
    # radius
    #    [0..radius) -> 1
    #    [radius..inf) -> 0
    #
    # [[radius1, score1], ...]
    #    [0..radius1) -> score1
    #    [radius1..radius2) -> score2
    #
    # [[radius1_from, radius1_to, score1], ...]
    #    [radius1_from, radius1_to) -> score1,
    #    [radius2_from, radius2_to) -> score2
    range_metrics: Optional[dict[str, Any]] = None


@dataclass
class DatasetDescriptor:
    namespace: Optional[str] = None
    tablename: Optional[str] = None
    partitions: Optional[List[str]] = None
    num_vectors: Optional[int] = None

    def __hash__(self):
        return hash(self.get_filename())

    def get_filename(
        self,
        prefix: str = "v",
    ) -> str:
        filename = prefix + "_"
        if self.namespace is not None:
            filename += self.namespace + "_"
        assert self.tablename is not None
        filename += self.tablename
        if self.partitions is not None:
            filename += "_" + "_".join(self.partitions).replace("=", "_")
        if self.num_vectors is not None:
            filename += f"_{self.num_vectors}"
        filename += "."
        return filename
