# -*- coding: utf-8 -*-

"""Multimodal KGE Models.

.. [kristiadi2018] Kristiadi, A.., *et al.* (2018) `Incorporating literals into knowledge graph embeddings.
   <https://arxiv.org/abs/1802.00934>`_. *arXiv*, 1802.00934.
"""

from .base import LiteralModel
from .complex_literal import ComplExLiteral
from .distmult_literal import DistMultLiteral
from .transe_text import TransEText
from .transe_textv1 import TransETextV1
from .transe_textv2 import TransETextV2
from .transe_textv3 import TransETextV3
from .transe_textv4 import TransETextV4 
from .rotate_text import RotatEText
from .distmult_text import DistMultText
from .proje_text import ProjEText
__all__ = [
    'LiteralModel',
    'ComplExLiteral',
    'DistMultLiteral',
    'TransEText',
    'TransETextV1',
    'TransETextV2',
    'TransETextV3',
    'TransETextV4',
    'RotatEText',
    'DistMultText',
    'ProjEText'
]
