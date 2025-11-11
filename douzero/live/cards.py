"""Utilities for converting between card symbols and internal ids."""

from __future__ import annotations

from typing import Dict, Optional

from douzero.env.env import RANKS, SUITS, card_to_id

SUIT_SYMBOLS = {"♣": 0, "♦": 1, "♥": 2, "♠": 3}
SUIT_INDEX_TO_SYMBOL = {idx: symbol for symbol, idx in SUIT_SYMBOLS.items()}
SYMBOL_TO_CARD: Dict[str, int] = {}
CARD_TO_SYMBOL: Dict[int, str] = {}

_symbol_to_rank = {rank: idx for idx, rank in enumerate(RANKS)}

for suit_symbol, suit_idx in SUIT_SYMBOLS.items():
    for rank_str, rank_idx in _symbol_to_rank.items():
        card_id = card_to_id((suit_idx, rank_idx))
        CARD_TO_SYMBOL[card_id] = f"{RANKS[rank_idx]}{suit_symbol}"
        SYMBOL_TO_CARD[f"{suit_symbol}{RANKS[rank_idx]}"] = card_id
        SYMBOL_TO_CARD[f"{RANKS[rank_idx]}{suit_symbol}"] = card_id


def symbol_to_card_id(symbol: str) -> Optional[int]:
    """Convert a string like ``'J♠'`` or ``'♠J'`` into an internal card id."""

    if not symbol:
        return None
    symbol = symbol.strip()
    if symbol in SYMBOL_TO_CARD:
        return SYMBOL_TO_CARD[symbol]
    suit_symbol = symbol[0]
    rank_symbol = symbol[1:]
    if suit_symbol in SUIT_SYMBOLS and rank_symbol in _symbol_to_rank:
        suit = SUIT_SYMBOLS[suit_symbol]
        rank = _symbol_to_rank[rank_symbol]
        return card_to_id((suit, rank))
    if symbol[-1] in SUIT_SYMBOLS:
        suit = SUIT_SYMBOLS[symbol[-1]]
        rank_symbol = symbol[:-1]
        if rank_symbol in _symbol_to_rank:
            rank = _symbol_to_rank[rank_symbol]
            return card_to_id((suit, rank))
    return None


def card_id_to_symbol(card_id: int) -> str:
    """Convert an internal card id back to the ``rank + suit`` unicode form."""

    if card_id in CARD_TO_SYMBOL:
        return CARD_TO_SYMBOL[card_id]
    suit_idx = card_id // len(RANKS)
    rank = RANKS[card_id % len(RANKS)]
    suit_symbol = SUIT_INDEX_TO_SYMBOL.get(suit_idx, "?")
    return f"{rank}{suit_symbol}"


__all__ = [
    "CARD_TO_SYMBOL",
    "SYMBOL_TO_CARD",
    "SUIT_INDEX_TO_SYMBOL",
    "card_id_to_symbol",
    "symbol_to_card_id",
]
