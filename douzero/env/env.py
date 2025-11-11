from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SUITS = ["clubs", "diamonds", "hearts", "spades"]
RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
NUM_PLAYERS = 2
NUM_CARDS = len(SUITS) * len(RANKS)
MAX_ATTACK_CARDS = 6
ACTION_END_ATTACK = 36
ACTION_TAKE_CARDS = 37
NUM_ACTIONS = ACTION_TAKE_CARDS + 1

COUNTS_VECTOR_LENGTH = 12
STATE_VECTOR_LENGTH = (
    4 * NUM_CARDS  # player hand, table attack, table defense, seen cards
    + len(SUITS)
    + len(RANKS)
    + 3  # last winner one-hot
    + 2  # current player one-hot
    + 2  # attacker one-hot
    + 2  # defender one-hot
    + COUNTS_VECTOR_LENGTH
)
ACTION_VECTOR_LENGTH = NUM_ACTIONS


@dataclass
class DurakState:
    hands: List[List[int]]
    talon: List[int]
    discard: List[int]
    table: List[Tuple[int, Optional[int]]]
    attacker: int
    defender: int
    phase: str  # "attack" or "defense"
    trump_suit: int
    trump_card: int
    seen_cards: set[int] = field(default_factory=set)
    round_attack_limit: int = MAX_ATTACK_CARDS
    terminal: bool = False
    winner: Optional[int] = None
    last_round_winner: Optional[int] = None
    defender_taking: bool = False
    post_take_additions_remaining: int = 0

    def copy(self) -> "DurakState":
        return DurakState(
            hands=[hand.copy() for hand in self.hands],
            talon=self.talon.copy(),
            discard=self.discard.copy(),
            table=[(atk, defn) for atk, defn in self.table],
            attacker=self.attacker,
            defender=self.defender,
            phase=self.phase,
            trump_suit=self.trump_suit,
            trump_card=self.trump_card,
            seen_cards=self.seen_cards.copy(),
            round_attack_limit=self.round_attack_limit,
            terminal=self.terminal,
            winner=self.winner,
            last_round_winner=self.last_round_winner,
            defender_taking=self.defender_taking,
            post_take_additions_remaining=self.post_take_additions_remaining,
        )


def card_to_id(card: Tuple[int, int]) -> int:
    suit, rank = card
    return suit * len(RANKS) + rank


def id_to_card(card_id: int) -> Tuple[int, int]:
    suit = card_id // len(RANKS)
    rank = card_id % len(RANKS)
    return suit, rank


def card_suit(card_id: int) -> int:
    return card_id // len(RANKS)


def card_rank(card_id: int) -> int:
    return card_id % len(RANKS)


def initial_state(rng: np.random.Generator) -> DurakState:
    deck = rng.permutation(NUM_CARDS).tolist()
    hands: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
    for _ in range(6):
        for pid in range(NUM_PLAYERS):
            hands[pid].append(deck.pop())
    trump_card = deck.pop()
    trump_suit = card_suit(trump_card)
    talon = deck
    talon.append(trump_card)
    for hand in hands:
        hand.sort()
    attacker = _choose_initial_attacker(hands, trump_suit)
    defender = 1 - attacker
    seen_cards = {trump_card}
    state = DurakState(
        hands=hands,
        talon=talon,
        discard=[],
        table=[],
        attacker=attacker,
        defender=defender,
        phase="attack",
        trump_suit=trump_suit,
        trump_card=trump_card,
        seen_cards=seen_cards,
        round_attack_limit=_compute_attack_limit(hands[defender]),
    )
    state.last_round_winner = None
    return state


def _choose_initial_attacker(hands: Sequence[Sequence[int]], trump_suit: int) -> int:
    lowest_trumps: List[Tuple[int, int]] = []
    for pid, hand in enumerate(hands):
        trumps = [card for card in hand if card_suit(card) == trump_suit]
        if trumps:
            lowest = min(trumps, key=card_rank)
            lowest_trumps.append((card_rank(lowest), pid))
    if lowest_trumps:
        _, pid = min(lowest_trumps)
        return pid
    return 0


def _compute_attack_limit(defender_hand: Sequence[int]) -> int:
    return min(MAX_ATTACK_CARDS, max(1, len(defender_hand)))


def legal_actions(state: DurakState) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    if state.terminal:
        return mask
    if state.phase == "attack":
        attacker_hand = state.hands[state.attacker]
        if state.defender_taking:
            playable: List[int] = []
            if (
                state.post_take_additions_remaining > 0
                and len(state.table) < MAX_ATTACK_CARDS
            ):
                ranks_on_table = _ranks_on_table(state.table)
                playable = [card for card in attacker_hand if card_rank(card) in ranks_on_table]
            for card in playable:
                mask[card] = True
            mask[ACTION_END_ATTACK] = True
        elif state.table:
            if _all_cards_covered(state.table):
                if len(state.table) < state.round_attack_limit and attacker_hand:
                    ranks_on_table = _ranks_on_table(state.table)
                    playable = [card for card in attacker_hand if card_rank(card) in ranks_on_table]
                else:
                    playable = []
                for card in playable:
                    mask[card] = True
                if state.table and _all_cards_covered(state.table):
                    mask[ACTION_END_ATTACK] = True
            else:
                pass
        else:
            for card in attacker_hand:
                mask[card] = True
        mask[ACTION_TAKE_CARDS] = False
        if not state.table and not attacker_hand:
            mask[:] = False
    else:
        defender_hand = state.hands[state.defender]
        uncovered = _uncovered_attack_cards(state.table)
        if uncovered:
            targets = [atk for _, atk in uncovered]
            for card in defender_hand:
                if any(_can_cover(card, atk, state.trump_suit) for atk in targets):
                    mask[card] = True
            mask[ACTION_TAKE_CARDS] = True
        if state.table and not uncovered:
            mask[:] = False
        mask[ACTION_END_ATTACK] = False
    return mask


def apply_action(state: DurakState, action: int) -> Tuple[DurakState, bool, Optional[int]]:
    if state.terminal:
        return state, True, state.winner
    if state.phase == "attack":
        if action == ACTION_END_ATTACK:
            if state.defender_taking:
                _finalize_take(state)
            elif state.table and _all_cards_covered(state.table):
                _finish_round_with_defense(state)
            else:
                raise ValueError("Cannot end attack before defender covers all cards.")
        else:
            _play_attack_card(state, action)
    else:
        if action == ACTION_TAKE_CARDS:
            _defender_takes(state)
        else:
            _defend_card(state, action)
    _check_terminal(state)
    return state, state.terminal, state.winner


def _play_attack_card(state: DurakState, card: int) -> None:
    if card not in state.hands[state.attacker]:
        raise ValueError("Attacker does not hold this card.")
    if state.table:
        if state.defender_taking:
            if len(state.table) >= MAX_ATTACK_CARDS:
                raise ValueError("Attack limit reached for this round.")
            if state.post_take_additions_remaining <= 0:
                raise ValueError("No additional cards allowed after defender takes.")
            ranks = _ranks_on_table(state.table)
            if card_rank(card) not in ranks:
                raise ValueError("Attack card must match rank already on table.")
        else:
            if not _all_cards_covered(state.table):
                raise ValueError("Cannot add new attack card until defender covers current cards.")
            if len(state.table) >= state.round_attack_limit:
                raise ValueError("Attack limit reached for this round.")
            ranks = _ranks_on_table(state.table)
            if card_rank(card) not in ranks:
                raise ValueError("Attack card must match rank already on table.")
    state.hands[state.attacker].remove(card)
    state.table.append((card, None))
    state.seen_cards.add(card)
    if state.defender_taking:
        state.post_take_additions_remaining = max(0, state.post_take_additions_remaining - 1)
        if (
            state.post_take_additions_remaining == 0
            or len(state.table) >= MAX_ATTACK_CARDS
            or not _attacker_has_matching_rank(state)
        ):
            _finalize_take(state)
        return
    state.phase = "defense"


def _defend_card(state: DurakState, card: int) -> None:
    if card not in state.hands[state.defender]:
        raise ValueError("Defender does not hold this card.")
    uncovered = _uncovered_attack_cards(state.table)
    if not uncovered:
        raise ValueError("No cards to defend against.")
    cover_index = None
    for idx, attack_card in uncovered:
        if _can_cover(card, attack_card, state.trump_suit):
            cover_index = idx
            break
    if cover_index is None:
        raise ValueError("Card cannot cover any attack card.")
    state.hands[state.defender].remove(card)
    attack_card, _ = state.table[cover_index]
    state.table[cover_index] = (attack_card, card)
    state.seen_cards.add(card)
    if _all_cards_covered(state.table):
        if (
            len(state.table) < state.round_attack_limit
            and state.hands[state.attacker]
            and state.hands[state.defender]
            and _attacker_has_matching_rank(state)
        ):
            state.phase = "attack"
        else:
            _finish_round_with_defense(state)
    else:
        state.phase = "defense"


def _defender_takes(state: DurakState) -> None:
    if state.defender_taking:
        return
    state.defender_taking = True
    remaining_slots = max(0, MAX_ATTACK_CARDS - len(state.table))
    state.post_take_additions_remaining = min(
        remaining_slots, len(state.hands[state.defender])
    )
    state.phase = "attack"
    if (
        state.post_take_additions_remaining == 0
        or len(state.table) >= MAX_ATTACK_CARDS
        or not _attacker_has_matching_rank(state)
    ):
        _finalize_take(state)
    state.last_round_winner = None


def _finish_round_with_defense(state: DurakState) -> None:
    for attack_card, defense_card in state.table:
        state.discard.append(attack_card)
        state.seen_cards.add(attack_card)
        if defense_card is not None:
            state.discard.append(defense_card)
            state.seen_cards.add(defense_card)
    state.table.clear()
    old_attacker = state.attacker
    old_defender = state.defender
    _refill_hands(state, old_attacker, old_defender)
    state.attacker = old_defender
    state.defender = old_attacker
    state.phase = "attack"
    state.round_attack_limit = _compute_attack_limit(state.hands[state.defender])
    state.hands[state.attacker].sort()
    state.hands[state.defender].sort()
    state.last_round_winner = state.attacker
    state.defender_taking = False
    state.post_take_additions_remaining = 0


def _finalize_take(state: DurakState) -> None:
    for attack_card, defense_card in state.table:
        state.hands[state.defender].append(attack_card)
        if defense_card is not None:
            state.hands[state.defender].append(defense_card)
    state.hands[state.defender].sort()
    state.table.clear()
    _refill_hands(state, state.attacker, state.defender)
    state.phase = "attack"
    state.round_attack_limit = _compute_attack_limit(state.hands[state.defender])
    state.defender_taking = False
    state.post_take_additions_remaining = 0
    state.last_round_winner = None


def _refill_hands(state: DurakState, first: int, second: int) -> None:
    for pid in (first, second):
        while len(state.hands[pid]) < 6 and state.talon:
            card = state.talon.pop(0)
            state.hands[pid].append(card)
        state.hands[pid].sort()


def _attacker_has_matching_rank(state: DurakState) -> bool:
    ranks = _ranks_on_table(state.table)
    for card in state.hands[state.attacker]:
        if card_rank(card) in ranks:
            return True
    return False


def _uncovered_attack_cards(table: Sequence[Tuple[int, Optional[int]]]) -> List[Tuple[int, int]]:
    return [(idx, atk) for idx, (atk, defense) in enumerate(table) if defense is None]


def _all_cards_covered(table: Sequence[Tuple[int, Optional[int]]]) -> bool:
    return bool(table) and all(defense is not None for _, defense in table)


def _can_cover(defense_card: int, attack_card: int, trump_suit: int) -> bool:
    defense_suit = card_suit(defense_card)
    attack_suit = card_suit(attack_card)
    if defense_suit == attack_suit:
        return card_rank(defense_card) > card_rank(attack_card)
    if defense_suit == trump_suit and attack_suit != trump_suit:
        return True
    return False


def _ranks_on_table(table: Sequence[Tuple[int, Optional[int]]]) -> set[int]:
    ranks = {card_rank(atk) for atk, _ in table}
    ranks.update(card_rank(defense) for _, defense in table if defense is not None)
    return ranks


def _check_terminal(state: DurakState) -> None:
    if state.terminal:
        return
    if state.table:
        return
    if state.talon:
        return
    hand_sizes = [len(hand) for hand in state.hands]
    zero_players = [pid for pid, size in enumerate(hand_sizes) if size == 0]
    if len(zero_players) == 1:
        state.terminal = True
        state.winner = zero_players[0]
    elif len(zero_players) == 2:
        winner = state.last_round_winner if state.last_round_winner is not None else state.attacker
        state.terminal = True
        state.winner = winner


def current_player(state: DurakState) -> int:
    return state.attacker if state.phase == "attack" else state.defender


def discard_seen(state: DurakState) -> np.ndarray:
    seen = np.zeros(NUM_CARDS, dtype=np.bool_)
    for card in state.discard:
        seen[card] = True
    for attack_card, defense_card in state.table:
        seen[attack_card] = True
        if defense_card is not None:
            seen[defense_card] = True
    for card in state.seen_cards:
        seen[card] = True
    return seen


class DurakEnv:
    """Gym-style wrapper around the Durak environment."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.state: Optional[DurakState] = None

    def reset(self) -> Dict[str, np.ndarray]:
        self.state = initial_state(self.rng)
        return self._build_observation()

    def step(self, action: int) -> Tuple[Optional[Dict[str, np.ndarray]], float, bool, Dict]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if action is None:
            raise ValueError("Action cannot be None.")
        self.state, done, winner = apply_action(self.state, action)
        if done:
            reward = 1.0 if winner == 0 else -1.0
            obs = None
        else:
            reward = 0.0
            obs = self._build_observation()
        return obs, reward, done, {"winner": winner}

    def close(self) -> None:
        self.state = None

    def _build_observation(self) -> Dict[str, np.ndarray]:
        assert self.state is not None
        state = self.state
        player = current_player(state)
        opponent = 1 - player
        position = f"player_{player}"

        state_vector = _encode_state_vector(state, player)
        legal_mask = legal_actions(state)
        legal_ids = np.flatnonzero(legal_mask)
        if legal_ids.size == 0:
            raise RuntimeError("State has no legal actions available.")
        action_embeddings = np.stack([_action_to_vector(a) for a in legal_ids], axis=0)

        assert state_vector.shape[0] == STATE_VECTOR_LENGTH

        return {
            "position": position,
            "state": state_vector.astype(np.float32),
            "legal_actions": legal_ids.astype(np.int64),
            "action_embeddings": action_embeddings.astype(np.float32),
        }


def _encode_state_vector(state: DurakState, player: int) -> np.ndarray:
    opponent = 1 - player
    attacker = state.attacker
    defender = state.defender

    player_hand = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in state.hands[player]:
        player_hand[card] = 1.0

    table_attack = np.zeros(NUM_CARDS, dtype=np.float32)
    table_defense = np.zeros(NUM_CARDS, dtype=np.float32)
    for atk, defense in state.table:
        table_attack[atk] = 1.0
        if defense is not None:
            table_defense[defense] = 1.0

    seen = discard_seen(state).astype(np.float32)

    trump_suit_vec = np.zeros(len(SUITS), dtype=np.float32)
    trump_suit_vec[state.trump_suit] = 1.0

    trump_rank_vec = np.zeros(len(RANKS), dtype=np.float32)
    trump_rank_vec[card_rank(state.trump_card)] = 1.0

    last_winner_vec = np.zeros(3, dtype=np.float32)
    if state.last_round_winner is None:
        last_winner_vec[0] = 1.0
    elif state.last_round_winner == 0:
        last_winner_vec[1] = 1.0
    else:
        last_winner_vec[2] = 1.0

    current_player_vec = np.zeros(2, dtype=np.float32)
    current_player_vec[player] = 1.0

    attacker_vec = np.zeros(2, dtype=np.float32)
    attacker_vec[attacker] = 1.0

    defender_vec = np.zeros(2, dtype=np.float32)
    defender_vec[defender] = 1.0

    counts = np.array([
        len(state.hands[player]) / NUM_CARDS,
        len(state.hands[opponent]) / NUM_CARDS,
        len(state.hands[attacker]) / NUM_CARDS,
        len(state.hands[defender]) / NUM_CARDS,
        len(state.talon) / NUM_CARDS,
        state.round_attack_limit / MAX_ATTACK_CARDS,
        float(state.defender_taking),
        state.post_take_additions_remaining / MAX_ATTACK_CARDS,
        1.0 if state.phase == "attack" else 0.0,
        1.0 if player == attacker else 0.0,
        len(state.table) / MAX_ATTACK_CARDS,
        1.0 if not state.talon else 0.0,
    ], dtype=np.float32)

    feature_list = [
        player_hand,
        table_attack,
        table_defense,
        seen,
        trump_suit_vec,
        trump_rank_vec,
        last_winner_vec,
        current_player_vec,
        attacker_vec,
        defender_vec,
        counts,
    ]
    return np.concatenate(feature_list, axis=0)


def _action_to_vector(action: int) -> np.ndarray:
    vec = np.zeros(ACTION_VECTOR_LENGTH, dtype=np.float32)
    if 0 <= action < NUM_CARDS:
        vec[action] = 1.0
    elif action in (ACTION_END_ATTACK, ACTION_TAKE_CARDS):
        vec[action] = 1.0
    else:
        raise ValueError(f"Invalid action id: {action}")
    return vec


def create_env(seed: Optional[int] = None) -> DurakEnv:
    return DurakEnv(seed=seed)


def build_observation(state: DurakState, player: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Construct the model observation for an arbitrary ``DurakState``.

    This mirrors :meth:`DurakEnv._build_observation` but operates on a provided
    state object so that tooling (such as the live-play helper) can evaluate
    positions without instantiating a full environment.
    """

    player_id = current_player(state) if player is None else player
    state_vector = _encode_state_vector(state, player_id)
    legal_mask = legal_actions(state)
    legal_ids = np.flatnonzero(legal_mask)
    if legal_ids.size == 0:
        raise RuntimeError("State has no legal actions available.")
    action_embeddings = np.stack([_action_to_vector(a) for a in legal_ids], axis=0)

    return {
        "position": f"player_{player_id}",
        "state": state_vector.astype(np.float32),
        "legal_actions": legal_ids.astype(np.int64),
        "action_embeddings": action_embeddings.astype(np.float32),
    }
