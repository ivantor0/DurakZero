"""Live Durak tracker that mirrors real games from network events."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from douzero.dmc.models import Model
from douzero.env.env import (
    ACTION_END_ATTACK,
    ACTION_TAKE_CARDS,
    MAX_ATTACK_CARDS,
    NUM_CARDS,
    DurakState,
    build_observation,
    card_suit,
)

from .cards import card_id_to_symbol, symbol_to_card_id


@dataclass
class LiveGameEvent:
    """Container for a decoded network message."""

    tag: str
    payload: Dict


class LiveDurakTracker:
    """Track a remote Durak game and produce DurakZero recommendations."""

    def __init__(
        self,
        model: Model,
        device: torch.device,
        player_id: Optional[int] = None,
        verbose: bool = True,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.player_id = player_id
        self.verbose = verbose
        self.flags = SimpleNamespace(exp_epsilon=0.0)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._log_handle: Optional[io.TextIOWrapper] = None
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if hasattr(self, "_log_handle") and self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self.game_id: Optional[int] = None
        self.game_active = False
        self.table: List[Tuple[int, Optional[int]]] = []
        self.my_hand: List[int] = []
        self.discard_cards: List[int] = []
        self._discard_cache: set[int] = set()
        self.talon_count: int = 0
        self.trump_card: Optional[int] = None
        self.trump_suit: Optional[int] = None
        self.attacker: Optional[int] = None
        self.defender: Optional[int] = None
        self.defender_taking: bool = False
        self.pending_take_cards: List[int] = []
        self.last_round_winner: Optional[int] = None
        self.last_mode: Optional[Dict] = None
        self.seen_cards: set[int] = set()
        self.pending_cleanup: Optional[str] = None
        self._last_recommendation_key: Optional[Tuple] = None
        self._last_summary_timestamp: float = 0.0
        self._role_prompted: bool = False
        self._initial_role: Optional[str] = None
        self._state_dirty: bool = True
        self._force_recommend: bool = False
        self._round_attack_limit: Optional[int] = None

    def process_raw_line(self, raw_line: str) -> None:
        event = self._parse_line(raw_line)
        if event is None:
            return
        handler = getattr(self, f"_handle_{event.tag}", None)
        if handler is not None:
            handler(event.payload)
        self._log_raw_line(raw_line)
        self._update_phase_after_event()
        force = self._force_recommend
        self._force_recommend = False
        self._maybe_recommend(force=True)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_line(self, raw_line: str) -> Optional[LiveGameEvent]:
        raw_line = raw_line.strip()
        if not raw_line:
            return None
        if raw_line.startswith("//"):
            return None
        if "{" in raw_line:
            tag, rest = raw_line.split("{", 1)
            json_payload = "{" + rest
            try:
                payload = json.loads(json_payload)
            except json.JSONDecodeError:
                return None
            return LiveGameEvent(tag.strip(), payload)
        return LiveGameEvent(raw_line, {})

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_game(self, payload: Dict) -> None:
        incoming_id = payload.get("id")
        incoming_position = payload.get("position")

        if incoming_id is None and self.game_id is not None:
            self.game_active = True
            if self.player_id is None and incoming_position is not None:
                self.player_id = incoming_position
            if self.verbose:
                print(
                    f"\n>>> Joined game None as player {self.player_id}"
                    f" (continuing {self.game_id})"
                )
            self._log("# Received null game id; continuing current game context")
            return

        self.reset()
        self.game_active = True
        self.game_id = incoming_id
        if self.player_id is None:
            self.player_id = incoming_position if incoming_position is not None else 0
        self._open_log_file()
        if self.verbose:
            print(f"\n>>> Joined game {self.game_id} as player {self.player_id}")
        self._log(f"# Joined game {self.game_id} as player {self.player_id}")

    def _handle_game_start(self, payload: Dict) -> None:  # noqa: D401 - alias
        self._handle_game(payload)

    def _handle_game_reset(self, payload: Dict) -> None:
        if self.verbose:
            print("Game reset.")
        self.reset()

    def _handle_game_over(self, payload: Dict) -> None:
        if self.verbose:
            winner_list = payload.get("players", [])
            print(f"Game over. Winner ids: {winner_list}")
        self.game_active = False
        self.pending_cleanup = None
        self._log(f"# Game over payload: {payload}")

    def _handle_win(self, payload: Dict) -> None:
        if self.verbose:
            print(f"Win update: {payload}")

    def _handle_hand(self, payload: Dict) -> None:
        cards = [symbol_to_card_id(card) for card in payload.get("cards", [])]
        self.my_hand = sorted(card for card in cards if card is not None)
        self.seen_cards.update(self.my_hand)
        if self.verbose:
            formatted = " ".join(card_id_to_symbol(card) for card in self.my_hand)
            print(f"Updated hand ({len(self.my_hand)}): {formatted}")
        if self.game_active and not self._role_prompted and self.player_id is not None:
            self._prompt_initial_role()
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_turn(self, payload: Dict) -> None:
        self.talon_count = int(payload.get("deck", self.talon_count))
        trump_symbol = payload.get("trump")
        if trump_symbol:
            trump_id = symbol_to_card_id(trump_symbol)
            if trump_id is not None:
                self.trump_card = trump_id
                self.trump_suit = card_suit(trump_id)
                self.seen_cards.add(trump_id)
            else:
                suit_symbol = trump_symbol[0]
                suit_idx = symbol_to_card_id(f"6{suit_symbol}")
                if suit_idx is not None:
                    self.trump_suit = card_suit(suit_idx)
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_mode(self, payload: Dict) -> None:
        parsed: Dict[int, int] = {}
        for key, value in payload.items():
            try:
                parsed[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
        self.last_mode = parsed
        self._infer_roles_from_mode()

    def _handle_t(self, payload: Dict) -> None:
        card = symbol_to_card_id(payload.get("c", ""))
        if card is None:
            return
        attacker = payload.get("id")
        if attacker is None:
            if card in self.my_hand:
                attacker = self.player_id
            elif self.attacker is not None:
                attacker = self.attacker
            else:
                attacker = 1 - (self.player_id or 0)
        starting_new_round = not self.table
        self.attacker = attacker
        self.defender = 1 - attacker
        if starting_new_round:
            self._round_attack_limit = self._compute_round_attack_limit(self.defender)
        if attacker == self.player_id and card in self.my_hand:
            self.my_hand.remove(card)
        self.table.append((card, None))
        self.seen_cards.add(card)
        self.pending_cleanup = None
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_b(self, payload: Dict) -> None:
        attack_card = symbol_to_card_id(payload.get("c", ""))
        defense_card = symbol_to_card_id(payload.get("b", ""))
        if defense_card is None:
            return
        raw_index = payload.get("id")
        pair_index: Optional[int]
        if raw_index is not None:
            try:
                pair_index = int(raw_index)
            except (TypeError, ValueError):
                pair_index = None
            else:
                if pair_index < 0:
                    pair_index = None
                elif pair_index >= len(self.table):
                    candidate = pair_index - 1
                    if 0 <= candidate < len(self.table):
                        pair_index = candidate
                    else:
                        pair_index = None
        else:
            pair_index = None

        if pair_index is None:
            pair_index = self._find_uncovered_index(attack_card)
        if pair_index is None or pair_index >= len(self.table):
            return
        attack_value, _ = self.table[pair_index]
        if attack_card is not None and attack_card != attack_value:
            pair_index = self._find_uncovered_index(attack_card)
            if pair_index is None:
                return
        self.table[pair_index] = (self.table[pair_index][0], defense_card)
        if self.defender == self.player_id and defense_card in self.my_hand:
            self.my_hand.remove(defense_card)
        self.seen_cards.add(defense_card)
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_take(self, payload: Dict) -> None:
        self.defender_taking = True
        self.pending_take_cards = []
        for attack_card, defense_card in self.table:
            self.pending_take_cards.append(attack_card)
            if defense_card is not None:
                self.pending_take_cards.append(defense_card)
        self.pending_cleanup = "take"
        self._log("[event] defender announced take")
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_done(self, payload: Dict) -> None:
        self.pending_cleanup = "defense"
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_pass(self, payload: Dict) -> None:
        # Optional message from the server when the attacker declines to add more cards.
        self.pending_cleanup = self.pending_cleanup or "defense"
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_end_turn(self, payload: Dict) -> None:
        self._finalize_round()
        next_attacker = payload.get("id")
        if next_attacker is not None:
            self.attacker = next_attacker
            self.defender = 1 - next_attacker
        self._log(f"[event] end_turn -> next attacker: {self.attacker}")
        self._mark_state_dirty()
        self._maybe_recommend(force=True)

    def _handle_order(self, payload: Dict) -> None:  # pragma: no cover - informational
        pass

    def _handle_turn_timeout(self, payload: Dict) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------

    def _find_uncovered_index(self, attack_card: Optional[int]) -> Optional[int]:
        if attack_card is None:
            for idx, (_, defense) in enumerate(self.table):
                if defense is None:
                    return idx
            return None
        for idx, (attack, defense) in enumerate(self.table):
            if attack == attack_card and defense is None:
                return idx
        return None

    def _finalize_round(self) -> None:
        if self.defender_taking:
            if self.defender == self.player_id:
                self.my_hand.extend(self.pending_take_cards)
                self.my_hand.sort()
            self.defender_taking = False
            self.pending_take_cards = []
            self.pending_cleanup = None
            self.table.clear()
            self._round_attack_limit = None
            self._log("[round] defender took the table")
            return

        if self.table:
            for attack_card, defense_card in self.table:
                self.discard_cards.append(attack_card)
                self._discard_cache.add(attack_card)
                self.seen_cards.add(attack_card)
                if defense_card is not None:
                    self.discard_cards.append(defense_card)
                    self._discard_cache.add(defense_card)
                    self.seen_cards.add(defense_card)
        prev_attacker, prev_defender = self.attacker, self.defender
        self.table.clear()
        self.pending_cleanup = None
        self._round_attack_limit = None
        if prev_attacker is not None and prev_defender is not None:
            self.attacker = prev_defender
            self.defender = prev_attacker
            self.last_round_winner = self.attacker
        self._log("[round] cards moved to discard")
        self._mark_state_dirty()

    def _update_phase_after_event(self) -> None:
        if not self.table:
            return
        if any(defense is None for _, defense in self.table) and not self.defender_taking:
            # Defender still owes at least one response.
            return
        # Otherwise it is attacker's decision space.

    # ------------------------------------------------------------------
    # Recommendation logic
    # ------------------------------------------------------------------

    def _maybe_recommend(self, *, force: bool = False) -> None:
        if not self.game_active:
            return
        if self.player_id is None:
            return
        if self.trump_card is None:
            return
        if self.attacker is None or self.defender is None:
            return
        state = self._build_state()
        if state is None:
            return
        player_turn = state.attacker if state.phase == "attack" else state.defender
        if player_turn != self.player_id:
            return
        key = self._recommendation_key(state)
        if not force and key == self._last_recommendation_key:
            if not self._state_dirty:
                return
        self._state_dirty = False
        self._last_recommendation_key = key
        self._emit_recommendation(state)

    def _recommendation_key(self, state: DurakState) -> Tuple:
        table_key = tuple((atk, defn if defn is not None else -1) for atk, defn in state.table)
        return (
            state.phase,
            tuple(state.hands[self.player_id]),
            table_key,
            state.defender_taking,
            len(state.talon),
        )

    def _build_state(self) -> Optional[DurakState]:
        if self.attacker is None or self.defender is None or self.trump_card is None or self.trump_suit is None:
            return None
        table = [(atk, defense) for atk, defense in self.table]
        seen = set(self.seen_cards)
        for atk, defense in table:
            seen.add(atk)
            if defense is not None:
                seen.add(defense)
        seen.update(self.my_hand)
        seen.add(self.trump_card)

        talon = [None] * self.talon_count
        hands: List[List[Optional[int]]] = [[], []]
        hands[self.player_id] = sorted(self.my_hand)
        opponent = 1 - self.player_id
        opponent_count = self._estimate_opponent_count()
        hands[opponent] = [None] * opponent_count

        defender_hand_len = len(hands[self.defender]) if self.defender == opponent else len(self.my_hand)
        if self._round_attack_limit is None:
            round_limit = min(MAX_ATTACK_CARDS, max(1, defender_hand_len))
        else:
            round_limit = self._round_attack_limit
        round_limit = max(len(table), round_limit)
        post_take = 0
        if self.defender_taking:
            post_take = max(0, min(MAX_ATTACK_CARDS - len(table), defender_hand_len))

        state = DurakState(
            hands=hands,  # type: ignore[arg-type]
            talon=talon,  # type: ignore[list-item]
            discard=list(self.discard_cards),
            table=table,
            attacker=self.attacker,
            defender=self.defender,
            phase="defense" if self._defender_needs_to_act() else "attack",
            trump_suit=self.trump_suit,
            trump_card=self.trump_card,
            seen_cards=seen,
            round_attack_limit=round_limit,
            terminal=False,
            winner=None,
            last_round_winner=self.last_round_winner,
            defender_taking=self.defender_taking,
            post_take_additions_remaining=post_take,
        )
        return state

    def _defender_needs_to_act(self) -> bool:
        if self.defender_taking:
            return False
        return any(defense is None for _, defense in self.table)

    def _compute_round_attack_limit(self, defender: int) -> int:
        if defender == self.player_id:
            defender_hand_len = len(self.my_hand)
        else:
            defender_hand_len = self._estimate_opponent_count()
        return min(MAX_ATTACK_CARDS, max(1, defender_hand_len))

    def _estimate_opponent_count(self) -> int:
        discard_total = len(self._discard_cache)
        table_cards = sum(1 + (defense is not None) for _, defense in self.table)
        remaining = NUM_CARDS - discard_total - self.talon_count - len(self.my_hand) - table_cards
        if self.defender_taking and self.defender == 1 - self.player_id:
            remaining += len(self.pending_take_cards)
        return max(0, remaining)

    def _emit_recommendation(self, state: DurakState) -> None:
        obs = build_observation(state, player=self.player_id)
        state_tensor = torch.from_numpy(obs["state"]).to(self.device)
        action_embeddings = torch.from_numpy(obs["action_embeddings"]).to(self.device)
        legal_actions = obs["legal_actions"]
        position = obs["position"]

        with torch.no_grad():
            output = self.model.act(position, state_tensor, action_embeddings, flags=self.flags)
        values = output["values"].cpu().numpy()
        action_index = output["action_index"]
        suggestions = list(zip(legal_actions, values))
        suggestions.sort(key=lambda item: item[1], reverse=True)

        if not suggestions:
            self._log("[recommendation] no legal actions available")
            return

        best_action_id, best_value = suggestions[0]
        best_label = self._action_to_text(best_action_id)

        if not self.verbose:
            self._log(
                f"[recommendation] best={best_label} ({best_value:.3f}), "
                f"legal={len(legal_actions)}"
            )
            return

        now = time.time()
        self._clear_terminal()
        self._print_state_summary(state)
        self._last_summary_timestamp = now

        print(f"\n>>> BEST MOVE: {best_label} <<<")
        print(f"Estimated value: {best_value: .3f}")
        print("\nOther options:")
        for idx, (action_id, value) in enumerate(suggestions[:5], start=1):
            label = self._action_to_text(action_id)
            marker = "*" if action_id == legal_actions[action_index] else " "
            print(f"  {marker} #{idx}: {label:<12s} | Q = {value: .3f}")
        print("-")
        self._log(
            "[recommendation] "
            + ", ".join(
                f"{self._action_to_text(a)}={v:.3f}" for a, v in suggestions[:5]
            )
        )

    def _print_state_summary(self, state: DurakState) -> None:
        trump_symbol = card_id_to_symbol(self.trump_card) if self.trump_card is not None else "?"
        table_strings = []
        for attack, defense in self.table:
            attack_symbol = card_id_to_symbol(attack)
            if defense is None:
                table_strings.append(f"{attack_symbol}")
            else:
                table_strings.append(f"{attack_symbol}/{card_id_to_symbol(defense)}")
        table_repr = " ".join(table_strings) if table_strings else "(empty)"
        hand_repr = " ".join(card_id_to_symbol(card) for card in sorted(self.my_hand))
        print("""------------------------------------------------------------""")
        print(
            f"Phase: {state.phase} | Attacker: P{self.attacker} | Defender: P{self.defender}"
        )
        print(f"Table: {table_repr}")
        print(
            f"Talon remaining: {self.talon_count} | Discarded: {len(self.discard_cards)} | Trump: {trump_symbol}"
        )
        print(f"Your hand ({len(self.my_hand)}): {hand_repr}")

    def _action_to_text(self, action_id: int) -> str:
        if action_id == ACTION_END_ATTACK:
            return "End attack"
        if action_id == ACTION_TAKE_CARDS:
            return "Take cards"
        return card_id_to_symbol(action_id)

    # ------------------------------------------------------------------
    # Logging and utilities
    # ------------------------------------------------------------------

    def _open_log_file(self) -> None:
        if self.log_dir is None or self.game_id is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{self.game_id}.log"
        if self._log_handle is not None:
            self._log_handle.close()
        self._log_handle = log_path.open("w", encoding="utf-8")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._log_handle.write(f"# Durak game {self.game_id} started at {timestamp}\n")
        self._log_handle.flush()

    def _log_raw_line(self, raw_line: str) -> None:
        if self._log_handle is None:
            return
        self._log_handle.write(raw_line.strip() + "\n")
        self._log_handle.flush()

    def _log(self, message: str) -> None:
        if self._log_handle is None:
            return
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self._log_handle.write(f"[{timestamp}] {message}\n")
        self._log_handle.flush()

    def _prompt_initial_role(self) -> None:
        if self._role_prompted:
            return
        prompt = "Are you a defender [D] or an attacker [A]? Default: attacker > "
        try:
            choice = input(prompt)
        except EOFError:
            choice = ""
        answer = (choice or "").strip().lower()
        defender_selected = answer.startswith("d")
        self._role_prompted = True
        default_player = self.player_id if self.player_id is not None else 0
        if defender_selected:
            self.defender = default_player
            self.attacker = 1 - default_player
            role_text = "defender"
        else:
            self.attacker = default_player
            self.defender = 1 - default_player
            role_text = "attacker"
        self._initial_role = role_text
        if self.verbose:
            print(f"Assuming you start as {role_text}.")
        self._log(f"[role] user selected {role_text}")
        self._mark_state_dirty()

    def _infer_roles_from_mode(self) -> None:
        if not self.last_mode:
            return
        if self.attacker is not None and self.defender is not None:
            return
        attacker_candidates = [pid for pid, value in self.last_mode.items() if value < 8]
        defender_candidates = [pid for pid, value in self.last_mode.items() if value >= 8]
        if attacker_candidates:
            attacker = attacker_candidates[0]
            self.attacker = attacker
            self.defender = 1 - attacker
        elif defender_candidates:
            defender = defender_candidates[0]
            self.defender = defender
            self.attacker = 1 - defender

    def _clear_terminal(self) -> None:
        if not self.verbose:
            return
        print("\033[2J\033[H", end="")

    def _mark_state_dirty(self) -> None:
        self._state_dirty = True
        self._last_recommendation_key = None

    def _schedule_recommendation(self) -> None:
        self._force_recommend = True


__all__ = ["LiveDurakTracker", "LiveGameEvent"]
