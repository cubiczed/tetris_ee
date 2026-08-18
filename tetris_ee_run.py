"""
IB CS Extended Essay — Extended Experiment
Tetris Reward Shaping with DQN
Run on Apple Silicon with MPS acceleration.
15,000 episodes per strategy. Send results/ folder back when complete.
"""

import random, math, collections, time, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Device selection (MPS > CPU) ────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS (GPU acceleration) ✓")
else:
    DEVICE = torch.device("cpu")
    print("MPS not available — using CPU")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Output folder ────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "tetris_results")
os.makedirs(OUT, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# TETRIS ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════════
BOARD_W, BOARD_H = 10, 20

TETROMINOES = {
    "I": [[(0,0),(0,1),(0,2),(0,3)], [(0,0),(1,0),(2,0),(3,0)]],
    "O": [[(0,0),(0,1),(1,0),(1,1)]],
    "T": [[(0,1),(1,0),(1,1),(1,2)], [(0,0),(1,0),(2,0),(1,1)],
          [(0,0),(0,1),(0,2),(1,1)], [(0,1),(1,1),(2,1),(1,0)]],
    "S": [[(0,1),(0,2),(1,0),(1,1)], [(0,0),(1,0),(1,1),(2,1)]],
    "Z": [[(0,0),(0,1),(1,1),(1,2)], [(0,1),(1,0),(1,1),(2,0)]],
    "J": [[(0,0),(1,0),(1,1),(1,2)], [(0,0),(0,1),(1,0),(2,0)],
          [(0,0),(0,1),(0,2),(1,2)], [(0,1),(1,1),(2,0),(2,1)]],
    "L": [[(0,2),(1,0),(1,1),(1,2)], [(0,0),(0,1),(1,0),(2,0)],
          [(0,0),(0,1),(0,2),(1,0)], [(0,0),(0,1),(1,1),(2,1)]],
}
PIECE_NAMES = list(TETROMINOES.keys())
LINE_REWARDS = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}

def _col_heights(board):
    heights = []
    for c in range(BOARD_W):
        h = 0
        for r in range(BOARD_H):
            if board[r][c]:
                h = BOARD_H - r; break
        heights.append(h)
    return heights

def _count_holes(board):
    holes = 0
    for c in range(BOARD_W):
        found = False
        for r in range(BOARD_H):
            if board[r][c]: found = True
            elif found: holes += 1
    return holes

def _bumpiness(heights):
    return sum(abs(heights[i]-heights[i+1]) for i in range(len(heights)-1))


class TetrisEnv:
    def reset(self):
        self.board = [[0]*BOARD_W for _ in range(BOARD_H)]
        self.piece = self._new_piece()
        self.game_over = False
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        return self._state()

    def _new_piece(self):
        n = random.choice(PIECE_NAMES)
        return {"name": n, "rotations": TETROMINOES[n]}

    def _get_placements(self):
        placements = []
        for ri, shape in enumerate(self.piece["rotations"]):
            max_c = max(c for r,c in shape)
            min_c = min(c for r,c in shape)
            for col in range(-min_c, BOARD_W - max_c):
                dropped = self._drop(shape, col)
                if dropped is not None:
                    placements.append((ri, col, dropped))
        return placements

    def _drop(self, shape, col_offset):
        board_copy = [row[:] for row in self.board]
        max_row = -1
        for row_offset in range(BOARD_H + 1):
            valid = True
            for (pr, pc) in shape:
                r = row_offset + pr; c = col_offset + pc
                if r >= BOARD_H or c < 0 or c >= BOARD_W: valid = False; break
                if r >= 0 and board_copy[r][c]: valid = False; break
            if not valid: break
            max_row = row_offset
        if max_row < 0: return None
        for (pr, pc) in shape:
            r = max_row + pr; c = col_offset + pc
            if r < 0: return None
            board_copy[r][c] = 1
        return board_copy

    def _clear_lines(self, board):
        new_board = [row for row in board if not all(row)]
        cleared = BOARD_H - len(new_board)
        return [[0]*BOARD_W]*cleared + new_board, cleared

    def step(self, action_idx):
        placements = self._get_placements()
        if not placements:
            self.game_over = True
            return self._state(), 0, self.board, True, {}
        action_idx = action_idx % len(placements)
        _, _, new_board = placements[action_idx]
        new_board, lines = self._clear_lines(new_board)
        self.board = new_board
        self.lines_cleared_total += lines
        self.pieces_placed += 1
        for r in range(2):
            if any(self.board[r]): self.game_over = True; break
        self.piece = self._new_piece()
        if not self._get_placements(): self.game_over = True
        h = _col_heights(self.board)
        return self._state(), lines, self.board, self.game_over, {
            "lines": lines, "heights": h,
            "holes": _count_holes(self.board), "bumpiness": _bumpiness(h)
        }

    def _state(self):
        h = _col_heights(self.board)
        piece_oh = [0.0] * 7
        piece_oh[PIECE_NAMES.index(self.piece["name"])] = 1.0
        return np.array([sum(h), _count_holes(self.board), _bumpiness(h), max(h)]
                        + piece_oh, dtype=np.float32)

    def n_actions(self):
        return len(self._get_placements())

    def board_stats(self):
        h = _col_heights(self.board)
        return {"agg_h": sum(h), "max_h": max(h),
                "holes": _count_holes(self.board), "bump": _bumpiness(h)}


# ════════════════════════════════════════════════════════════════════════════
# REWARD STRATEGIES
# ════════════════════════════════════════════════════════════════════════════
def r_sparse(lines, board, info, pieces):
    return LINE_REWARDS.get(lines, 0)

def r_height(lines, board, info, pieces):
    return LINE_REWARDS.get(lines, 0) - 0.51 * sum(info["heights"])

def r_dense(lines, board, info, pieces):
    h = info["heights"]
    return (LINE_REWARDS.get(lines, 0)
            - 0.51 * sum(h)
            - 0.36 * info["holes"]
            - 0.18 * _bumpiness(h))

def r_survival(lines, board, info, pieces):
    return LINE_REWARDS.get(lines, 0) + 1.0

STRATEGIES = {
    "Sparse":           r_sparse,
    "Height Penalty":   r_height,
    "Dense Heuristic":  r_dense,
    "Survival Bonus":   r_survival,
}
COLORS = {
    "Sparse":          "#E63946",
    "Height Penalty":  "#457B9D",
    "Dense Heuristic": "#2A9D8F",
    "Survival Bonus":  "#E9C46A",
}

# ════════════════════════════════════════════════════════════════════════════
# DQN
# ════════════════════════════════════════════════════════════════════════════
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 34),
        )
    def forward(self, x): return self.net(x)

Exp = collections.namedtuple("Exp", ["s","a","r","s2","done"])

class ReplayBuffer:
    def __init__(self, cap=20000): self.buf = collections.deque(maxlen=cap)
    def push(self, *a): self.buf.append(Exp(*a))
    def sample(self, n): return random.sample(self.buf, n)
    def __len__(self): return len(self.buf)

# ── Hyperparameters ──────────────────────────────────────────────────────────
GAMMA         = 0.99
LR            = 0.0005        # slightly lower for stability over longer runs
BATCH         = 128           # larger batch for smoother gradients
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 800           # episodes to reach EPS_END
TARGET_UPDATE = 20
BUFFER_CAP    = 20000
MIN_BUF       = 1000
NUM_EPISODES  = 15000         # ← the meaningful number


class DQNAgent:
    # max_actions=34: analytically verified upper bound (T/L/J pieces each
    # have exactly 34 legal placements across 4 rotations × 8–9 columns).
    def __init__(self):
        self.q   = QNet().to(DEVICE)
        self.tgt = QNet().to(DEVICE)
        self.tgt.load_state_dict(self.q.state_dict()); self.tgt.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.buf = ReplayBuffer(BUFFER_CAP)
        self.eps = EPS_START
        self.ep  = 0

    def act(self, s, n):
        if n == 0: return 0
        if random.random() < self.eps: return random.randrange(n)
        st = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): return self.q(st)[0, :n].argmax().item()

    def push(self, *a): self.buf.push(*a)

    def learn(self):
        if len(self.buf) < MIN_BUF: return
        batch = self.buf.sample(BATCH)
        s  = torch.FloatTensor(np.array([e.s  for e in batch])).to(DEVICE)
        a  = torch.LongTensor ([e.a  for e in batch]).to(DEVICE)
        r  = torch.FloatTensor([e.r  for e in batch]).to(DEVICE)
        s2 = torch.FloatTensor(np.array([e.s2 for e in batch])).to(DEVICE)
        d  = torch.FloatTensor([e.done for e in batch]).to(DEVICE)
        q_cur  = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.tgt(s2).max(1)[0]
        target = r + GAMMA * q_next * (1 - d)
        loss   = nn.MSELoss()(q_cur, target)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10)
        self.opt.step()

    def update_eps(self):
        self.ep += 1
        self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(-self.ep / EPS_DECAY)

    def sync_target(self):
        self.tgt.load_state_dict(self.q.state_dict())


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════
def train(name, reward_fn, n_eps=NUM_EPISODES, seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = DQNAgent(); env = TetrisEnv()

    hist = {k: [] for k in [
        "lines", "pieces", "h_agg", "holes", "bump", "eps"
    ]}

    print(f"\n{'═'*55}")
    print(f"  {name}  ({n_eps} episodes, device={DEVICE})")
    print(f"{'═'*55}")
    t0 = time.time()

    for ep in range(1, n_eps + 1):
        s = env.reset()
        ep_r = ep_h = ep_o = ep_b = steps = 0

        while not env.game_over:
            na = env.n_actions()
            a  = agent.act(s, na)
            s2, lines, board, done, info = env.step(a)
            shaped = reward_fn(lines, board, info, env.pieces_placed)
            agent.push(s, a, shaped, s2, float(done))
            agent.learn()
            bs = env.board_stats()
            ep_h += bs["agg_h"]; ep_o += bs["holes"]
            ep_b += bs["bump"]; steps += 1
            ep_r += shaped; s = s2

        agent.update_eps()
        if ep % TARGET_UPDATE == 0: agent.sync_target()

        ss = max(steps, 1)
        hist["lines"].append(env.lines_cleared_total)
        hist["pieces"].append(env.pieces_placed)
        hist["h_agg"].append(ep_h / ss)
        hist["holes"].append(ep_o / ss)
        hist["bump"].append(ep_b / ss)
        hist["eps"].append(agent.eps)

        if ep % 50 == 0 or ep == 1:
            rec = hist["lines"][-50:]
            print(f"  ep {ep:4d}  lines(ma50)={np.mean(rec):5.2f}"
                  f"  pieces={np.mean(hist['pieces'][-50:]):5.1f}"
                  f"  eps={agent.eps:.3f}"
                  f"  buf={len(agent.buf):5d}"
                  f"  {time.time()-t0:.0f}s")

    print(f"  Finished in {time.time()-t0:.1f}s")
    return hist


# ════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════
def smooth(arr, w=60):
    out = []
    for i in range(len(arr)):
        lo = max(0, i-w+1)
        out.append(np.mean(arr[lo:i+1]))
    return np.array(out)

def dark_ax(ax):
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="#8B949E")
    ax.spines[["top","right"]].set_visible(False)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363D")

def make_figs(all_hist):
    eps = list(range(1, NUM_EPISODES + 1))
    # ── Fig 1: individual learning curves ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0D1117")
    fig.suptitle("Figure 1 – Learning Curves: Lines Cleared per Episode",
                 color="white", fontsize=14, fontweight="bold")
    for i, (name, hist) in enumerate(all_hist.items()):
        ax = axes[i//2][i%2]; dark_ax(ax)
        raw = hist["lines"]; ma = smooth(raw)
        col = COLORS[name]
        ax.plot(eps, raw, color=col, alpha=0.15, lw=0.7)
        ax.plot(eps, ma,  color=col, lw=2.0, label="MA-60")
        ax.axhline(np.mean(raw[-200:]), color=col, ls="--", alpha=0.5, lw=1)
        ax.text(NUM_EPISODES*0.98, np.mean(raw[-200:])+0.3,
                f"μ={np.mean(raw[-200:]):.2f}", ha="right", va="bottom",
                color=col, fontsize=8)
        ax.set_title(name, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", color="#8B949E")
        ax.set_ylabel("Lines Cleared", color="#8B949E")
        ax.legend(fontsize=8, facecolor="#161B22", labelcolor="white")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_learning_curves.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # ── Fig 2: all strategies overlaid ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0D1117")
    dark_ax(ax)
    for name, hist in all_hist.items():
        ma = smooth(hist["lines"])
        ax.plot(eps, ma, color=COLORS[name], lw=2.2, label=name)
    ax.set_title("Figure 2 – All Strategies: Lines Cleared (60-Ep MA)",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode", color="#8B949E"); ax.set_ylabel("Lines Cleared", color="#8B949E")
    ax.legend(facecolor="#161B22", labelcolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_comparative.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # ── Fig 3: board height & holes ────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0D1117")
    fig.suptitle("Figure 3 – Emergent Board Behaviour",
                 color="white", fontsize=13, fontweight="bold")
    for ax in [a1, a2]: dark_ax(ax)
    for name, hist in all_hist.items():
        col = COLORS[name]
        a1.plot(eps, smooth(hist["h_agg"]),   color=col, lw=2, label=name)
        a2.plot(eps, smooth(hist["holes"]),    color=col, lw=2, label=name)
    a1.set_title("Avg Aggregate Board Height", color="white", fontsize=11)
    a1.set_xlabel("Episode", color="#8B949E"); a1.set_ylabel("Height (cells)", color="#8B949E")
    a2.set_title("Avg Holes per Step",         color="white", fontsize=11)
    a2.set_xlabel("Episode", color="#8B949E"); a2.set_ylabel("Holes", color="#8B949E")
    for ax in [a1, a2]: ax.legend(facecolor="#161B22", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_emergent.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # ── Fig 4: bumpiness ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5), facecolor="#0D1117")
    dark_ax(ax)
    for name, hist in all_hist.items():
        ax.plot(eps, smooth(hist["bump"]), color=COLORS[name], lw=2, label=name)
    ax.set_title("Figure 4 – Avg Board Bumpiness (60-Ep MA)",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode", color="#8B949E"); ax.set_ylabel("Bumpiness", color="#8B949E")
    ax.legend(facecolor="#161B22", labelcolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_bumpiness.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # ── Fig 5: summary bar chart ───────────────────────────────────────────
    N = 200   # final N episodes for steady-state stats
    names  = list(all_hist.keys())
    colors = [COLORS[n] for n in names]
    x      = np.arange(len(names))
    metrics = {n: {
        "lines":  np.mean(all_hist[n]["lines"][-N:]),
        "holes":  np.mean(all_hist[n]["holes"][-N:]),
        "h_agg":  np.mean(all_hist[n]["h_agg"][-N:]),
        "pieces": np.mean(all_hist[n]["pieces"][-N:]),
    } for n in names}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0D1117")
    fig.suptitle("Figure 5 – Steady-State Metrics (Final 200 Episodes)",
                 color="white", fontsize=13, fontweight="bold")
    for ax, key, label in zip(axes,
            ["lines", "holes", "h_agg"],
            ["Mean Lines Cleared", "Mean Holes/Step", "Mean Agg Height/Step"]):
        dark_ax(ax)
        vals = [metrics[n][key] for n in names]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="#30363D")
        ax.set_title(label, color="white", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ","\n") for n in names],
                           color="#8B949E", fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_summary.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print("\nAll figures saved to", OUT)
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════
def print_summary(all_hist, metrics):
    N = 200
    print(f"\n{'═'*75}")
    print("  FINAL RESULTS SUMMARY  (mean ± std, last 200 episodes)")
    print(f"{'═'*75}")
    print(f"{'Strategy':<20} {'Lines':>8} {'Pieces':>8} {'Holes':>8} {'Height':>9} {'Bumpiness':>11}")
    print("-"*75)
    for name, hist in all_hist.items():
        lm = np.mean(hist["lines"][-N:]);  ls = np.std(hist["lines"][-N:])
        pm = np.mean(hist["pieces"][-N:]); ps = np.std(hist["pieces"][-N:])
        om = np.mean(hist["holes"][-N:]);  os_ = np.std(hist["holes"][-N:])
        hm = np.mean(hist["h_agg"][-N:]);  hs = np.std(hist["h_agg"][-N:])
        bm = np.mean(hist["bump"][-N:]);   bs = np.std(hist["bump"][-N:])
        print(f"{name:<20} {lm:5.2f}±{ls:.2f}  {pm:5.1f}±{ps:.1f}  "
              f"{om:5.2f}±{os_:.2f}  {hm:6.2f}±{hs:.2f}  {bm:6.2f}±{bs:.2f}")
    print(f"{'═'*75}")

    print("\n  CONVERGENCE: episode at which 60-ep MA first exceeds"
          " 60% of final-200-ep mean")
    print("-"*50)
    for name, hist in all_hist.items():
        lines = hist["lines"]
        target = 0.6 * np.mean(lines[-N:])
        ma = smooth(lines, 60)
        cross = next((i for i, v in enumerate(ma) if v >= target), NUM_EPISODES)
        print(f"  {name:<22}: episode {cross}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═"*55)
    print(f"  IB CS EE — Tetris Reward Shaping Experiment")
    print(f"  Episodes: {NUM_EPISODES} × 4 strategies")
    print(f"  Device:   {DEVICE}")
    print(f"  Output:   {OUT}/")
    print("═"*55)

    all_hist = {}
    for name, fn in STRATEGIES.items():
        all_hist[name] = train(name, fn)

    # Save raw data
    safe = {k: {kk: [float(x) for x in v] for kk, v in h.items()}
            for k, h in all_hist.items()}
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(safe, f, indent=2)
    print(f"\nRaw data saved → {OUT}/results.json")

    metrics = make_figs(all_hist)
    print_summary(all_hist, metrics)
    print(f"\nDone. Send the '{OUT}/' folder back.\n")
