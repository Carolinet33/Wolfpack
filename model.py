"""
model.py  --  Quantathon runtime predictor (v2).

featurize(qasm_text) -> dict of circuit features   (pure Python, QASM 2 + 3)
predict(features, threshold) -> predicted seconds (LightGBM on log10 runtime)

Feature families (one pass over the circuit):
  1. Size        : #qubits, #gates, #1q, #2q, #3q+, depth, 2q-depth,
                   gates per layer, active-qubit width per layer (mean / std)
  2. Gate mix    : fraction 2q, parameterised, non-Clifford, rank-4 2q, 3q+
  3. Locality    : two-qubit gate span (mean / max / nearest-neighbour
                   fraction), number of gates crossing each cut
  4. Bond tracker: worst-case log2 bond dimension per cut (an UPPER BOUND on
                   entanglement), turned into a threshold-capped cost estimate
  5. Entanglement entropy (Clifford skeleton): a stabilizer tableau is evolved
                   alongside the parse. Non-Clifford gates are rounded to the
                   nearest Clifford (rotation angles to multiples of pi/2,
                   T gates dropped), so the entropy is EXACT for Clifford
                   circuits and a structural estimate otherwise. Mid-circuit
                   measurements are applied exactly. We record the entropy of
                   every prefix cut at ~16 snapshots and keep the peak.
  6. Measurement : #measurements, #mid-circuit measurements, #classically
                   controlled gates, #if-blocks
  7. Setting     : log2(threshold) + threshold-interaction features
  8. Graph       : interaction graph of 2q gates (distinct pairs, degree)
  9. Sampling    : cost of drawing SHOTS samples at the end (~shots*n*chi^2)
     Mid-circuit : gates after the first measurement x shots (likely per-shot
                   re-simulation), time-integrated entanglement over snapshots
"""

import math
import os
import re
import time

CAP_S = 14400.0
MIN_S = 0.01
MAX_LC = 64
PARSE_BUDGET_S = 6.0        # stop detailed parsing after this and extrapolate
N_SNAPSHOTS = 16            # entropy snapshots through the circuit
SHOTS = 1024                # fixed in every training run (runtime-data.csv)
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "model_lgb.txt")
RIDGE_FILE = os.path.join(HERE, "model_ridge.json")   # optional blend partner
CLF_FILE = os.path.join(HERE, "model_clf.txt")        # optional timeout classifier
CONFIG_FILE = os.path.join(HERE, "model_config.json")
PI2 = math.pi / 2

# ----------------------------------------------------------------------------
# Gate knowledge
# ----------------------------------------------------------------------------
RANK4_2Q = {"swap", "iswap", "fsim", "dcx", "xx_plus_yy", "xx_minus_yy",
            "can", "u4", "su4", "unitary"}
SKIP_KW = {"OPENQASM", "include", "qreg", "creg", "qubit", "bit", "barrier",
           "input", "output", "const", "let", "def", "opaque", "defcalgrammar",
           "cal", "defcal", "gphase", "int", "uint", "float", "angle", "bool",
           "delay", "pragma", "return", "extern", "box"}
NONCLIFF_FIXED = {"t", "tdg", "ccx", "cswap", "ccz", "ch", "csx", "rccx",
                  "c3x", "c4x", "mcx", "c3sqrtx", "toffoli"}

_REG2 = re.compile(r"\bqreg\s+(\w+)\s*\[\s*(\d+)\s*\]")
_REG3 = re.compile(r"\bqubit\s*(?:\[\s*(\d+)\s*\])?\s+(\w+)\s*;")
_GATEDEF = re.compile(
    r"\bgate\s+(\w+)\s*(?:\(([^)]*)\))?\s*([^{]*)\{([^}]*)\}", re.S)
_COMMENT_LINE = re.compile(r"//[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_NUMEXPR = re.compile(r"[0-9eE+\-*/.,() ]*")

_param_cache = {}


def _param_vals(pstr):
    """Evaluate a parameter string like 'pi/4, -0.3' -> tuple of floats
    (None if symbolic). Cached, since the same strings repeat constantly."""
    r = _param_cache.get(pstr, 0)
    if r != 0:
        return r
    r = None
    try:
        expr = pstr.replace("pi", "3.141592653589793").replace("π", "3.141592653589793")
        if _NUMEXPR.fullmatch(expr):
            r = tuple(float(v) for v in eval("(" + expr + ",)", {"__builtins__": {}}, {}))
    except Exception:
        r = None
    if len(_param_cache) < 200000:
        _param_cache[pstr] = r
    return r


def _k(theta, unit=PI2):
    """Nearest integer multiple of `unit`, mod 4 (for pi/2) -- Clifford rounding."""
    return int(round(theta / unit)) % 4


def _is_clifford_params(vals):
    return vals is not None and all(abs(v / PI2 - round(v / PI2)) < 1e-6 for v in vals)


def _split_call(s):
    """'name(params) ops' -> (name, params_or_None, ops). Handles nested ()."""
    i, n = 0, len(s)
    while i < n and (s[i].isalnum() or s[i] == "_"):
        i += 1
    name, j, params = s[:i], i, None
    while j < n and s[j] == " ":
        j += 1
    if j < n and s[j] == "(":
        depth, k = 0, j
        while k < n:
            if s[k] == "(":
                depth += 1
            elif s[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        params, j = s[j + 1:k], k + 1
    return name, params, s[j:]


# ----------------------------------------------------------------------------
# Stabilizer tableau (phase-free) for Clifford-skeleton entanglement entropy
# ----------------------------------------------------------------------------
class Tableau:
    """n stabilizer generators stored COLUMN-wise: X[q] / Z[q] are Python ints
    whose bit r says whether generator r has an X / Z component on qubit q.
    Signs are ignored -- they never affect entanglement entropy.
    Gates become a couple of integer XORs, which is very fast in Python."""

    def __init__(self, n):
        self.n = n
        self.X = [0] * n
        self.Z = [1 << q for q in range(n)]      # |0...0> : generators Z_q

    # --- Clifford primitives ---
    def h(self, a):
        X, Z = self.X, self.Z
        X[a], Z[a] = Z[a], X[a]

    def s(self, a):                               # S / Sdg (same without signs)
        self.Z[a] ^= self.X[a]

    def sx(self, a):                              # sqrt(X) / sqrt(X)^dag
        self.X[a] ^= self.Z[a]

    def cx(self, c, t):
        X, Z = self.X, self.Z
        X[t] ^= X[c]
        Z[c] ^= Z[t]

    def cz(self, a, b):
        X, Z = self.X, self.Z
        Z[a] ^= X[b]
        Z[b] ^= X[a]

    def swap(self, a, b):
        X, Z = self.X, self.Z
        X[a], X[b] = X[b], X[a]
        Z[a], Z[b] = Z[b], Z[a]

    def measure(self, a):
        """Z-measurement: if some generator anticommutes with Z_a, it is
        replaced by Z_a (after multiplying it into the other anticommuting
        generators). Outcome/sign irrelevant for entropy."""
        X, Z = self.X, self.Z
        anti = X[a]
        if not anti:
            return
        pbit = anti & -anti
        others = anti ^ pbit
        keep = ~pbit
        for q in range(self.n):
            xq, zq = X[q], Z[q]
            if xq & pbit:
                xq ^= others
            if zq & pbit:
                zq ^= others
            X[q], Z[q] = xq & keep, zq & keep
        Z[a] |= pbit

    # --- entropy of every prefix cut [0..c] | [c+1..n-1] ---
    def prefix_entropies(self):
        """S(A) = rank_GF2(generators restricted to A) - |A|  (Fattal et al.).
        Rank of the restricted matrix = rank of its column vectors, which we
        build incrementally with an XOR basis -> all prefix cuts in one pass."""
        basis = {}
        rank = 0
        out = []
        X, Z = self.X, self.Z
        for q in range(self.n - 1):
            for v in (X[q], Z[q]):
                while v:
                    hb = v.bit_length() - 1
                    b = basis.get(hb)
                    if b is None:
                        basis[hb] = v
                        rank += 1
                        break
                    v ^= b
            out.append(rank - (q + 1))
        return out


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------
def parse_qasm(text, budget_s=PARSE_BUDGET_S):
    t0 = time.perf_counter()
    feats = {"n_bytes": len(text), "truncated": 0, "failed": 0,
             "n_if": text.count("if (") + text.count("if(")}

    if "/*" in text:
        text = _COMMENT_BLOCK.sub("", text)
    if "//" in text:
        text = _COMMENT_LINE.sub("", text)

    # ---- registers -> global qubit indices ----
    regs, nq = {}, 0
    for name, size in _REG2.findall(text):
        regs[name] = (nq, int(size)); nq += int(size)
    for size, name in _REG3.findall(text):
        sz = int(size) if size else 1
        regs[name] = (nq, sz); nq += sz
    nq = max(nq, 1)

    # ---- custom gate definitions (expanded recursively into primitives) ----
    gdefs = {}
    if re.search(r"(^|\n|;)\s*gate\s", text):
        for gname, _p, qargs, body in _GATEDEF.findall(text):
            qa = [x.strip() for x in qargs.split(",") if x.strip()]
            prims = []
            for st in body.split(";"):
                st = st.strip()
                if not st or st.startswith("barrier"):
                    continue
                if "@" in st:
                    st = st.rsplit("@", 1)[1].strip()
                nm, pr, ops = _split_call(st)
                loc = [qa.index(o.strip()) for o in ops.split(",") if o.strip() in qa]
                prims.append((nm, pr, loc))
            gdefs[gname] = prims
        text = _GATEDEF.sub("", text)

    expand_cache = {}

    def expand(nm, depth=0):
        if nm in expand_cache:
            return expand_cache[nm]
        out = []
        for pnm, ppr, loc in gdefs.get(nm, []):
            if pnm in gdefs and depth < 20:
                for snm, spr, sloc in expand(pnm, depth + 1):
                    out.append((snm, spr, [loc[i] for i in sloc if i < len(loc)]))
            else:
                out.append((pnm, ppr, loc))
        expand_cache[nm] = out
        return out

    # ---- state ----
    ncut = max(nq - 1, 1)
    lc = [0] * ncut                                  # bond tracker, log2 chi
    maxlc = [min(c + 1, nq - c - 1) for c in range(ncut)]
    cross = [0] * ncut
    layer = [0] * nq
    layer2 = [0] * nq
    occ = []                                          # qubits active per layer
    touched = [0] * nq
    hist2 = [0] * (MAX_LC + 1)
    hist1 = [0] * (MAX_LC + 1)
    c = dict(n_1q=0, n_2q=0, n_3q=0, n_meas=0, n_cond=0, n_reset=0,
             n_param=0, n_nonclifford=0, n_rank4=0, span_sum=0, span_max=0,
             n_nn=0, meas_before_last_gate=0, reset_before_last_gate=0)
    st = {"meas_seen": 0, "ops": 0, "first_meas_ops": -1, "snap_ops": 0,
          "resets_seen": 0}
    pairs = set()
    snaps = []                                        # (S_max at snapshot, gates in segment)
    tab = Tableau(nq)
    ent = {"peak_max": 0, "peak_mid": 0, "peak_mean": 0.0, "final_max": 0,
           "n_snap": 0}

    def snapshot():
        prof = tab.prefix_entropies()
        if not prof:
            return
        mx = max(prof)
        snaps.append((mx, st["ops"] - st["snap_ops"]))
        st["snap_ops"] = st["ops"]
        ent["n_snap"] += 1
        ent["final_max"] = mx
        if mx >= ent["peak_max"]:
            ent["peak_max"] = mx
            ent["peak_mean"] = sum(prof) / len(prof)
        mid = prof[len(prof) // 2]
        if mid > ent["peak_mid"]:
            ent["peak_mid"] = mid

    def place(qs):
        """Depth / width bookkeeping for a gate on qubits qs."""
        L = 0
        for q in qs:
            if layer[q] > L:
                L = layer[q]
        for q in qs:
            layer[q] = L + 1
            touched[q] = 1
        if L >= len(occ):
            occ.append(0)
        occ[L] += len(qs)

    # --- Clifford-skeleton action for each gate on the tableau ---
    def tab_1q(nm, vals, a):
        if nm == "h":
            tab.h(a)
        elif nm in ("s", "sdg"):
            tab.s(a)
        elif nm in ("sx", "sxdg"):
            tab.sx(a)
        elif vals is None:
            return                                   # x,y,z,id,t,tdg,... no-op
        elif nm in ("rz", "p", "u1", "phase"):
            if _k(vals[0]) & 1:
                tab.s(a)
        elif nm == "rx":
            if _k(vals[0]) & 1:
                tab.sx(a)
        elif nm == "ry":
            if _k(vals[0]) & 1:
                tab.h(a)
        elif nm == "u2" and len(vals) >= 2:          # rz(phi) ry(pi/2) rz(lam)
            if _k(vals[1]) & 1:
                tab.s(a)
            tab.h(a)
            if _k(vals[0]) & 1:
                tab.s(a)
        elif nm in ("u3", "u", "U") and len(vals) >= 3:
            if _k(vals[2]) & 1:
                tab.s(a)
            if _k(vals[0]) & 1:
                tab.h(a)
            if _k(vals[1]) & 1:
                tab.s(a)

    def tab_2q(nm, vals, a, b):
        if nm in ("cx", "CX", "cnot", "ecr", "ch", "csx", "cy"):
            tab.cx(a, b)
        elif nm == "cz":
            tab.cz(a, b)
        elif nm == "swap":
            tab.swap(a, b)
        elif nm == "iswap":
            tab.swap(a, b); tab.cz(a, b); tab.s(a); tab.s(b)
        elif nm == "dcx":
            tab.cx(a, b); tab.cx(b, a)
        elif nm in ("cp", "cu1", "crz", "cphase"):
            if vals is not None and int(round(vals[0] / math.pi)) & 1:
                tab.cz(a, b)
        elif nm in ("crx", "cry", "cu3", "cu"):
            if vals is not None and int(round(vals[0] / math.pi)) & 1:
                tab.cx(a, b)
        elif nm in ("rzz", "rxx", "ryy", "rzx"):
            if vals is not None and _k(vals[0]) & 1:
                if nm == "rxx" or nm == "ryy":
                    tab.h(a); tab.h(b)
                elif nm == "rzx":
                    tab.h(b)
                tab.cz(a, b); tab.s(a); tab.s(b)
                if nm == "rxx" or nm == "ryy":
                    tab.h(a); tab.h(b)
                elif nm == "rzx":
                    tab.h(b)
        elif nm == "fsim":
            if vals is not None and len(vals) >= 2:
                if _k(vals[0]) & 1:
                    tab.swap(a, b); tab.cz(a, b); tab.s(a); tab.s(b)
                if int(round(vals[1] / math.pi)) & 1:
                    tab.cz(a, b)
        elif nm in ("xx_plus_yy", "xx_minus_yy"):
            if vals is not None and _k(vals[0]) & 1:
                tab.swap(a, b); tab.cz(a, b); tab.s(a); tab.s(b)
        else:
            tab.cx(a, b)                             # unknown / ctrl@ gates: treat as an entangler

    def apply(nm, pr, qs):
        k = len(qs)
        if k == 0:
            return
        vals = _param_vals(pr) if pr is not None else None
        if pr is not None:
            c["n_param"] += 1
            if not _is_clifford_params(vals):
                c["n_nonclifford"] += 1
        elif nm in NONCLIFF_FIXED:
            c["n_nonclifford"] += 1
        c["meas_before_last_gate"] = st["meas_seen"]
        c["reset_before_last_gate"] = st["resets_seen"]
        st["ops"] += 1
        if k == 1:
            a = qs[0]
            c["n_1q"] += 1
            place(qs)
            m = lc[a - 1] if a > 0 else 0
            if a < nq - 1 and lc[a] > m:
                m = lc[a]
            hist1[m if m < MAX_LC else MAX_LC] += 1
            tab_1q(nm, vals, a)
            return
        a, b = min(qs), max(qs)
        if a == b:
            return
        span = b - a
        if k == 2:
            c["n_2q"] += 1
            d = 2 if nm in RANK4_2Q else 1
            if d == 2:
                c["n_rank4"] += 1
            tab_2q(nm, vals, qs[0], qs[1])
        else:
            c["n_3q"] += 1
            d = k - 1
            for ctl in qs[:-1]:                      # multi-controlled: CX chain proxy
                tab.cx(ctl, qs[-1])
        if k == 2:
            pairs.add((a, b))
        else:
            for i in range(len(qs) - 1):
                pairs.add((min(qs[i], qs[i + 1]), max(qs[i], qs[i + 1])))
        c["span_sum"] += span
        if span > c["span_max"]:
            c["span_max"] = span
        if span == 1:
            c["n_nn"] += 1
        m = 0
        for cc in range(a, b):
            v = lc[cc] + d
            if v > maxlc[cc]:
                v = maxlc[cc]
            lc[cc] = v
            cross[cc] += 1
            if v > m:
                m = v
        hist2[m if m < MAX_LC else MAX_LC] += span
        L2 = 0
        for q in qs:
            if layer2[q] > L2:
                L2 = layer2[q]
        for q in qs:
            layer2[q] = L2 + 1
        place(qs)

    def q_of(tok):
        tok = tok.strip()
        br = tok.find("[")
        if br < 0:
            r = regs.get(tok)
            return list(range(r[0], r[0] + r[1])) if r else []
        r = regs.get(tok[:br].strip())
        inner = tok[br + 1:tok.find("]", br)]
        if ":" in inner:                              # QASM3 slice q[0:3]
            try:
                lo, hi = (int(x) for x in inner.split(":")[:2])
            except ValueError:
                return []
            base = r[0] if r else 0
            return [base + i for i in range(lo, hi + 1)]
        try:
            idx = int(inner)
        except ValueError:
            return []
        if r is None:
            return [idx] if idx < nq else []
        return [r[0] + idx]

    stmts = text.split(";")
    n_stmts = len(stmts)
    snap_every = max(1, n_stmts // N_SNAPSHOTS)
    last_snap = 0
    gates_since_snap = 0
    processed = 0

    for si, s in enumerate(stmts):
        if (si & 0x3FFF) == 0 and si and time.perf_counter() - t0 > budget_s:
            feats["truncated"] = 1
            break
        processed = si + 1
        if si - last_snap >= snap_every and gates_since_snap:
            snapshot(); last_snap = si; gates_since_snap = 0
        s = s.strip()
        if not s:
            continue
        while s and s[0] in "}{":
            s = s[1:].lstrip()
        if not s:
            continue
        cond = False
        if s.startswith("if"):
            p = s.find(")")
            if p < 0:
                continue
            s = s[p + 1:].lstrip()
            while s and s[0] == "{":
                s = s[1:].lstrip()
            cond = True
        elif s.startswith("else"):
            s = s[4:].lstrip().lstrip("{").lstrip()
            cond = True
        if not s:
            continue
        if "measure" in s:
            ops = s.split("measure", 1)[1].split("->")[0]
            qs = q_of(ops) if ops.strip() else []
            # snapshot just before measurements start (captures pre-collapse peak)
            if gates_since_snap and (si - last_snap) * 4 >= snap_every:
                snapshot(); last_snap = si; gates_since_snap = 0
            for q in qs:
                tab.measure(q)
            if st["first_meas_ops"] < 0:
                st["first_meas_ops"] = st["ops"]
            c["n_meas"] += max(len(qs), 1)
            st["meas_seen"] += max(len(qs), 1)
            continue
        first = s.split(None, 1)[0].split("(")[0].split("[")[0]
        if first in SKIP_KW or first.startswith("for") or first.startswith("while"):
            continue
        if first == "reset":
            # reset = measure + conditional flip: non-unitary, disentangles the
            # qubit, and (like a mid-circuit measurement) likely forces per-shot work
            rq = q_of(s[5:])
            for q in rq:
                tab.measure(q)
            if st["first_meas_ops"] < 0:
                st["first_meas_ops"] = st["ops"]
            c["n_reset"] += max(len(rq), 1)
            st["resets_seen"] += max(len(rq), 1)
            continue
        if "@" in s:
            s = s.rsplit("@", 1)[1].strip()
        nm, pr, ops = _split_call(s)
        if not nm:
            continue
        if cond:
            c["n_cond"] += 1
        opl = [q_of(o) for o in ops.split(",") if o.strip()]
        if not opl or any(not o for o in opl):
            continue
        if all(len(o) == 1 for o in opl):
            groups = [[o[0] for o in opl]]
        else:
            L = max(len(o) for o in opl)
            groups = [[o[i] if len(o) > 1 else o[0] for o in opl] for i in range(L)]
        if nm in gdefs:
            prims = expand(nm)
            for qs in groups:
                for pnm, ppr, loc in prims:
                    apply(pnm, ppr, [qs[i] for i in loc if i < len(qs)])
        else:
            for qs in groups:
                apply(nm, pr, qs)
        gates_since_snap += 1

    if gates_since_snap or ent["n_snap"] == 0:
        snapshot()

    # ---- extrapolate counts if we ran out of time ----
    scale = n_stmts / max(processed, 1) if feats["truncated"] else 1.0
    if scale != 1.0:
        for key in ("n_1q", "n_2q", "n_3q", "n_meas", "n_cond", "n_reset",
                    "n_param", "n_nonclifford", "n_rank4", "span_sum", "n_nn",
                    "meas_before_last_gate", "reset_before_last_gate"):
            c[key] = int(c[key] * scale)
        hist1 = [h * scale for h in hist1]
        hist2 = [h * scale for h in hist2]

    n2 = c["n_2q"] + c["n_3q"]
    depth = max(layer) if layer else 0
    deg = [0] * nq
    for a_, b_ in pairs:
        deg[a_] += 1; deg[b_] += 1
    after_meas = (st["ops"] - st["first_meas_ops"]) if st["first_meas_ops"] >= 0 else 0
    if occ:
        wm = sum(occ) / len(occ)
        wv = sum((o - wm) ** 2 for o in occ) / len(occ)
    else:
        wm = wv = 0.0
    feats.update(c)
    feats.update({
        "n_qubits": nq,
        "n_used": sum(touched),
        "depth": depth * scale,
        "depth2q": (max(layer2) if layer2 else 0) * scale,
        "width_mean": wm / nq,
        "width_std": math.sqrt(wv) / nq,
        "span_mean": c["span_sum"] / n2 if n2 else 0.0,
        "frac_nn": c["n_nn"] / n2 if n2 else 0.0,
        "max_lc": max(lc) if lc else 0,
        "mean_lc": sum(lc) / len(lc) if lc else 0.0,
        "cross_max": max(cross) * scale if cross else 0,
        "cross_mean": (sum(cross) / len(cross)) * scale if cross else 0.0,
        "S_peak_max": ent["peak_max"],
        "S_peak_mid": ent["peak_mid"],
        "S_peak_mean": ent["peak_mean"],
        "S_final_max": ent["final_max"],
        "snaps": [[int(sv), int(g * scale)] for sv, g in snaps],
        "n_pairs": len(pairs),
        "deg_mean": sum(deg) / nq,
        "deg_max": max(deg) if deg else 0,
        "gates_after_meas": int(after_meas * scale),
        "hist1": hist1,
        "hist2": hist2,
        "parse_s": time.perf_counter() - t0,
    })
    return feats


# ----------------------------------------------------------------------------
# Feature vector for one (circuit, threshold) pair
# ----------------------------------------------------------------------------
FEATURE_GROUPS = {
    "setting":     ["log2_thr"],
    "size":        ["log_nq", "frac_used", "log_ngates", "log_n1q", "log_n2q",
                    "log_n3q", "log_depth", "log_depth2q", "log_gates_per_layer",
                    "width_mean", "width_std"],
    "gate_mix":    ["frac_2q", "frac_param", "frac_nonclifford", "frac_rank4",
                    "frac_3q"],
    "locality":    ["span_mean", "log_span_max", "frac_nn", "log_cross_max",
                    "log_cross_mean"],
    "bond_tracker": ["lcost2", "lcost1", "frac_sat", "max_lc", "mean_lc"],
    "entropy":     ["S_peak_max", "S_peak_mid", "S_peak_mean", "S_frac",
                    "S_excess_thr", "S_capped_thr", "lcost_entropy",
                    "S_time_mean", "frac_time_sat", "lcost_entropy_time"],
    "graph":       ["log_pairs", "pair_density", "deg_mean", "log_deg_max"],
    "sampling":    ["lcost_sample", "lcost_sample_tracker"],
    "measurement": ["log_meas", "log_meas_mid", "log_cond", "log_if",
                    "log_reset", "log_reset_mid",
                    "log_gates_after_meas", "log_shot_resim"],
    "flags":       ["log_bytes", "failed", "truncated"],
}
FEATURE_NAMES = [n for g in FEATURE_GROUPS.values() for n in g]


def feature_dict(f, thr):
    lg = lambda x: math.log10(1.0 + max(float(x), 0.0))
    thr = float(thr)
    lthr = math.log2(max(thr, 1.0))
    h1, h2 = f.get("hist1", []), f.get("hist2", [])
    cost2 = cost1 = sat = tot = 0.0
    for m, w in enumerate(h2):
        if w:
            cost2 += w * min(2.0 ** m, thr) ** 3
            tot += w
            if 2.0 ** m >= thr:
                sat += w
    for m, w in enumerate(h1):
        if w:
            cost1 += w * min(2.0 ** m, thr) ** 2
    nq = max(f.get("n_qubits", 1), 1)
    n1, n2, n3 = f.get("n_1q", 0), f.get("n_2q", 0), f.get("n_3q", 0)
    ng = n1 + n2 + n3
    depth = f.get("depth", 0)
    S = f.get("S_peak_max", 0)
    chi_S = min(2.0 ** min(S, 60), thr)
    snaps = f.get("snaps", [])
    gw = sum(g for _, g in snaps)
    S_time = sum(sv * g for sv, g in snaps) / gw if gw else float(S)
    t_sat = sum(g for sv, g in snaps if sv >= lthr) / gw if gw else 0.0
    cost_time = sum(g * min(2.0 ** min(sv, 60), thr) ** 3 for sv, g in snaps)
    chi_T = min(2.0 ** min(f.get("max_lc", 0), 60), thr)
    n_used = f.get("n_used", nq)
    pairs_possible = nq * (nq - 1) / 2.0
    mid = f.get("meas_before_last_gate", 0) + f.get("reset_before_last_gate", 0)
    return {
        "log2_thr": lthr,
        "log_nq": math.log10(nq), "frac_used": f.get("n_used", nq) / nq,
        "log_ngates": lg(ng), "log_n1q": lg(n1), "log_n2q": lg(n2), "log_n3q": lg(n3),
        "log_depth": lg(depth), "log_depth2q": lg(f.get("depth2q", 0)),
        "log_gates_per_layer": lg(ng / depth) if depth else 0.0,
        "width_mean": f.get("width_mean", 0.0), "width_std": f.get("width_std", 0.0),
        "frac_2q": (n2 + n3) / ng if ng else 0.0,
        "frac_param": f.get("n_param", 0) / ng if ng else 0.0,
        "frac_nonclifford": f.get("n_nonclifford", 0) / ng if ng else 0.0,
        "frac_rank4": f.get("n_rank4", 0) / max(n2, 1),
        "frac_3q": n3 / ng if ng else 0.0,
        "span_mean": f.get("span_mean", 0.0), "log_span_max": lg(f.get("span_max", 0)),
        "frac_nn": f.get("frac_nn", 0.0),
        "log_cross_max": lg(f.get("cross_max", 0)), "log_cross_mean": lg(f.get("cross_mean", 0)),
        "lcost2": lg(cost2), "lcost1": lg(cost1), "frac_sat": sat / tot if tot else 0.0,
        "max_lc": f.get("max_lc", 0), "mean_lc": f.get("mean_lc", 0.0),
        "S_peak_max": S, "S_peak_mid": f.get("S_peak_mid", 0),
        "S_peak_mean": f.get("S_peak_mean", 0.0),
        "S_frac": S / (nq / 2.0) if nq > 1 else 0.0,
        "S_excess_thr": max(0.0, S - lthr),
        "S_capped_thr": min(S, lthr),
        "lcost_entropy": lg((n2 + n3) * chi_S ** 3),
        "S_time_mean": S_time, "frac_time_sat": t_sat,
        "lcost_entropy_time": lg(cost_time),
        "log_pairs": lg(f.get("n_pairs", 0)),
        "pair_density": f.get("n_pairs", 0) / pairs_possible if pairs_possible else 0.0,
        "deg_mean": f.get("deg_mean", 0.0), "log_deg_max": lg(f.get("deg_max", 0)),
        "lcost_sample": lg(SHOTS * n_used * chi_S ** 2),
        "lcost_sample_tracker": lg(SHOTS * n_used * chi_T ** 2),
        "log_gates_after_meas": lg(f.get("gates_after_meas", 0)),
        "log_shot_resim": lg(SHOTS * f.get("gates_after_meas", 0)) if mid else 0.0,
        "log_meas": lg(f.get("n_meas", 0)), "log_meas_mid": lg(f.get("meas_before_last_gate", 0)),
        "log_cond": lg(f.get("n_cond", 0)), "log_if": lg(f.get("n_if", 0)),
        "log_reset": lg(f.get("n_reset", 0)),
        "log_reset_mid": lg(f.get("reset_before_last_gate", 0)),
        "log_bytes": lg(f.get("n_bytes", 0)),
        "failed": float(f.get("failed", 0)), "truncated": float(f.get("truncated", 0)),
    }


def feature_vector(f, thr):
    d = feature_dict(f, thr)
    return [d[n] for n in FEATURE_NAMES]


# ----------------------------------------------------------------------------
# Harness interface
# ----------------------------------------------------------------------------
class RuntimeModel:
    """Loads whatever train.py saved:
       model_lgb.txt     LightGBM regressor on log10(seconds)       (required)
       model_ridge.json  ridge regression, blended in with weight    (optional)
       model_clf.txt     timeout classifier: P(timeout) > 0.5 -> cap (optional)
       model_config.json blend weight / which parts are active"""

    def __init__(self):
        import json
        self.booster = self.clf = self.ridge = None
        self.cfg = {"blend_w": 1.0, "use_clf": False}
        try:
            if os.path.exists(CONFIG_FILE):
                self.cfg.update(json.load(open(CONFIG_FILE)))
            if os.path.exists(MODEL_FILE):
                import lightgbm as lgb
                self.booster = lgb.Booster(model_file=MODEL_FILE)
                if self.cfg.get("use_clf") and os.path.exists(CLF_FILE):
                    self.clf = lgb.Booster(model_file=CLF_FILE)
            if self.cfg.get("blend_w", 1.0) < 1.0 and os.path.exists(RIDGE_FILE):
                self.ridge = json.load(open(RIDGE_FILE))
        except Exception as e:
            print(f"[model] could not load trained model: {e}")

    def _ridge(self, x):
        r = self.ridge
        z = [(xi - m) / s for xi, m, s in zip(x, r["mean"], r["scale"])]
        return r["intercept"] + sum(zi * ci for zi, ci in zip(z, r["coef"]))

    def featurize(self, qasm_text):
        try:
            return parse_qasm(qasm_text)
        except Exception:
            return {"failed": 1, "n_bytes": len(qasm_text), "n_qubits": 1,
                    "hist1": [], "hist2": []}

    def predict(self, features, threshold):
        try:
            x = feature_vector(features, threshold)
            if self.booster is not None:
                log_s = float(self.booster.predict([x])[0])
                if self.ridge is not None:
                    w = self.cfg["blend_w"]
                    log_s = w * log_s + (1 - w) * self._ridge(x)
                if self.clf is not None and float(self.clf.predict([x])[0]) > 0.5:
                    return CAP_S
            else:
                log_s = -0.8 + 0.25 * x[FEATURE_NAMES.index("lcost2")]
        except Exception:
            log_s = 1.0
        if not math.isfinite(log_s):
            log_s = 1.0
        return float(min(max(10.0 ** log_s, MIN_S), CAP_S))
