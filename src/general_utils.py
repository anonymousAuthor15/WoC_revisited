import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import json, os, pickle
import heapq

def save_to_json(data, path, dir='./'):
    json_file = os.path.join(dir, f"{path}.json")
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def load_from_file(filepath):
    """
    Load data from a JSON or pickle file.
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use '.json' or '.pkl'.")
    
def get(value, default):
    return value if value is not None else default

def my_max(data, key= lambda x: x, arg=False):
    """
    Returns the element in data with the highest value according to key function.
    """
    if not data:
        raise ValueError("argmax() data is an empty sequence")
    
    max_elem = data[0]
    max_value = key(max_elem)
    
    for i, elem in enumerate(data[1:], start=1):
        value = key(elem)
        if value > max_value:
            max_value = value
            max_elem = elem
            arg_max = i

    max_elem_list = []
    arg_max_list = []
    for i, elem in enumerate(data):
        value = key(elem)
        if value == max_value: 
            max_elem_list.append(elem)
            arg_max_list.append(i)
            
    return max_elem_list if not arg else arg_max_list

def choice(options,p=None):
    x = np.random.rand()
    if p is None: p = [1/len(options) for _ in options]
    cum = 0
    for i,prob in enumerate(p):
        cum += prob
        if x < cum:
            break
    return options[i]

class MedianHeaps:
    def __init__(self):
        self.lower_half = []  # Max-heap (inverted min-heap)
        self.upper_half = []  # Min-heap

    def add_number(self, num):
        if not self.lower_half or num <= -self.lower_half[0]:
            heapq.heappush(self.lower_half, -num)  # Invert to simulate max-heap
        else:
            heapq.heappush(self.upper_half, num)

        # Balance the heaps
        if len(self.lower_half) > len(self.upper_half) + 1:
            heapq.heappush(self.upper_half, -heapq.heappop(self.lower_half))
        elif len(self.upper_half) > len(self.lower_half):
            heapq.heappush(self.lower_half, -heapq.heappop(self.upper_half))

    def get_median(self):
        if len(self.lower_half) > len(self.upper_half):
            return -self.lower_half[0]
        return (-self.lower_half[0] + self.upper_half[0]) / 2
    
class DisplayProgressBar:
    def __init__(self, total, prefix='', suffix='', length=50):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.current = 0

    def update(self, increment=1):
        self.current += increment
        percent = self.current / self.total
        filled_length = int(self.length * percent)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {self.current}/{self.total} ({percent:.1%}) {self.suffix}', end='\r')
        if self.current >= self.total:
            print()

def last_update(seq, tolerance=0):
    return len(seq) + 1 - np.min(np.where(abs(seq[::-1] - seq[-1]) > tolerance), initial=len(seq))

def make_color_darker(color, factor=0.7):
    """
    Darken a matplotlib color by multiplying RGB by factor in (0,1].
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))

def inside(a, b):
    if not isinstance(a, (list, np.ndarray)):
        return 1 if (a in b) else 0
    return np.array([1 if x in b else 0 for x in a])

def max0(x):
    return max(0, x)

def minN(x, N):
    return min(x, N-1)

def normalize_func(normalization, ans, cutoff=1):
    # if ans is not np array, convert it
    if isinstance(ans, list):
        ans_h = np.array(ans)
    else:
        ans_h = ans
        
    if normalization == "perc":
        # print("Using percentile normalization")
        def foo(x, side='right'):
            if isinstance(x, list):
                x_new = np.array(x)
            else:
                x_new = x
            indices = np.searchsorted(ans_h, x_new, side=side)
            # return (indices - inside(x_new, ans_h)) / len(ans_h)
            return (indices) / len(ans_h)
        return foo
    
    elif normalization == "linear":
        # print("Using cutoff normalization")
        denom = ans_h[-cutoff-1] - ans_h[cutoff]
        return lambda x, side=None: ((np.array(x) if isinstance(x, list) else x) - ans[cutoff]) / denom if denom != 0 else \
                                    ((np.array(x) if isinstance(x, list) else x) - ans[0]) / (max(ans) - ans[0])
    
    elif normalization == "linear2":
        # print("Using cutoff normalization")
        denom = ans_h[-cutoff-1] - ans_h[cutoff]
        return lambda x, side=None: (ans_h[cutoff]  + (np.array(x) if isinstance(x, list) else x) - ans[cutoff]) / denom if denom != 0 else \
                                    (ans_h[0] + (np.array(x) if isinstance(x, list) else x) - ans[0]) / (max(ans) - ans[0])
    
    elif normalization == 'perc1_1':
        # print("Using percentile 1-1 normalization")
        def foo(x, side='right'):
            if isinstance(x, list) or x is np.ndarray:
                result = np.zeros(len(x))
                for i, val in enumerate(x):
                    if val == np.median(ans_h):
                        result[i] = 0.5
                    elif val in ans_h:
                        result[i] = np.searchsorted(ans_h, val, side='right' if val < np.median(ans_h) else 'left') / len(ans_h)
                    else:
                        idx = np.searchsorted(ans_h, val)
                        x1 = ans_h[max0(idx-1)]
                        x2 = ans_h[idx]
                        frac = (val - x1) / (x2 - x1) if x2 != x1 else 1
                        result[i] = min(1, (idx + frac) / len(ans_h))
                return result
            else:
                if x == np.median(ans_h):
                    return 0.5
                if x in ans_h:
                    return np.searchsorted(ans_h, x, side='right' if x < np.median(ans_h) else 'left') / len(ans_h)
                idx = min(np.searchsorted(ans_h, x, side='right'), len(ans_h)-1)
                x1 = ans_h[max0(idx-1)]
                x2 = ans_h[idx]
                frac = (x - x1) / (x2 - x1) if x2 != x1 else 1
                return min(1, (idx + frac) / len(ans_h))
        return foo
    else:
        raise ValueError(f"normalization must be 'perc' or 'linear' or 'perc1_1', not {normalization}")


def fit_percentiles(p, sorted_array, N=None):
    if N is None:
        N = len(sorted_array)
    p0 = minN(max0(int(np.floor(p * N))), N)
    p1 = max0(minN(int(np.ceil(p * N)), N))
    x0 = sorted_array[p0]
    x1 = sorted_array[p1]
    if not p0 <= minN(p*N, N) <= p1:
        raise ValueError(f"Percentile calculation error: p*N={p*N}, p0={p0}, p1={p1}")
    if x0 == x1:
        return x0
    intermediate = (p * N - p0) / (p1 - p0) * (x1 - x0) + x0
    if not x0 <= intermediate <=  x1:
        raise ValueError(f"Percentile calculation error: x0={x0}, intermediate={intermediate}, x1={x1}")
    return intermediate

def domain_func(domain, func, **kwargs):
    task_ids = df[df['domain_name'] == domain]['task_id'].unique()
    results = {}
    for task_id in task_ids:
        results[int(task_id)] = func(task_id, **kwargs)
    return results

def percentile_score(point, correct, data):
    if isinstance(data, int):
        answers = get_answers(data)
    else:
        answers = data
    ctrl_distances = sorted(-abs(np.array(answers) - correct))
    normalizer_dist = normalize_func('perc', ctrl_distances)
    if isinstance(point, list) or isinstance(point, np.ndarray):
        return np.array([normalizer_dist(-abs(p - correct)) for p in point])
    return normalizer_dist(-abs(point - correct))

def find_ms(answers, k, confidence=0.95):
    N = len(answers)
    if N == 0:
        raise ValueError("Empty answers provided")
    answers_sorted = np.sort(answers)

    if not (0 <= k <= 1):
        raise ValueError(f"k must be in [0, 1] and not {k}")
    
    # normalize percentiles
    m_minus = fit_percentiles(0.5 - k/2, answers_sorted, N)
    m_plus = fit_percentiles(0.5 + k/2, answers_sorted, N)

    # Calculate the confidence interval
    # eps = np.sqrt(np.log(2/confidence) / (2 * N))
    eps = np.sqrt(np.log(2/(1-confidence)) / (2 * N))
    conf_minus = (m_minus - fit_percentiles(max0(0.5 - k/2 - eps), answers_sorted, N),
                    fit_percentiles(max0(0.5 - k/2 + eps), answers_sorted, N) - m_minus)
    conf_plus = (m_plus - fit_percentiles(minN(0.5 + k/2 - eps, N), answers_sorted, N),
                    fit_percentiles(minN(0.5 + k/2 + eps, N), answers_sorted, N) - m_plus)
    
    return m_minus, m_plus, conf_minus, conf_plus

def influence_curve_new(M_minus, M_plus, confidence=(0,0), a=0, 
                        colors=["#0D7AE7", "#5DA7FB"],
                        x_range=(0, 1), n_points=100, eq=None, 
                        correct=None, correct_perc=None, 
                        medians=None, control_medians=None, 
                        median=None, points=None, 
                        ax=None, title="Influence Curve", plot=1):
    
    x_vals = np.linspace(*x_range, n_points)
    def infl_curve(M_m, M_p):
        return [a*x + (1-a)*M_m if x < M_m else 
                       (x if x < M_p else
                        a*x + (1-a)*M_p) for x in x_vals]
    
    influence_curve = infl_curve(M_minus, M_plus)
    interval = M_plus - M_minus
    
    if confidence != (0,0):
        lower_bound = infl_curve(M_minus - confidence[0][0], M_plus - confidence[1][0])
        upper_bound = infl_curve(M_minus + confidence[0][1], M_plus + confidence[1][1])

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        if control_medians:
            ax.axvspan(control_medians[0], control_medians[1], 
                       color='gray', alpha=0.2, zorder=0, label='Control medians range')

        # Plot the curve and identity line
        ax.plot(x_vals, influence_curve,
                # label=fr'[{M_minus:.2f} +- {confidence[0]:.1f}, {M_plus:.2f}] +- {confidence[1]:.1f}',
                color=colors[0], alpha=0.8)

        ax.plot(x_vals, x_vals, '--', color='black', alpha=0.5)

        # ax.axvline(M_plus, ymax=M_plus, color=colors[1], linestyle='--', marker='^',label=fr'$M^+ = {M_plus:.2f}$')
        # ax.axvline(M_minus, ymax=M_minus, color=colors[1], linestyle='--', marker='v',label=fr'$M^- = {M_minus:.2f}$')

        ax.plot(M_plus, M_plus, 'v', color=colors[1], markersize=6, label=fr'U = {M_plus:.2f}')   
        ax.plot(M_minus, M_minus, '^', color=colors[1], markersize=6, label=fr'L = {M_minus:.2f}')    

        if confidence != (0,0):
            ax.fill_between(x_vals, lower_bound, upper_bound, color='lightblue', alpha=0.8)

        if eq is not None:
            ax.plot(eq, eq, 'D', color="#ea08b1", markersize=5, label='Final median')
        if correct is not None:
            if correct < min(x_range) or correct > max(x_range):
                ax.plot([], [], label=f'Correct answer: {correct:.2f}', alpha=0)
            else:
                ax.plot(correct, correct, '*', color="#542AD2", markersize=8, label=f'Correct answer')

        if medians:
            ax.plot(medians[:-1], medians[1:], color="#ea08b1", marker='x', alpha=0.2)

        if median:
            ax.axvline(median, ymax=median, color='black', linestyle=':', label=fr'Median = {median:.2f}', alpha=0.5)

        if points:
            for point in points:
                ax.plot(point[0], point[1], 'o', color=point[2] if point[2] != None else 'black', 
                        label = point[3] if point[3] != None else "", markersize=3, alpha=0.8)

        ax.set_xlabel('Observed median $m$')
        ax.set_ylabel('Expected median')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    return ax, M_minus - confidence[0][0] <= eq <= M_plus + confidence[1][1] if eq!=None else -100


def non_fliers(vals, whis=1.5):
    vals = np.asarray(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo = q1 - whis * iqr
    hi = q3 + whis * iqr
    mask = (vals >= lo) & (vals <= hi)
    return vals[mask], vals[~mask], lo, hi

def myboxplot(x=None,
              vals=None, 
              func=None,
              data=None,
              title=None,

            # colors
              box_color="#A4DBF4C1",
              dots_colors='blue',

            # labels
             label=None,
              xlabels=None,
              ylabel=None,
              ax=None,
              show_zero_line=False,
              w=None):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,4))
    
    if data is None and vals is None:
        raise ValueError("Provide either raw data (x) or precomputed stats (data).")

    if x is None: 
        x = range(len(vals) if vals is not None else len(data))

    if data is None:
        # compute stats
        data = {}
        for i in vals:
            data[i] = func(i)
    
    if show_zero_line:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

    # persistent storage on the axis
    if not hasattr(ax, "_myboxplot_legend"):
        ax._myboxplot_legend = []

    # add only if new label
    existing = {hh.get_label() for hh in ax._myboxplot_legend}

    # boxplot
    bp = ax.boxplot(
        [data[i] for i in data.keys()],
        positions=x,
        tick_labels=xlabels if xlabels else None,
        patch_artist=True,
        boxprops=dict(facecolor=box_color, edgecolor=box_color, alpha=0.3),
        medianprops=dict(color="black", linewidth=2, alpha=0),
        whiskerprops=dict(color="gray", linewidth=1.5),
        capprops=dict(color="gray", linewidth=1.5),
        flierprops=dict(marker="x", markerfacecolor=dots_colors[i] if isinstance(dots_colors, (list, dict)) else dots_colors, 
                                    markeredgecolor=dots_colors[i] if isinstance(dots_colors, (list, dict)) else dots_colors, markersize=3, alpha=0.7),
        meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor=make_color_darker(box_color), markersize=5, alpha=0.9),
        showmeans=True,
        widths=w,
    )

    for w in bp["whiskers"]:
        w.set_visible(False)
    for c in bp["caps"]:
        c.set_visible(False)
    
    # overlay all points except fliers
    for i, key in zip(x, data.keys()):
        vals = data[key]
        keep, out, lo, hi = non_fliers(vals, whis=1.5)
        x_jitter = np.random.normal(i, 0.06, size=len(keep))  # jitter
        ax.scatter(x_jitter, keep, s=10, alpha=0.5, linewidths=0, color=dots_colors[i] if isinstance(dots_colors, (list, dict)) else dots_colors)

    legend_handles = [
        # plt.Line2D([0], [0], marker='o', color='w', label='Mean', markerfacecolor="white", markeredgecolor=make_color_darker(box_color), markersize=5, alpha=0.9),
        plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=dots_colors[i] if isinstance(dots_colors, (list, dict)) else dots_colors, 
                                    markeredgecolor=dots_colors[i] if isinstance(dots_colors, (list, dict)) else dots_colors, markersize=5, alpha=0.7),
    ]

    for h in legend_handles:
        ax._myboxplot_legend.append(h)
    

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # ax.tick_params(axis="x", rotation=45, labelsize=10) 
    if xlabels is not None:
        ax.tick_params(axis="x", labelsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticks([])          # remove tick locations
        ax.set_xticklabels([])     # remove tick text (safe even if none)
        ax.tick_params(axis="x", bottom=False)  # hide tick marks
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(handles=ax._myboxplot_legend, loc="lower left", frameon=True)
    # fig.tight_layout()
