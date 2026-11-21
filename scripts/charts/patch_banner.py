p = '/Users/nikshepsvn/Code/inkling-abl/scripts/charts/make_banner.py'
s = open(p).read()
a = s.index('    callout(fig, xs[2], "CAPABILITY"')
b = s.index('    for x in xs[1:]:')
new = '''    drop = (1 - kl_u / kl_ou) * 100
    callout(fig, xs[2], "DRIFT", f"\\u2212{drop:.0f}%",
            "less drift from the original\\nthan OpenUncensored", "", dot=False)
    fig.text(xs[2], 0.33, f"KL {kl_u:.3f} vs {kl_ou:.3f}  \\u00B7  sealed {split}",
             fontsize=9.5, color=INK, va="center")
    mini_bars(fig, xs[0], 0.25, 0.25, (ref, ref_ou), 100)
    mini_bars(fig, xs[1], 0.25, 0.25, (use, use_ou), 100)
    mini_bars(fig, xs[2], 0.25, 0.25, (kl_u, kl_ou), kl_ou * 1.1)
'''
s = s[:a] + new + s[b:]
s = s.replace('def callout(fig, x, label, big, sub, ou_text):', 'def callout(fig, x, label, big, sub, ou_text, dot=True):')
s = s.replace('''    fig.text(x, 0.375, sub, fontsize=9.5, color=INK2, va="bottom")
    fig.text(x, 0.33, "\\u25cf", fontsize=9, color=ORANGE, va="center")
    fig.text(x + 0.02, 0.33, ou_text, fontsize=10.5, color=INK, va="center")''',
'''    fig.text(x, 0.375, sub, fontsize=9.5, color=INK2, va="bottom", linespacing=1.15)
    if dot:
        fig.text(x, 0.33, "\\u25cf", fontsize=9, color=ORANGE, va="center")
        fig.text(x + 0.02, 0.33, ou_text, fontsize=10.5, color=INK, va="center")''')
open(p, 'w').write(s)
