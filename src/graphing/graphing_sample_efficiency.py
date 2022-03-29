# DATA: https://drive.google.com/file/d/1DUrerhXX8Q9LMpXqxK_gtwpmxJJKNfzo/view?usp=sharing
data_dict_2 = {
    16: [(10000, 0.42339998483657800),
        (1000, 0.20589999854564700),
        (100, 0.14219999313354500)],

    8: [(10000, 0.42829999327659600),
        (1000, 0.39149999618530300),
        (100, 0.311499983072281)],

    4: [(10000, 0.6482999920845030),
        (1000, 0.3594000041484830),
        (100, 0.25949999690055800)],

    2: [(10000, 0.6184999942779540),
        (1000, 0.5410999655723570),
        (100, 0.428099989891052)],

    1: [(10000, 0.7566999793052670),
        (1000, 0.690699994564056),
        (100, 0.583799958229065)],

    0.5:
        [(10000, 0.654500007629395),
         (1000, 0.621299982070923),
         (100, 0.48199999332428)],

    0:
        [(10000, 0.6291),
         (1000, 0.6608999967575070),
         (100, 0.4528999924659730)]
}

data_dict_10 = {
    16: [(10000, 0.6355),
        (1000, 0.3567),
        (100, 0.2308)],

    8: [(10000, 0.7205),
        (1000, 0.4393),
        (100, 0.2786)],

    4: [(10000, 0.8899),
        (1000, 0.8108),
        (100, 0.6111)],

    2: [(10000, 0.9131),
        (1000, 0.7985),
        (100, 0.5875)],

    1: [(10000, 0.9312),
        (1000, 0.8167),
        (100, 0.6147)],

    0.5:
        [(10000, 0.9292),
         (1000, 0.8723),
         (100, 0.5448)],

    0:
        [(10000, 0.7451),
         (1000, 0.7216),
         (100, 0.5648)]
}

colours = {
    0: "darkred",
    0.5: "darkorange",
    1: "goldenrod",
    2: "olivedrab",
    4: "lightseagreen",
    8: "steelblue",
    16: "slateblue",
}

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
lines = []
labels = []
for beta in data_dict_2.keys():
    x = [a for a, _ in data_dict_2[beta]]
    y = [a for _, a in data_dict_2[beta]]
    lines.append(ax1.plot(x, y, label=f"β={beta}", color=colours[beta], linewidth=3))
ax1.set_title("LS Dim. = 2")
ax1.set_ylabel("Test accuracy")
ax1.set_xscale("log")

for beta in data_dict_10.keys():
    x = [a for a, _ in data_dict_10[beta]]
    y = [a for _, a in data_dict_10[beta]]
    ax2.plot(x, y, label=f"β={beta}", color=colours[beta], linewidth=3)
ax2.set_title("LS Dim. = 10")
ax2.set_xscale("log")
ax2.legend()
fig.text(0.5, 0.02, 'Num. Samples (log)', ha='center')

plt.show()
