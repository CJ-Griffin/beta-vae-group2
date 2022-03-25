# DATA: https://drive.google.com/file/d/1DUrerhXX8Q9LMpXqxK_gtwpmxJJKNfzo/view?usp=sharing
data_dict_2 = {
    1: [(10000, 0.7441999912261960),
        (1000, 0.675599992275238),
        (100, 0.5203999876976010)],

    4: [(10000, 0.7382000088691710),
        (1000, 0.7109999656677250),
        (100, 0.586899995803833)],

    16:
        [(10000, 0.6715999841690060),
         (1000, 0.6078999638557430),
         (100, 0.43939998745918300)]
}

data_dict_10 = {
    1: [(10000, 0.929899990558624),
        (1000, 0.869899988174439),
        (100, 0.645799994468689)],

    4: [(10000, 0.934099972248077),
        (1000, 0.802599966526032),
        (100, 0.576600015163422)],

    16:
        [(10000, 0.93369996547699),
         (1000, 0.866499960422516),
         (100, 0.611199975013733)]
}
colours = {
    1: "darkred",
    4: "darkorange",
    16: "goldenrod"
}

import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))

for beta in data_dict_2.keys():
    x = [a for a,_ in data_dict_2[beta]]
    y = [a for _,a in data_dict_2[beta]]
    plt.plot(x, y, label=f"β={beta}", color=colours[beta], linewidth=3)
plt.title("Sample efficiency by beta")
plt.xlabel("Num. Samples (log)")
plt.ylabel("Test accuracy")
plt.xscale("log")
plt.legend()
plt.show()
