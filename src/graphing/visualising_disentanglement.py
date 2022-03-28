import numpy as np
from matplotlib import pyplot as plt

# encoder_losses = [
#     ["BVAE-451", 25.0, 81, 499.91318125],
#     ["BVAE-450", 25.0, 27, 495.49854375],
#     ["BVAE-449", 25.0, 9, 494.9940125],
#     ["BVAE-448", 25.0, 3, 494.74219375],
#     ["BVAE-447", 5.0, 81, 451.08926875],
#     ["BVAE-446", 5.0, 27, 449.052690625],
#     ["BVAE-445", 5.0, 9, 445.404075],
#     ["BVAE-444", 5.0, 3, 449.887946875],
#     ["BVAE-443", 1.0, 81, 408.8562375],
#     ["BVAE-442", 1.0, 27, 406.0368375],
#     ["BVAE-441", 1.0, 9, 404.264178125],
#     ["BVAE-440", 1.0, 3, 426.8773],
#     ["BVAE-439", 0.2, 81, 391.818878125],
#     ["BVAE-438", 0.2, 27, 394.155884375],
#     ["BVAE-437", 0.2, 9, 391.072671875],
#     ["BVAE-436", 0.2, 3, 419.77736875],
#     ["BVAE-435", 0.04, 81, 385.7251125],
#     ["BVAE-434", 0.04, 27, 384.339259375],
#     ["BVAE-433", 0.04, 9, 384.979928125],
#     ["BVAE-432", 0.04, 3, 418.432175]
# ]

encoder_losses = [
    ["BVAE-536", 125, 20.0, 520.798675],
    ["BVAE-535", 95, 20.0, 521.521721875],
    ["BVAE-534", 65, 20.0, 520.56343125],
    ["BVAE-533", 35, 20.0, 522.0039125],
    ["BVAE-532", 5, 20.0, 519.933903125],
    ["BVAE-531", 125, 2.0, 489.8647125],
    ["BVAE-530", 95, 2.0, 488.3576875],
    ["BVAE-529", 65, 2.0, 498.512490625],
    ["BVAE-528", 35, 2.0, 516.51175625],
    ["BVAE-527", 5, 2.0, 522.0947125],
    ["BVAE-526", 125, 0.2, 417.497771875],
    ["BVAE-525", 95, 0.2, 422.1374625],
    ["BVAE-524", 65, 0.2, 430.363921875],
    ["BVAE-523", 35, 0.2, 447.60311875],
    ["BVAE-522", 5, 0.2, 502.1202875],
    ["BVAE-521", 125, 0.02, 389.507353125],
    ["BVAE-520", 95, 0.02, 392.26689375],
    ["BVAE-519", 65, 0.02, 394.15695625],
    ["BVAE-518", 35, 0.02, 397.153140625],
    ["BVAE-517", 5, 0.02, 431.162540625],
    ["BVAE-516", 125, 0.002, 382.112053125],
    ["BVAE-515", 95, 0.002, 383.497165625],
    ["BVAE-514", 65, 0.002, 384.380553125],
    ["BVAE-513", 35, 0.002, 383.190796875],
    ["BVAE-512", 5, 0.002, 401.998540625],
]

print([n for n, _, _, _ in encoder_losses])

betas = set()
lss = set()
for _, ls, b, _ in encoder_losses:
    betas.add(b)
    lss.add(ls)

betas = list(betas)
betas.sort(reverse=True)
lss = list(lss)
lss.sort()

img = np.empty((len(betas), len(lss)))

for _, ls, b, loss in encoder_losses:
    yind = betas.index(b)
    xind = lss.index(ls)
    img[yind, xind] = loss

plt.figure(figsize=(5.2, 4))
plt.imshow(img, cmap="plasma")
plt.yticks(ticks=[i for i in range(len(betas))], labels=betas)
plt.ylabel("Beta")
plt.xticks(ticks=[i for i in range(len(lss))], labels=lss)
plt.xlabel("Latent Space Size")
plt.title("Test Reconstruction + KL Losses for Shapes")
plt.colorbar()
plt.show()

# disentanglement_score_dict = {
#     ('BVAE-451', 25.0, 81): float(0.2360),
#     ('BVAE-450', 25.0, 27): float(0.3680),
#     ('BVAE-449', 25.0, 9): float(0.2640),
#     ('BVAE-448', 25.0, 3): float(0.2430),
#     ('BVAE-447', 5.0, 81): float(0.2640),
#     ('BVAE-446', 5.0, 27): float(0.2650),
#     ('BVAE-445', 5.0, 9): float(0.2160),
#     ('BVAE-444', 5.0, 3): float(0.1140),
#     ('BVAE-443', 1.0, 81): float(0.3420),
#     ('BVAE-442', 1.0, 27): float(0.2640),
#     ('BVAE-441', 1.0, 9): float(0.1980),
#     ('BVAE-440', 1.0, 3): float(0.2590),
#     ('BVAE-439', 0.2, 81): float(0.2980),
#     ('BVAE-438', 0.2, 27): float(0.2950),
#     ('BVAE-437', 0.2, 9): float(0.2900),
#     ('BVAE-436', 0.2, 3): float(0.2450),
#     ('BVAE-435', 0.04, 81): float(0.3750),
#     ('BVAE-434', 0.04, 27): float(0.3790),
#     ('BVAE-433', 0.04, 9): float(0.2950),
#     ('BVAE-432', 0.04, 3): float(0.2780)
# }

# disentanglement_score_dict = {
#     ('BVAE-451', 25.0, 81): float(0.6870),
#     ('BVAE-450', 25.0, 27): float(0.6910),
#     ('BVAE-449', 25.0, 9): float(0.6680),
#     ('BVAE-448', 25.0, 3): float(0.6360),
#     ('BVAE-447', 5.0, 81): float(0.8750),
#     ('BVAE-446', 5.0, 27): float(0.8780),
#     ('BVAE-445', 5.0, 9): float(0.8430),
#     ('BVAE-444', 5.0, 3): float(0.6570),
#     ('BVAE-443', 1.0, 81): float(0.9810),
#     ('BVAE-442', 1.0, 27): float(0.9540),
#     ('BVAE-441', 1.0, 9): float(0.8740),
#     ('BVAE-440', 1.0, 3): float(0.6640),
#     ('BVAE-439', 0.2, 81): float(0.9770),
#     ('BVAE-438', 0.2, 27): float(0.9900),
#     ('BVAE-437', 0.2, 9): float(0.8250),
#     ('BVAE-436', 0.2, 3): float(0.8530),
#     ('BVAE-435', 0.04, 81): float(0.9950),
#     ('BVAE-434', 0.04, 27): float(0.9830),
#     ('BVAE-433', 0.04, 9): float(0.8930),
#     ('BVAE-432', 0.04, 3): float(0.8250)
# }

disentanglement_score_dict = {
    ('BVAE-512', 0.002, 5): float(0.6990), ('BVAE-513', 0.002, 35): float(0.9960),
    ('BVAE-514', 0.002, 65): float(0.9990), ('BVAE-515', 0.002, 95): float(0.9990), ('BVAE-516', 0.002, 125): float(1.),
    ('BVAE-517', 0.02, 5): float(0.8740), ('BVAE-518', 0.02, 35): float(0.9860), ('BVAE-519', 0.02, 65): float(0.9940),
    ('BVAE-520', 0.02, 95): float(0.9990), ('BVAE-521', 0.02, 125): float(0.9910), ('BVAE-522', 0.2, 5): float(0.7020),
    ('BVAE-523', 0.2, 35): float(0.9940), ('BVAE-524', 0.2, 65): float(0.9880), ('BVAE-525', 0.2, 95): float(0.9830),
    ('BVAE-526', 0.2, 125): float(0.9930), ('BVAE-527', 2.0, 5): float(0.3530), ('BVAE-528', 2.0, 35): float(0.8110),
    ('BVAE-529', 2.0, 65): float(0.9290), ('BVAE-530', 2.0, 95): float(0.9460), ('BVAE-531', 2.0, 125): float(0.9320),
    ('BVAE-532', 20, 5): float(0.2490), ('BVAE-533', 20, 35): float(0.2860), ('BVAE-534', 20, 65): float(0.2670),
    ('BVAE-535', 20, 95): float(0.2930),
    ('BVAE-536', 20, 125): float(0.4560)
}

img2 = np.empty((len(betas), len(lss)))
for name, ls, b, _ in encoder_losses:
    yind = betas.index(b)
    xind = lss.index(ls)
    img2[yind, xind] = disentanglement_score_dict[(name, b, ls)]

plt.figure(figsize=(5.2, 4))
plt.imshow(img2, cmap="Purples")
plt.yticks(ticks=[i for i in range(len(betas))], labels=betas)
plt.ylabel("Beta")
plt.xticks(ticks=[i for i in range(len(lss))], labels=lss)
plt.xlabel("Latent Space Size")
plt.title("Disentanglement Scores (%Acc)")
plt.colorbar()
plt.show()
