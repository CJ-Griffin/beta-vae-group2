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
    ["BVAE-483", 125, 20.0, 522.072690625],
    ["BVAE-481", 95, 20.0, 521.8145875],	
    ["BVAE-480", 65, 20.0, 521.154965625],	
    ["BVAE-479", 35, 20.0, 520.775471875],	
    ["BVAE-478", 5, 20.0, 521.643178125],	
    ["BVAE-477", 125, 2.0, 475.36836875],	
    ["BVAE-476", 95, 2.0, 489.350140625],	
    ["BVAE-475", 65, 2.0, 497.317675],	
    ["BVAE-474", 35, 2.0, 515.9836125],	
    ["BVAE-473", 5, 2.0, 521.4261125],	
    ["BVAE-472", 125, 0.2, 416.915884375],	
    ["BVAE-471", 95, 0.2, 426.884865625],	
    ["BVAE-470", 65, 0.2, 431.968184375],	
    ["BVAE-469", 35, 0.2, 446.090015625],	
    ["BVAE-468", 5, 0.2, 501.008259375],	
    ["BVAE-467", 125, 0.02, 390.0424],	
    ["BVAE-466", 95, 0.02, 390.6615125],	
    ["BVAE-465", 65, 0.02, 393.09146875],	
    ["BVAE-464", 35, 0.02, 396.79918125],	
    ["BVAE-463", 5, 0.02, 432.48499375],	
    ["BVAE-462", 125, 0.002, 382.916103125],	
    ["BVAE-461", 95, 0.002, 384.50009375],	
    ["BVAE-460", 65, 0.002, 384.000571875],	
    ["BVAE-459", 35, 0.002, 383.47683125],	
    ["BVAE-458", 5, 0.002, 401.96835625]
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

disentanglement_score_dict = {
    ('BVAE-451', 25.0, 81): float(0.6870),
    ('BVAE-450', 25.0, 27): float(0.6910),
    ('BVAE-449', 25.0, 9): float(0.6680),
    ('BVAE-448', 25.0, 3): float(0.6360),
    ('BVAE-447', 5.0, 81): float(0.8750),
    ('BVAE-446', 5.0, 27): float(0.8780),
    ('BVAE-445', 5.0, 9): float(0.8430),
    ('BVAE-444', 5.0, 3): float(0.6570),
    ('BVAE-443', 1.0, 81): float(0.9810),
    ('BVAE-442', 1.0, 27): float(0.9540),
    ('BVAE-441', 1.0, 9): float(0.8740),
    ('BVAE-440', 1.0, 3): float(0.6640),
    ('BVAE-439', 0.2, 81): float(0.9770),
    ('BVAE-438', 0.2, 27): float(0.9900),
    ('BVAE-437', 0.2, 9): float(0.8250),
    ('BVAE-436', 0.2, 3): float(0.8530),
    ('BVAE-435', 0.04, 81): float(0.9950),
    ('BVAE-434', 0.04, 27): float(0.9830),
    ('BVAE-433', 0.04, 9): float(0.8930),
    ('BVAE-432', 0.04, 3): float(0.8250)
}

img2 = np.empty((len(betas), len(lss)))
for name, ls, b, _ in encoder_losses:
    yind = betas.index(b)
    xind = lss.index(ls)
    img2[yind, xind] = disentanglement_score_dict[(name, b, ls)]
# img2 -= img2.min()
# img2 /= img2.max()
plt.figure(figsize=(5.2, 4))
plt.imshow(img2, cmap="Purples")
plt.yticks(ticks=[i for i in range(len(betas))], labels=betas)
plt.ylabel("Beta")
plt.xticks(ticks=[i for i in range(len(lss))], labels=lss)
plt.xlabel("Latent Space Size")
plt.title("Normalised Disentanglement Scores (Acc)")
plt.colorbar()
plt.show()
