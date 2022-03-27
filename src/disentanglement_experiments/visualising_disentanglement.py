import numpy as np
from matplotlib import pyplot as plt

encoder_losses = [
    ["BVAE-451", 25.0, 81, 499.91318125],
    ["BVAE-450", 25.0, 27, 495.49854375],
    ["BVAE-449", 25.0, 9, 494.9940125],
    ["BVAE-448", 25.0, 3, 494.74219375],
    ["BVAE-447", 5.0, 81, 451.08926875],
    ["BVAE-446", 5.0, 27, 449.052690625],
    ["BVAE-445", 5.0, 9, 445.404075],
    ["BVAE-444", 5.0, 3, 449.887946875],
    ["BVAE-443", 1.0, 81, 408.8562375],
    ["BVAE-442", 1.0, 27, 406.0368375],
    ["BVAE-441", 1.0, 9, 404.264178125],
    ["BVAE-440", 1.0, 3, 426.8773],
    ["BVAE-439", 0.2, 81, 391.818878125],
    ["BVAE-438", 0.2, 27, 394.155884375],
    ["BVAE-437", 0.2, 9, 391.072671875],
    ["BVAE-436", 0.2, 3, 419.77736875],
    ["BVAE-435", 0.04, 81, 385.7251125],
    ["BVAE-434", 0.04, 27, 384.339259375],
    ["BVAE-433", 0.04, 9, 384.979928125],
    ["BVAE-432", 0.04, 3, 418.432175]
]

betas = set()
lss = set()
for _, b, ls, _ in encoder_losses:
    betas.add(b)
    lss.add(ls)

betas = list(betas)
betas.sort()
lss = list(lss)
lss.sort(reverse=True)

img = np.empty((len(lss), len(betas)))

for _, b, ls, loss in encoder_losses:
    xind = betas.index(b)
    yind = lss.index(ls)
    img[yind, xind] = loss

plt.figure(figsize=(5,4))
plt.imshow(img, cmap="plasma")
plt.xticks(ticks=[i for i in range(len(betas))], labels=betas)
plt.xlabel("Beta")
plt.yticks(ticks=[i for i in range(len(lss))], labels=lss)
plt.ylabel("Latent Space Size")
plt.title("Test Reconstruction + KL Losses for Shapes")
plt.colorbar()
plt.show()

disentanglement_score_dict = {
    ('BVAE-451', 25.0, 81): float(0.2360),
    ('BVAE-450', 25.0, 27): float(0.3680),
    ('BVAE-449', 25.0, 9): float(0.2640),
    ('BVAE-448', 25.0, 3): float(0.2430),
    ('BVAE-447', 5.0, 81): float(0.2640),
    ('BVAE-446', 5.0, 27): float(0.2650),
    ('BVAE-445', 5.0, 9): float(0.2160),
    ('BVAE-444', 5.0, 3): float(0.1140),
    ('BVAE-443', 1.0, 81): float(0.3420),
    ('BVAE-442', 1.0, 27): float(0.2640),
    ('BVAE-441', 1.0, 9): float(0.1980),
    ('BVAE-440', 1.0, 3): float(0.2590),
    ('BVAE-439', 0.2, 81): float(0.2980),
    ('BVAE-438', 0.2, 27): float(0.2950),
    ('BVAE-437', 0.2, 9): float(0.2900),
    ('BVAE-436', 0.2, 3): float(0.2450),
    ('BVAE-435', 0.04, 81): float(0.3750),
    ('BVAE-434', 0.04, 27): float(0.3790),
    ('BVAE-433', 0.04, 9): float(0.2950),
    ('BVAE-432', 0.04, 3): float(0.2780)
}

img2 = np.empty((len(lss), len(betas)))
for name, b, ls, _ in encoder_losses:
    xind = betas.index(b)
    yind = lss.index(ls)
    img2[yind, xind] = disentanglement_score_dict[(name, b, ls)]
# img2 -= img2.min()
# img2 /= img2.max()
plt.figure(figsize=(5,4))
plt.imshow(img2, cmap="Purples")
plt.xticks(ticks=[i for i in range(len(betas))], labels=betas)
plt.xlabel("Beta")
plt.yticks(ticks=[i for i in range(len(lss))], labels=lss)
plt.ylabel("Latent Space Size")
plt.title("Normalised Disentanglement Scores (Acc)")
plt.colorbar()
plt.show()
