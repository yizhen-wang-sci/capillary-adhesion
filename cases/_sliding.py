# import numpy.random as random

# from a_package.modelling import Region, wavevector_norm, SelfAffineRoughness, PSD_to_height, CapillaryBridge
# from a_package.routine import simulate_quasi_static_pull_push, simulate_quasi_static_slide
# from a_package.solving import AugmentedLagrangian
# from a_package.storing import working_directory
# from a_package.visualizing import *


# seed = random.SeedSequence().entropy
# print(f"seed = {seed}")
# rng = random.default_rng(seed)


# if __name__ == "__main__":
#     # modelling parameters
#     a = 1.0         # pixel size
#     N = 128         # num of nodes along one axis
#     L = N * a       # lateral size
#     V = L**2 * a * 1.5    # volume of the droplet

#     eta = a       # scaling interface width
#     gamma = 0.5     # surface tensions: (SL - SG) / LG

#     # the region where simulation runs
#     region = Region(L, L, N, N)

#     # generate rough plates
#     C0 = 1e5  # prefactor
#     qR = (2*np.pi/L) * 5  # roll-off
#     qS = (2*np.pi/L) * 50  # cut-off
#     H = 0.95  # Hurst exponent
#     roughness = SelfAffineRoughness(C0, qR, qS, H)
#     q_2D = wavevector_norm(region.qx, region.qy)
#     C_2D = roughness.mapto_psd(q_2D)
#     h1 = PSD_to_height(C_2D, rng=rng)
#     h2 = PSD_to_height(C_2D, rng=rng)

#     # combine into the model object
#     capi = CapillaryBridge(region, eta, gamma, h1, h2, z1=1.5*eta)
#     capi.update_gap()

#     # Check the preset volume
#     capi.phi = np.ones_like(h1)
#     capi.update_phase_field()
#     assert capi.volume > V, "The preset volume is too large"

#     # A plot to check
#     data = DropletData(region, eta, h1, h2, capi.displacement, capi.g, capi.phi)
#     fig, ax = plt.subplots()
#     # im = plot_height_topography(ax, data)
#     im = plot_gap_topography(ax, data)
#     plot_contact_topography(ax, data)
#     fig.colorbar(im)
#     plt.show()

#     skip = input("Run simulation [Y/n]? ")[:1].upper() == "N"
#     if skip:
#         quit()

#     # solving parameters
#     k_max = 3000
#     e_conv = 1e-8
#     e_volume = 1e-6
#     c_init = 1e-2
#     c_upper = 1e4
#     beta = 3.0
#     solver = AugmentedLagrangian(k_max, e_conv, e_volume, c_init, c_upper, beta)

#     # random phase field to start
#     phi = rng.random((N, N))

#     # run simulation routine
#     n_pixel_to_slide = N
#     n_step_per_pixel = 2
#     n_step = n_step_per_pixel * n_pixel_to_slide
#     l_step = 1 / n_step_per_pixel
#     m_track = np.column_stack((np.arange(n_step)*l_step, np.zeros(n_step)))

#     path = __file__.replace(".py", ".data")
#     with working_directory(path, read_only=False) as store:
#         store.brand_new()
#         simulate_quasi_static_slide(store, capi, solver, V, m_track)
